/*****************************************************************************
FILE:  bdac.cpp
AUTHOR:	Patrick S. Hamilton
REVISED:	5/13/2002
  ___________________________________________________________________________

bdac.cpp: Beat Detection And Classification
Copywrite (C) 2001 Patrick S. Hamilton

This file is free software; you can redistribute it and/or modify it under
the terms of the GNU Library General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option) any
later version.

This software is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Library General Public License for more
details.

You should have received a copy of the GNU Library General Public License along
with this library; if not, write to the Free Software Foundation, Inc., 59
Temple Place - Suite 330, Boston, MA 02111-1307, USA.

You may contact the author by e-mail (pat@eplimited.edu) or postal mail
(Patrick Hamilton, E.P. Limited, 35 Medford St., Suite 204 Somerville,
MA 02143 USA).  For updates to this software, please visit our website
(http://www.eplimited.com).
  __________________________________________________________________________

bdac.cpp contains functions for handling Beat Detection And Classification.
The primary function calls a qrs detector.  When a beat is detected it waits
until a sufficient number of samples from the beat have occurred.  When the
beat is ready, BeatDetectAndClassify passes the beat and the timing
information on to the functions that actually classify the beat.

Functions in bdac.cpp require functions in the following files:
		qrsfilt.cpp
		qrsdet.cpp
		classify.cpp
		rythmchk.cpp
		noisechk.cpp
		analbeat.cpp
		match.cpp
		postclas.cpp

 __________________________________________________________________________

	Revisions:
		5/13/02:
			Encapsulated down sampling from input stream to beat template in
			the function DownSampleBeat.

			Constants related to time are derived from SAMPLE_RATE in qrsdet
         and BEAT_SAMPLE_RATE in bcac.h.

*******************************************************************************/
#include "qrsdet.h"	// For base SAMPLE_RATE
#include "bdac.h"

#include "stdio.h"

#define ECG_BUFFER_LENGTH	1000	// Should be long enough for a beat
											// plus extra space to accommodate
											// the maximum detection delay.
#define BEAT_QUE_LENGTH	10			// Length of que for beats awaiting
											// classification.  Because of
											// detection delays, Multiple beats
											// can occur before there is enough data
											// to classify the first beat in the que.

#include "config.h"
#include "tsc_x86.h"

// Internal function prototypes.

void DownSampleBeat(float *beatOut, float *beatIn) ;

// External function prototypes.

void QRSDet( float* datum, int* delayArray, int sampleLength, int init ) ;
int NoiseCheck(float datum, int delay, int RR, int beatBegin, int beatEnd) ;
int Classify(float *newBeat,int rr, int noiseLevel, int *beatMatch, int *fidAdj, int init) ;
int GetDominantType(void) ;
int GetBeatEnd(int type) ;
int GetBeatBegin(int type) ;
int gcd(int x, int y) ;

// Global variables
float BeatBuffer[BEATLGTH] ;

#if (BDAC_OPT==0)
float ECGBuffer[ECG_BUFFER_LENGTH];  // Circular data buffer.
int BeatQue[BEAT_QUE_LENGTH];  // Buffer of detection delays.
int ECGBufferIndex = 0  ;
#endif

#if(BDAC_OPT==1)
// PRECOMPUTE

#define CHECK_BEAT_QUE_0 (BEATLGTH-FIDMARK)*(SAMPLE_RATE/BEAT_SAMPLE_RATE)
#define SAMPLE_RATE_RATIO SAMPLE_RATE/BEAT_SAMPLE_RATE
#define SAMPLE_RATE_RATIO_FIDMARK (SAMPLE_RATE/BEAT_SAMPLE_RATE)*FIDMARK
#define SAMPLE_RATE_RATIO_BEATLGTH (SAMPLE_RATE/BEAT_SAMPLE_RATE)*BEATLGTH

#endif

#if (AVX_OPT==1)
#include <immintrin.h>
#endif



/******************************************************************************
	ResetBDAC() resets static variables required for beat detection and
	classification.
*******************************************************************************/

void ResetBDAC(void)
{
	int dummy;
	float fDummy = 0.0f;
	int outDummy = 0;
	QRSDet(&fDummy, &outDummy, 1,1) ;	// Reset the qrs detector
	RRCount = 0 ;
	Classify(BeatBuffer,0,0,&dummy,&dummy,1) ;
	InitBeatFlag = 1 ;
	BeatQueCount = 0 ;	// Flush the beat que.
}

/*****************************************************************************
Syntax:
	int BeatDetectAndClassify(int ecgSample, int *beatType, *beatMatch) ;
Description:
	BeatDetectAndClassify() implements a beat detector and classifier.
	ECG samples are passed into BeatDetectAndClassify() one sample at a
	time.  BeatDetectAndClassify has been designed for a sample rate of
	200 Hz.  When a beat has been detected and classified the detection
	delay is returned and the beat classification is returned through the
	pointer *beatType.  For use in debugging, the number of the template
   that the beat was matched to is returned in via *beatMatch.
Returns
	BeatDetectAndClassify() returns 0 if no new beat has been detected and
	classified.  If a beat has been classified, BeatDetectAndClassify returns
	the number of samples since the approximate location of the R-wave.
****************************************************************************/

#if(BDAC_OPT==1)

void BeatDetectAndClassify(float* ecgSample, int* delayArray, int sampleLength, int* beatType, int *beatMatch)
{
	// int detectDelay[MAIN_BLOCK_SIZE];
	int rr, i, j ;
	int noiseEst = 0, beatBegin, beatEnd ;
	int domType ;
	int fidAdj ;
	float tempBeat[SAMPLE_RATE_RATIO_BEATLGTH] ;

	static float ECGBuffer[ECG_BUFFER_LENGTH];  // Circular data buffer.
	static int BeatQue[BEAT_QUE_LENGTH];  // Buffer of detection delays.
	static int ECGBufferIndex = 0  ;

  //	FILE *fp;


	// Run the sample through the QRS detector.
	#ifdef RUNTIME_QRSDET
		start_QRSDet = start_tsc();
	#endif

	// delayArray[repetition] = QRSDet(ecgSample[repetition],0) ;
	QRSDet(ecgSample, delayArray, sampleLength, 0);

	#ifdef RUNTIME_QRSDET
		end_QRSDet += stop_tsc(start_QRSDet);
	#endif

	
	for(int repetition = 0; repetition < sampleLength; repetition++){
	  
		// Store new sample in the circular buffer.
		ECGBuffer[ECGBufferIndex] = ecgSample[repetition];
		if(++ECGBufferIndex == ECG_BUFFER_LENGTH)
			ECGBufferIndex = 0 ;

		// Increment RRInterval count.

		++RRCount ;

		// Increment detection delays for any beats in the que.

		for(i = 0; i < BeatQueCount; ++i)
			++BeatQue[i] ;
			
		if(delayArray[repetition] != 0)
			{
			BeatQue[BeatQueCount] = delayArray[repetition] ;
			++BeatQueCount ;
			}

		// Return if no beat is ready for classification.

		if((BeatQue[0] < CHECK_BEAT_QUE_0) //(BEATLGTH-FIDMARK)*(SAMPLE_RATE/BEAT_SAMPLE_RATE)
			|| (BeatQueCount == 0))
			{
			NoiseCheck(ecgSample[repetition],0,rr, beatBegin, beatEnd) ;	// Update noise check buffer
			delayArray[repetition] = 0 ;
			continue;
			}

		// Otherwise classify the beat at the head of the que.

		rr = RRCount - BeatQue[0] ;	// Calculate the R-to-R interval
		delayArray[repetition] = RRCount = BeatQue[0] ;

		// Estimate low frequency noise in the beat.
		// Might want to move this into classify().

		domType = GetDominantType() ;
		if(domType == -1)
			{
			beatBegin = MS250 ;
			beatEnd = MS300 ;
			}
		else
			{
			beatBegin = (SAMPLE_RATE_RATIO)*(FIDMARK-GetBeatBegin(domType)) ;
			beatEnd = (SAMPLE_RATE_RATIO)*(GetBeatEnd(domType)-FIDMARK) ;
			}
		noiseEst = NoiseCheck(ecgSample[repetition],delayArray[repetition],rr,beatBegin,beatEnd) ;

		// Copy the beat from the circular buffer to the beat buffer
		// and reduce the sample rate by averageing pairs of data
		// points.

		j = ECGBufferIndex - delayArray[repetition] - SAMPLE_RATE_RATIO_FIDMARK ;
		if(j < 0) j += ECG_BUFFER_LENGTH ;

		for(i = 0; i < SAMPLE_RATE_RATIO_BEATLGTH; ++i)
			{
			tempBeat[i] = ECGBuffer[j] ;
			if(++j == ECG_BUFFER_LENGTH)
				j = 0 ;
			}

		//printf("%d\n", BEATLGTH); // 100
#if(AVX_OPT==1)
		printf("BDAC OPT AND AVX OPT\n");
		__m256 scalar05 = _mm256_set1_ps(0.5);
		for(i = 0; i < BEATLGTH/16; i+=16)
		{
		  
		  //BeatBuffer[i] = (tempBeat[i*2]+tempBeat[(i*2)+1])/2 ;

			__m256 tempBeat_tmp_a = _mm256_load_ps(&tempBeat[i*2]);
			__m256 tempBeat_tmp_b = _mm256_load_ps(&tempBeat[i*2+16]);
			__m256 adj_sum = _mm256_hadd_ps(tempBeat_tmp_a, tempBeat_tmp_b);
			__m256 permBeatBuffer_tmp = _mm256_mul_ps(adj_sum, scalar05);
			__m256 BeatBuffer_tmp = _mm256_shuffle_ps(permBeatBuffer_tmp, permBeatBuffer_tmp, _MM_SHUFFLE(0,2,1,3));
			_mm256_store_ps(&BeatBuffer[i], BeatBuffer_tmp);

			
			#ifdef OPERATION_COUNTER 
			float_add_counter++;
			float_div_counter++;
			#endif
		}
		for(;i<BEATLGTH;i++){
		  BeatBuffer[i] = (tempBeat[i*2]+tempBeat[(i*2)+1])/2 ;
		}
		  
		
#else
		
		for(i = 0; i < BEATLGTH; ++i)
		{
		  
			BeatBuffer[i] = (tempBeat[i*2]+tempBeat[(i*2)+1])/2 ;
			
			#ifdef OPERATION_COUNTER 
			float_add_counter++;
			float_div_counter++;
			#endif
		}
		
#endif		

		// Update the QUE.

		

		//for(i = 0; i < BeatQueCount-1; ++i)
		//	BeatQue[i] = BeatQue[i+1] ;
		--BeatQueCount ;
		//printf("%d\n", BeatQueCount); // it's always 0


		// Skip the first beat.

		if(InitBeatFlag)
			{
			InitBeatFlag = 0 ;
			*beatType = 13 ;
			*beatMatch = 0 ;
			fidAdj = 0 ;
			}
		// Classify all other beats.
		else
			{
			#ifdef RUNTIME_CLASSIFY
			start_Classify = start_tsc();
			#endif

			*beatType = Classify(BeatBuffer,rr,noiseEst,beatMatch,&fidAdj,0) ;

			#ifdef RUNTIME_CLASSIFY
			end_Classify += stop_tsc(start_Classify);
			#endif

			fidAdj *= SAMPLE_RATE_RATIO;
	      }

		// Ignore detection if the classifier decides that this
		// was the trailing edge of a PVC.

		if(*beatType == 100)
			{
			RRCount += rr ;
			delayArray[repetition] = 0 ;
			continue;
			}

		// Limit the fiducial mark adjustment in case of problems with
		// beat onset and offset estimation.

		if(fidAdj > MS80)
			fidAdj = MS80 ;
		else if(fidAdj < -MS80)
			fidAdj = -MS80 ;

		delayArray[repetition] = delayArray[repetition] - fidAdj;
	}
	return;
}

#else

void BeatDetectAndClassify(float* ecgSample, int* delayArray, int sampleLength, int* beatType, int *beatMatch)
{
	// int detectDelay[MAIN_BLOCK_SIZE];
	int rr, i, j ;
	int noiseEst = 0, beatBegin, beatEnd ;
	int domType ;
	int fidAdj ;
	float tempBeat[(SAMPLE_RATE/BEAT_SAMPLE_RATE)*BEATLGTH] ;

  //	FILE *fp;


	// Run the sample through the QRS detector.
	#ifdef RUNTIME_QRSDET
		start_QRSDet = start_tsc();
	#endif

	// delayArray[repetition] = QRSDet(ecgSample[repetition],0) ;
	QRSDet(ecgSample, delayArray, sampleLength, 0);

	#ifdef RUNTIME_QRSDET
		end_QRSDet += stop_tsc(start_QRSDet);
	#endif

	for(int repetition = 0; repetition < sampleLength; repetition++){	
		// Store new sample in the circular buffer.
		ECGBuffer[ECGBufferIndex] = ecgSample[repetition];
		if(++ECGBufferIndex == ECG_BUFFER_LENGTH)
			ECGBufferIndex = 0 ;

		// Increment RRInterval count.

		++RRCount ;

		// Increment detection delays for any beats in the que.

		for(i = 0; i < BeatQueCount; ++i)
			++BeatQue[i] ;
			
		if(delayArray[repetition] != 0)
			{
			  //printf("here\n");

	                //fp = fopen("./to_plot/QRSDetDelay100.csv", "a+");
	                //fprintf(fp, "%d,\n", detectDelay);
		        //fclose(fp);
			
			BeatQue[BeatQueCount] = delayArray[repetition] ;
			++BeatQueCount ;
			}

		// Return if no beat is ready for classification.

		if((BeatQue[0] < (BEATLGTH-FIDMARK)*(SAMPLE_RATE/BEAT_SAMPLE_RATE))
			|| (BeatQueCount == 0))
			{
			NoiseCheck(ecgSample[repetition],0,rr, beatBegin, beatEnd) ;	// Update noise check buffer
			delayArray[repetition] = 0 ;
			continue;
			}

		// Otherwise classify the beat at the head of the que.

		rr = RRCount - BeatQue[0] ;	// Calculate the R-to-R interval
		delayArray[repetition] = RRCount = BeatQue[0] ;

		// Estimate low frequency noise in the beat.
		// Might want to move this into classify().

		domType = GetDominantType() ;
		if(domType == -1)
			{
			beatBegin = MS250 ;
			beatEnd = MS300 ;
			}
		else
			{
			beatBegin = (SAMPLE_RATE/BEAT_SAMPLE_RATE)*(FIDMARK-GetBeatBegin(domType)) ;
			beatEnd = (SAMPLE_RATE/BEAT_SAMPLE_RATE)*(GetBeatEnd(domType)-FIDMARK) ;
			}
		noiseEst = NoiseCheck(ecgSample[repetition],delayArray[repetition],rr,beatBegin,beatEnd) ;

		// Copy the beat from the circular buffer to the beat buffer
		// and reduce the sample rate by averageing pairs of data
		// points.

		j = ECGBufferIndex - delayArray[repetition] - (SAMPLE_RATE/BEAT_SAMPLE_RATE)*FIDMARK ;
		if(j < 0) j += ECG_BUFFER_LENGTH ;

		for(i = 0; i < (SAMPLE_RATE/BEAT_SAMPLE_RATE)*BEATLGTH; ++i)
			{
			tempBeat[i] = ECGBuffer[j] ;
			if(++j == ECG_BUFFER_LENGTH)
				j = 0 ;
			}

		for(i = 0; i < BEATLGTH; ++i)
		{
			BeatBuffer[i] = (tempBeat[i*2]+tempBeat[(i*2)+1])/2 ;
			#ifdef OPERATION_COUNTER 
			float_add_counter++;
			float_div_counter++;
			#endif
		}

		// Update the QUE.

		for(i = 0; i < BeatQueCount-1; ++i)
			BeatQue[i] = BeatQue[i+1] ;
		--BeatQueCount ;


		// Skip the first beat.

		if(InitBeatFlag)
			{
			InitBeatFlag = 0 ;
			*beatType = 13 ;
			*beatMatch = 0 ;
			fidAdj = 0 ;
			}
		// Classify all other beats.
		else
			{
			#ifdef RUNTIME_CLASSIFY
			start_Classify = start_tsc();
			#endif

			*beatType = Classify(BeatBuffer,rr,noiseEst,beatMatch,&fidAdj,0) ;

			#ifdef RUNTIME_CLASSIFY
			end_Classify += stop_tsc(start_Classify);
			#endif

			fidAdj *= SAMPLE_RATE/BEAT_SAMPLE_RATE ;
	      }

		// Ignore detection if the classifier decides that this
		// was the trailing edge of a PVC.

		if(*beatType == 100)
			{
			RRCount += rr ;
			delayArray[repetition] = 0 ;
			continue;
			}

		// Limit the fiducial mark adjustment in case of problems with
		// beat onset and offset estimation.

		if(fidAdj > MS80)
			fidAdj = MS80 ;
		else if(fidAdj < -MS80)
			fidAdj = -MS80 ;

		delayArray[repetition] = delayArray[repetition] - fidAdj;
	}
	return;
}

#endif

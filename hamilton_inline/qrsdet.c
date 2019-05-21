/*****************************************************************************
FILE:  qrsdet.cpp
AUTHOR:	Patrick S. Hamilton
REVISED:	12/04/2000
  ___________________________________________________________________________

qrsdet.cpp: A QRS detector.
Copywrite (C) 2000 Patrick S. Hamilton

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

This file contains functions for detecting QRS complexes in an ECG.  The
QRS detector requires filter functions in qrsfilt.cpp and parameter
definitions in qrsdet.h.  QRSDet is the only function that needs to be
visable outside of these files.

Syntax:
	int QRSDet(int ecgSample, int init) ;

Description:
	QRSDet() implements a modified version of the QRS detection
	algorithm described in:

	Hamilton, Tompkins, W. J., "Quantitative investigation of QRS
	detection rules using the MIT/BIH arrhythmia database",
	IEEE Trans. Biomed. Eng., BME-33, pp. 1158-1165, 1987.

	Consecutive ECG samples are passed to QRSDet.  QRSDet was
	designed for a 200 Hz sample rate.  QRSDet contains a number
	of static variables that it uses to adapt to different ECG
	signals.  These variables can be reset by passing any value
	not equal to 0 in init.

	Note: QRSDet() requires filters in QRSFilt.cpp

Returns:
	When a QRS complex is detected QRSDet returns the detection delay.

****************************************************************/

/* For memmove. */
#ifdef __STDC__
#include <string.h>
#else
#include <mem.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "qrsdet.h"
#define PRE_BLANK	MS200


#include "config.h"
#include "tsc_x86.h"

#ifdef OPERATION_COUNTER 
	extern long int float_add_counter;
	extern long int float_mul_counter;
	extern long int float_div_counter;
        extern long int float_comp_counter;
#endif

#define QRSFILT_LOOP_UNROLL
#define QRSDET_LOOP_UNROLL

#define QRS_SCALAR
// Local Prototypes.

void QRSFilter(float* datum, float* output, int sampleLength, int init) ;
void qrsfilt_opt1(float* input, float* output, int samples_to_process, int init) ;

float Peak( float datum, int init ) ;
float median(float *array) ; // Xia: called many times
float thresh(float qmedian, float nmedian) ; // Xia: called many times


double TH = 0.475  ;

//float DDBuffer[DER_DELAY];/* Buffer holding derivative data. */
int Dly  = 0 ;
//int DDPtr ;	

const int MEMMOVELEN = 7*sizeof(int);

// void QRSDet_Init( float* datum, int* delayArray, int sampleLength)
// {
// 	//QRSDet init
// 	for(i = 0; i < 8; ++i)
// 	{
// 		noise[i] = 0.0 ;	/* Initialize noise buffer */
// 		rrbuf[i] = MS1000_FLOAT ;/* and R-to-R interval buffer. */
// 	}
// 	maxder=lastmax= initMax= 0.0;
// 	qpkcnt  = count = sbpeak = 0 ;
// 	initBlank = preBlankCnt = 0; // DDPtr = 0 ;
// 	sbcount = MS1500_FLOAT ;
	
// 	max = 0.0;
// 	timeSinceMax = 0;


// 	//QRSfilt init
// 	// ------- initialize filters ------- //

// 	//lpfilt
// 	for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
// 		lp_data[i_init] = 0.f;

// 	//hpfilt
// 	for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
// 		hp_data[i_init] = 0.f;

// 	//derivative
// 	for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
// 		derBuff[i_init] = 0 ;
	
// 	//movint window integration
// 	for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
// 		data[i_init] = 0 ;
// }


void QRSDet( float* datum, int* delayArray, int sampleLength, int init )
	{
	static float qrsbuf[8], rsetBuff[8] ;
	static int rsetCount = 0 ;
	static float nmedian, qmedian, rrmedian, det_thresh, tempPeak ;
	static int sbloc ;
	
	int i ;
	float fdatum[sampleLength];
	float newPeak, aPeak;

	// ---------- Peak ---------- //
	static float lastDatum ;

	/*	Initialize all buffers to 0 on the first call.	*/

	#if INIT_INLINE == 0
		if( init )
		{
			#ifndef QRSDET_LOOP_UNROLL
			for(i = 0; i < 8; ++i)
			{
				noise[i] = 0.0 ;	/* Initialize noise buffer */
				rrbuf[i] = MS1000_FLOAT ;/* and R-to-R interval buffer. */
			}
			#endif

			#ifdef QRSDET_LOOP_UNROLL
			noise[0] = 0.0;
			noise[1] = 0.0;
			noise[2] = 0.0;
			noise[3] = 0.0;
			noise[4] = 0.0;
			noise[5] = 0.0;
			noise[6] = 0.0;
			noise[7] = 0.0;
			
			rrbuf[0] = MS1000_FLOAT ;
			rrbuf[1] = MS1000_FLOAT ;
			rrbuf[2] = MS1000_FLOAT ;
			rrbuf[3] = MS1000_FLOAT ;
			rrbuf[4] = MS1000_FLOAT ;
			rrbuf[5] = MS1000_FLOAT ;
			rrbuf[6] = MS1000_FLOAT ;
			rrbuf[7] = MS1000_FLOAT ;

			#endif


			maxder=lastmax= initMax= 0.0;
			qpkcnt  = count = sbpeak = 0 ;
			initBlank = preBlankCnt = 0; // DDPtr = 0 ;
			sbcount = MS1500_FLOAT ;
			float dummyData = 0.0f;
			float dummyOut;
			#if QRSFILT_OPT == 0
				QRSFilter(&dummyData,&dummyOut,1,1) ;	/* initialize filters. */
			#endif
			#if QRSFILT_OPT == 1
				qrsfilt_opt1(&dummyData,&dummyOut,1,1);
			#endif
			
			//Peak(0.0,1) ; -- initialize Peak variables
			max = 0.0;
			timeSinceMax = 0;
			
			for(int index = 0; index < sampleLength; index++){ // should only ever be entered with sampleLenght = 1, but use loop to be sure
				delayArray[index] = 0;
			}
		}
	#endif



	#ifdef RUNTIME_QRSDET
		start_QRSFilt = start_tsc();
	#endif
	
	#if QRSFILT_OPT == 0
		QRSFilter(datum, fdatum, sampleLength,0) ;	/* Filter data. */
	#endif
	#if QRSFILT_OPT == 1
		qrsfilt_opt1(datum, fdatum, sampleLength,0);
	#endif

	#ifdef RUNTIME_QRSDET
		end_QRSFilt += stop_tsc(start_QRSFilt);
	#endif

	for(int index = 0; index < sampleLength; index++){
		int QrsDelay = 0 ;
		int pk = 0 ;
		/* Wait until normal detector is ready before calling early detections. */

	
		/**************************************************************
		* peak() takes a datum as input and returns a peak height
		* when the signal returns to half its peak height, or 
		**************************************************************/

		if(timeSinceMax > 0)
			++timeSinceMax ;

		if((fdatum[index] > lastDatum) && (fdatum[index] > max))
			{
			max = fdatum[index] ;
			if(max > 2)
				timeSinceMax = 1 ;
			#ifdef OPERATION_COUNTER 
				float_comp_counter+=3;
			#endif
			}

		else if(fdatum[index] < (max/2))
			{
			#ifdef OPERATION_COUNTER
			float_div_counter++;
			float_comp_counter+=3; // 2 from the previous if, 1 from this if
			#endif
			
			pk = max ;
			max = 0 ;
			timeSinceMax = 0 ;
			Dly = 0 ;
			}

		else if(timeSinceMax > MS95)
			{
			#ifdef OPERATION_COUNTER
			  float_div_counter++; // from the previous else if
			  float_comp_counter+=4; // 3 from previous if's, 1 from this one
			#endif
			
			pk = max ;
			max = 0 ;
			timeSinceMax = 0 ;
			Dly = 3 ;
			}
		#ifdef OPERATION_COUNTER
		else{
		        float_div_counter++; // in the previous else if
			float_comp_counter+=4; // all the previous if's
			}
		#endif
		
		lastDatum = fdatum[index] ;
		aPeak = pk;

		// --- end Peak --- //
		

		// Hold any peak that is detected for 200 ms
		// in case a bigger one comes along.  There
		// can only be one QRS complex in any 200 ms window.

		newPeak = 0.0 ;
		if(aPeak && !preBlankCnt)			// If there has been no peak for 200 ms
			{										// save this one and start counting.
			tempPeak = aPeak ;
			preBlankCnt = PRE_BLANK ;			// MS200
			}

		else if(!aPeak && preBlankCnt)	// If we have held onto a peak for
			{										// 200 ms pass it on for evaluation.
			if(--preBlankCnt == 0)
				newPeak = tempPeak ;
			}

		else if(aPeak)							// If we were holding a peak, but
			{										// this ones bigger, save it and
	                #ifdef OPERATION_COUNTER
			  float_comp_counter++;
			#endif
			if(aPeak > tempPeak)				// start counting to 200 ms again.
				{
				tempPeak = aPeak ;
				preBlankCnt = PRE_BLANK ; // MS200
				}
			else if(--preBlankCnt == 0)
				newPeak = tempPeak ;
			}


		/* Initialize the qrs peak buffer with the first eight 	*/
		/* local maximum peaks detected.						*/

		if( qpkcnt < 8 )
			{
			++count ;
			#ifdef OPERATION_COUNTER
			  float_comp_counter++;
			#endif
			if(newPeak > 0.0){
			  count = WINDOW_WIDTH ;
			}
			if(++initBlank == MS1000)
				{
				initBlank = 0 ;
				qrsbuf[qpkcnt] = initMax ;
				initMax = 0.0 ;
				++qpkcnt ;
				if(qpkcnt == 8)
					{
					qmedian = median(qrsbuf) ;
					nmedian = 0.0 ;
					rrmedian = MS1000_FLOAT ;
					sbcount = MS1500_FLOAT+MS1500_FLOAT ;
					det_thresh = thresh(qmedian,nmedian) ;
					}
				}
			#ifdef OPERATION_COUNTER
			  float_comp_counter++;
			#endif
			if( newPeak > initMax )
				initMax = newPeak ;
			}

		else	/* Else test for a qrs. */
			{
			++count ;
			#ifdef OPERATION_COUNTER
			  float_comp_counter++;
			#endif
			if(newPeak > 0.0)
				{
			           
					// Classify the beat as a QRS complex
					// if the peak is larger than the detection threshold.
                    #ifdef OPERATION_COUNTER
                	float_comp_counter++;
	                #endif

					if(newPeak > det_thresh)
						{
						memmove(&qrsbuf[1], qrsbuf, MEMMOVELEN) ;
						qrsbuf[0] = newPeak ;
						qmedian = median(qrsbuf) ;
						det_thresh = thresh(qmedian,nmedian) ;
						memmove(&rrbuf[1], rrbuf, MEMMOVELEN) ;
						rrbuf[0] = (float)(count - WINDOW_WIDTH) ;
						rrmedian = median(rrbuf) ;
						sbcount = rrmedian + (rrmedian/2) + WINDOW_WIDTH_FLOAT ;
						count = WINDOW_WIDTH ;

						#ifdef OPERATION_COUNTER
						float_add_counter+=3; // assuming the numbers converted to float are first coneverted and then computed
						float_div_counter++;
						#endif

						sbpeak = 0 ;

						lastmax = maxder ;
						maxder = initMax = 0.0 ;
						QrsDelay =  WINDOW_WIDTH + FILTER_DELAY ;
						initBlank = rsetCount = 0 ;

				//		preBlankCnt = PRE_BLANK ;
						}

					// If a peak isn't a QRS update noise buffer and estimate.
					// Store the peak for possible search back.


					else
						{
						memmove(&noise[1],noise,MEMMOVELEN) ;
						noise[0] = newPeak ;
						nmedian = median(noise) ;
						det_thresh = thresh(qmedian,nmedian) ;

						// Don't include early peaks (which might be T-waves)
						// in the search back process.  A T-wave can mask
						// a small following QRS.
	                                        #ifdef OPERATION_COUNTER
			                          float_comp_counter+=2;
			                        #endif
						if((newPeak > sbpeak) && ((count-WINDOW_WIDTH) >= MS360))
							{
							sbpeak = newPeak ;
							sbloc = count  - WINDOW_WIDTH ;
							}
						}
				}
			
			/* Test for search back condition.  If a QRS is found in  */
			/* search back update the QRS buffer and det_thresh.      */
	                #ifdef OPERATION_COUNTER
			  float_comp_counter+=2;
			#endif
			if(((float)count > sbcount) && (sbpeak > (det_thresh/2)))
				{
				memmove(&qrsbuf[1],qrsbuf,MEMMOVELEN) ;
				qrsbuf[0] = (float)sbpeak ;
				qmedian = median(qrsbuf) ;
				det_thresh = thresh(qmedian,nmedian) ;
				memmove(&rrbuf[1],rrbuf,MEMMOVELEN) ;
				rrbuf[0] = (float)sbloc ;
				rrmedian = median(rrbuf) ;
				sbcount = rrmedian + (rrmedian/2) + WINDOW_WIDTH_FLOAT ;
				QrsDelay = count = count - sbloc ;
				QrsDelay += FILTER_DELAY ;
				sbpeak = 0 ;
				lastmax = maxder ;
				maxder = 0 ;
				initBlank = initMax = rsetCount = 0 ;
				}
			}

		// In the background estimate threshold to replace adaptive threshold
		// if eight seconds elapses without a QRS detection.

		if( qpkcnt == 8 )
			{
			if(++initBlank == MS1000)
				{
				initBlank = 0 ;
				rsetBuff[rsetCount] = initMax ;
				initMax = 0 ;
				++rsetCount ;

				// Reset threshold if it has been 8 seconds without
				// a detection.

				if(rsetCount == 8)
					{
					for(i = 0; i < 8; ++i)
						{
						qrsbuf[i] = rsetBuff[i] ;
						noise[i] = 0.0 ;
						}
					qmedian = median( rsetBuff) ;
					nmedian = 0.0 ;
					rrmedian = MS1000_FLOAT ;
					sbcount = MS1500_FLOAT+MS1500_FLOAT ;
					det_thresh = thresh(qmedian,nmedian) ;
					initBlank = initMax = rsetCount = 0 ;
	            sbpeak = 0 ;
					}
				}
			if( newPeak > initMax )
				initMax = newPeak ;
			}

		delayArray[index] = QrsDelay;
	}
	return;
}



/********************************************************************
median returns the median of an array of integers.  It uses a slow
sort algorithm, but these arrays are small, so it hardly matters.
********************************************************************/

float median(float *array)
	{
	int i, j, k;
	float sort[20], temp;
	for(i = 0; i < 8; ++i)
		sort[i] = array[i] ;
	for(i = 0; i < 8; ++i)
		{
		temp = sort[i] ;
		#ifdef OPERATION_COUNTER
		  float_comp_counter++;
		#endif
		for(j = 0; (temp < sort[j]) && (j < i) ; ++j) {
		  #ifdef OPERATION_COUNTER
		    float_comp_counter++;
		  #endif
		  };
		for(k = i - 1 ; k >= j ; --k)
			sort[k+1] = sort[k] ;
		sort[j] = temp ;
		}
	return(sort[4]) ;
	}


/****************************************************************************
 thresh() calculates the detection threshold from the qrs median and noise
 median estimates.
****************************************************************************/

float thresh(float qmedian, float nmedian)
	{
	float thrsh, dmed ;
	double temp ;
	dmed = qmedian - nmedian ;
/*	thrsh = nmedian + (dmed>>2) + (dmed>>3) + (dmed>>4); */
	temp = dmed ;
	temp *= TH ;
	dmed = temp ;
	thrsh = nmedian + dmed ; /* dmed * THRESHOLD */
	#ifdef OPERATION_COUNTER
	float_add_counter+=2;
	float_mul_counter++;
	#endif
	return(thrsh);
	}

#if QRSFILT_OPT == 1

void QRSFilter(float* datum, float* filtOutput, int sampleLength, int init)
{
	for(int index = 0; index < sampleLength; index++)
	{
		// data buffer for lpfilt
		static float lp_data[LPBUFFER_LGTH];

		// data buffer for hpfilt
		static float hp_data[HPBUFFER_LGTH];

		// data buffer for derivative
		static float derBuff[DERIV_LENGTH] ;

		// data buffer for moving window average
		static float data[WINDOW_WIDTH];
	    
		#if INIT_INLINE == 0
		if(init)
		{
		  // #ifndef QRSFILT_LOOP_UNROLL		    Xia: 	Where does this #ifndef end?
			// ------- initialize filters ------- //

			//lpfilt
			#ifndef QRSFILT_LOOP_UNROLL
			for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
				lp_data[i_init] = 0.f;
			#endif

			#ifdef QRSFILT_LOOP_UNROLL
			//LOOP_UNROLL replacement with LPBUFFER_LGTH = 10
			lp_data[0] = 0.f;
			lp_data[1] = 0.f;
			lp_data[2] = 0.f;
			lp_data[3] = 0.f;
			lp_data[4] = 0.f;
			lp_data[5] = 0.f;
			lp_data[6] = 0.f;
			lp_data[7] = 0.f;
			lp_data[8] = 0.f;
			lp_data[9] = 0.f;
			#endif
			//hpfilt
			#ifndef QRSFILT_LOOP_UNROLL
			for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
				hp_data[i_init] = 0.f;
			#endif
			//LOOP_UNROLL  with HPBUFFER_LGTH = 25
			#ifdef QRSFILT_LOOP_UNROLL
			hp_data[0] = 0.f;
			hp_data[1] = 0.f;
			hp_data[2] = 0.f;
			hp_data[3] = 0.f;
			hp_data[4] = 0.f;
			hp_data[5] = 0.f;
			hp_data[6] = 0.f;
			hp_data[7] = 0.f;
			hp_data[8] = 0.f;
			hp_data[9] = 0.f;
			hp_data[10] = 0.f;
			hp_data[11] = 0.f;
			hp_data[12] = 0.f;
			hp_data[13] = 0.f;
			hp_data[14] = 0.f;
			hp_data[15] = 0.f;
			hp_data[16] = 0.f;
			hp_data[17] = 0.f;
			hp_data[18] = 0.f;
			hp_data[19] = 0.f;
			hp_data[20] = 0.f;
			hp_data[21] = 0.f;
			hp_data[22] = 0.f;
			hp_data[23] = 0.f;
			hp_data[24] = 0.f;
			#endif
			//derivative
			#ifndef QRSFILT_LOOP_UNROLL
			for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
				derBuff[i_init] = 0 ;
			#endif
			// LOOP unroll with DERIV_LENGTH	2
			#ifdef QRSFILT_LOOP_UNROLL
			derBuff[0] = 0 ;
			derBuff[1] = 0 ;
			#endif
			//movint window integration
			// LOOP_UNROLL replacement WINDOW_WIDTH	16
			#ifndef QRSFILT_LOOP_UNROLL
			
			for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
				data[i_init] = 0 ;

			#endif
			#ifdef QRSFILT_LOOP_UNROLL
			data[0] = 0 ;
			data[1] = 0 ;
			data[2] = 0 ;
			data[3] = 0 ;
			data[4] = 0 ;
			data[5] = 0 ;
			data[6] = 0 ;
			data[7] = 0 ;
			data[8] = 0 ;
			data[9] = 0 ;
			data[10] = 0 ;
			data[11] = 0 ;
			data[12] = 0 ;
			data[13] = 0 ;
			data[14] = 0 ;
			data[15] = 0 ;
			#endif
			#ifndef QRSFILT_LOOP_UNROLL
				for(int i = 0; i < sampleLength; i++)
				{
					filtOutput[i] = 0;
				}
			#endif
			return;
		}
		#endif

		// ---------- Low pass filter data ---------- //
		// y[n] = 2*y[n-1] - y[n-2] + x[n] - 2*x[t-24 ms] + x[t-48 ms]
		// Note that the filter delay is (LPBUFFER_LGTH/2)-1

		// y[n] = y[n-1] + x[n] - x[n-128 ms]
		// z[n] = x[n-64 ms] - y[n] ;
		// Filter delay is (HPBUFFER_LGTH-1)/2

		// y[n] = x[n] - x[n - 10ms]
		// Filter delay is DERIV_LENGTH/2

		// mvwint() implements a moving window integrator.  Actually, mvwint() averages
		// the signal values over the last WINDOW_WIDTH samples.

		// Xia: I don't replace here y1 and y2 with lp_y1 and lp_y2 because y1 and y2 are used only in lpfilt and nowhere else in this .c file. The same for y0.
	    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
	    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
	    int halfPtr, index;
	    float fdatum, y0, z, y, output;

		halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;	// Use halfPtr to index
		if(halfPtr < 0)							// to x[n-6].
			halfPtr += LPBUFFER_LGTH ;

		y0 = (y1*2.0f) - y2 + datum[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
		y2 = y1;
		y1 = y0;
		fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);
		lp_data[lp_ptr] = datum[index] ;			// Stick most recent sample into
		
		hp_y += fdatum - hp_data[hp_ptr];
		halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;
		if(halfPtr < 0)
			halfPtr += HPBUFFER_LGTH ;
		hp_data[hp_ptr] = fdatum ;
		fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);
		y = fdatum - derBuff[derI] ;
		derBuff[derI] = fdatum;
		fdatum = y;
		fdatum = fabs(fdatum) ;				// Take the absolute value.
		sum += fdatum ;
		sum -= data[ptr] ;
		data[ptr] = fdatum ;

		#ifdef OPERATION_COUNTER
		  float_comp_counter++;
		#endif
		if((sum / (float)WINDOW_WIDTH) > 32000.f)
		{
			output = 32000.f ;
		} 
		else 
		{
			output = sum / (float)WINDOW_WIDTH ;
			#ifdef OPERATION_COUNTER
			float_div_counter += 1;
			#endif
		}

		if(++lp_ptr == LPBUFFER_LGTH)	// the circular buffer and update
			lp_ptr = 0 ;					// the buffer pointer.
		if(++derI == DERIV_LENGTH)
			derI = 0 ;
		if(++hp_ptr == HPBUFFER_LGTH)
			hp_ptr = 0 ;
		if(++ptr == WINDOW_WIDTH)
			ptr = 0 ;
		
		#ifdef OPERATION_COUNTER
			float_add_counter += 10;
			float_mul_counter+=3;
			float_div_counter += 4;
		#endif
		filtOutput[index] = output;
	}
	return;
}

#else



/*****************************************************************************
FILE:  qrsfilt.cpp
AUTHOR:	Patrick S. Hamilton
REVISED:	5/13/2002
  __________________________________________________________________________

	This file includes QRSFilt() and associated filtering files used for QRS

	detection.  Only QRSFilt() and deriv1() are called by the QRS detector
	other functions can be hidden.
	Revisions:
		5/13: Filter implementations have been modified to allow simplified
			modification for different sample rates.
*******************************************************************************/

/******************************************************************************
* Syntax:
*	int QRSFilter(int datum, int init) ;
* Description:
*	QRSFilter() takes samples of an ECG signal as input and returns a sample of
*	a signal that is an estimate of the local energy in the QRS bandwidth.  In
*	other words, the signal has a lump in it whenever a QRS complex, or QRS
*	complex like artifact occurs.  The filters were originally designed for data
*  sampled at 200 samples per second, but they work nearly as well at sample
*	frequencies from 150 to 250 samples per second.
*
*	The filter buffers and static variables are reset if a value other than
*	0 is passed to QRSFilter through init.
*******************************************************************************/
void QRSFilter(float* datum, float* filtOutput, int sampleLength, int init)
	{
	for(int index = 0; index < sampleLength; index++){

		// data buffer for lpfilt
		static float lp_data[LPBUFFER_LGTH];

		// data buffer for hpfilt
		static float hp_data[HPBUFFER_LGTH];

		// data buffer for derivative
		static float derBuff[DERIV_LENGTH] ;

		// data buffer for moving window average
		static float data[WINDOW_WIDTH];
	    
		#if INIT_INLINE == 0
			if(init)
			{
				
				// ------- initialize filters ------- //

				//lpfilt
				for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
					lp_data[i_init] = 0.f;

				//hpfilt
				for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
					hp_data[i_init] = 0.f;

				//derivative
				for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
					derBuff[i_init] = 0 ;
				
				//movint window integration
				for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
					data[i_init] = 0 ;
				
				for(int index = 0; index < sampleLength; index++){
					filtOutput[index] = 0;
				}

				return;
			}
		#endif

	// ---------- Low pass filter data ---------- //

/*************************************************************************
*  lpfilt() implements the digital filter represented by the difference
*  equation:
*
* 	y[n] = 2*y[n-1] - y[n-2] + x[n] - 2*x[t-24 ms] + x[t-48 ms]
*
*	Note that the filter delay is (LPBUFFER_LGTH/2)-1
*
**************************************************************************/
	
		// Xia: I don't replace here y1 and y2 with lp_y1 and lp_y2 because y1 and y2 are used only in lpfilt and nowhere else in this .c file. The same for y0.
	    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
	    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
	    int halfPtr, index;
	    float fdatum, y0, z, y;

		halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;	// Use halfPtr to index
		if(halfPtr < 0)							// to x[n-6].
			halfPtr += LPBUFFER_LGTH ;

		y0 = (y1*2.0f) - y2 + datum[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
		y2 = y1;
		y1 = y0;
		fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);
		lp_data[lp_ptr] = datum[index] ;			// Stick most recent sample into
		if(++lp_ptr == LPBUFFER_LGTH)	// the circular buffer and update
			lp_ptr = 0 ;					// the buffer pointer.

		#ifdef OPERATION_COUNTER
		float_add_counter += 4;
		float_mul_counter += 2;
		#endif
		

		// ---------- High pass filter data ---------- //

/******************************************************************************
*  hpfilt() implements the high pass filter represented by the following
*  difference equation:
*
*	y[n] = y[n-1] + x[n] - x[n-128 ms]
*	z[n] = x[n-64 ms] - y[n] ;
*
*  Filter delay is (HPBUFFER_LGTH-1)/2
******************************************************************************/


		//int halfPtr ;
		
		hp_y += fdatum - hp_data[hp_ptr];
		halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;
		if(halfPtr < 0)
			halfPtr += HPBUFFER_LGTH ;
		hp_data[hp_ptr] = fdatum ;
		fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);
		//hp_data[hp_ptr] = fdatum ;
		if(++hp_ptr == HPBUFFER_LGTH)
			hp_ptr = 0 ;

		#ifdef OPERATION_COUNTER
			float_add_counter += 3;
			float_div_counter += 1;
		#endif

			
		// ---------- Take the derivative ---------- //

/*****************************************************************************
*  deriv1 and deriv2 implement derivative approximations represented by
*  the difference equation:
*
*	y[n] = x[n] - x[n - 10ms]
*
*  Filter delay is DERIV_LENGTH/2
*****************************************************************************/
	// Xia: deriv1 is totally useless: it is actually the same code as deriv2 and
	// it's called only once in qrsdet to assign value to a variable which then is
	// not used at all.


		y = fdatum - derBuff[derI] ;
		derBuff[derI] = fdatum;
		if(++derI == DERIV_LENGTH)
			derI = 0 ;
		fdatum = y;

		#ifdef OPERATION_COUNTER
			float_add_counter += 1;
		#endif	


		// ---------- Take the absolute value ---------- //
		
		fdatum = fabs(fdatum) ;				// Take the absolute value.


		// ----------- Average over an 80 ms window ----------//
	
/*****************************************************************************
* mvwint() implements a moving window integrator.  Actually, mvwint() averages
* the signal values over the last WINDOW_WIDTH samples.
*****************************************************************************/

		float output;

		sum += fdatum ;
		sum -= data[ptr] ;
		data[ptr] = fdatum ;
		if(++ptr == WINDOW_WIDTH)
			ptr = 0 ;
		#ifdef OPERATION_COUNTER
		  float_comp_counter++;
		#endif
		if((sum / (float)WINDOW_WIDTH) > 32000.f){
			output = 32000.f ;
		} else {
			output = sum / (float)WINDOW_WIDTH ;
			#ifdef OPERATION_COUNTER
			float_div_counter += 1;
			#endif
		}

		#ifdef OPERATION_COUNTER
			float_add_counter += 2;
			float_div_counter += 1;
		#endif

		fdatum = output;

		filtOutput[index] = fdatum;
		}
		return;
	}
#endif

void qrsfilt_opt1(float* input, float* output, int samples_to_process, int init) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
#if INIT_INLINE == 0 // INIT_INLINE
	if(init)
	{
	  //		#ifndef QRSFILT_LOOP_UNROLL		Xia: where does this #ifndef end?	
		// ------- initialize filters ------- //

		//lpfilt
#ifndef QRSFILT_LOOP_UNROLL //lpfilt
		for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
			lp_data[i_init] = 0.f;
#endif //QRSFILT_LOOP_UNROLL lpfilt

#ifdef QRSFILT_LOOP_UNROLL //LPBUFFER_LGTH = 10
		//LOOP_UNROLL replacement with LPBUFFER_LGTH = 10
		lp_data[0] = 0.f;
		lp_data[1] = 0.f;
		lp_data[2] = 0.f;
		lp_data[3] = 0.f;
		lp_data[4] = 0.f;
		lp_data[5] = 0.f;
		lp_data[6] = 0.f;
		lp_data[7] = 0.f;
		lp_data[8] = 0.f;
		lp_data[9] = 0.f;
#endif //QRSFILT_LOOP_UNROLL //LPBUFFER_LGTH = 10
		//hpfilt
#ifndef QRSFILT_LOOP_UNROLL //hpfilt
		for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
			hp_data[i_init] = 0.f;
#endif // QRSFILT_LOOP_UNROLL //hpfilt
		
		//LOOP_UNROLL  with HPBUFFER_LGTH = 25
#ifdef QRSFILT_LOOP_UNROLL //HPBUFFER_LGTH = 25
		hp_data[0] = 0.f;
		hp_data[1] = 0.f;
		hp_data[2] = 0.f;
		hp_data[3] = 0.f;
		hp_data[4] = 0.f;
		hp_data[5] = 0.f;
		hp_data[6] = 0.f;
		hp_data[7] = 0.f;
		hp_data[8] = 0.f;
		hp_data[9] = 0.f;
		hp_data[10] = 0.f;
		hp_data[11] = 0.f;
		hp_data[12] = 0.f;
		hp_data[13] = 0.f;
		hp_data[14] = 0.f;
		hp_data[15] = 0.f;
		hp_data[16] = 0.f;
		hp_data[17] = 0.f;
		hp_data[18] = 0.f;
		hp_data[19] = 0.f;
		hp_data[20] = 0.f;
		hp_data[21] = 0.f;
		hp_data[22] = 0.f;
		hp_data[23] = 0.f;
		hp_data[24] = 0.f;
#endif // QRSFILT_LOOP_UNROLL //HPBUFFER_LGTH = 25
		//derivative
#ifndef QRSFILT_LOOP_UNROLL //derivative
		for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
			derBuff[i_init] = 0 ;
#endif // QRSFILT_LOOP_UNROLL //derivative
		// LOOP unroll with DERIV_LENGTH	2
#ifdef QRSFILT_LOOP_UNROLL // DERIV_LENGTH
		derBuff[0] = 0 ;
		derBuff[1] = 0 ;
#endif // QRSFILT_LOOP_UNROLL // DERIV_LENGTH
		
		//movint window integration
		// LOOP_UNROLL replacement WINDOW_WIDTH	16
#ifndef QRSFILT_LOOP_UNROLL // WINDOW_WIDTH
		
		for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
			data[i_init] = 0 ;

#endif // QRSFILT_LOOP_UNROLL // DERIV_LENGTH
		
#ifdef QRSFILT_LOOP_UNROLL // data
		data[0] = 0 ;
		data[1] = 0 ;
		data[2] = 0 ;
		data[3] = 0 ;
		data[4] = 0 ;
		data[5] = 0 ;
		data[6] = 0 ;
		data[7] = 0 ;
		data[8] = 0 ;
		data[9] = 0 ;
		data[10] = 0 ;
		data[11] = 0 ;
		data[12] = 0 ;
		data[13] = 0 ;
		data[14] = 0 ;
		data[15] = 0 ;
#endif // QRSFILT_LOOP_UNROLL // data
		
#ifndef QRSFILT_LOOP_UNROLL //filtOutput
			for(int i = 0; i < sampleLength; i++){
				filtOutput[i] = 0;
			}
#endif // QRSFILT_LOOP_UNROLL //filtOutput
		return;
	}
#endif // INIT_INLINE

    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0, sum_window = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;
    float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);
    int i;
    for(i=0; i < samples_to_process - BLOCKING_SIZE_QRSFILT + 1; i+= BLOCKING_SIZE_QRSFILT)
    {
        for(int j=0; j < BLOCKING_SIZE_QRSFILT; j++)
        {
            index = i + j;
            halfPtr = lp_ptr- lpbuffer_lgth_half ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 * lpbuffer_sqr_div_4;
            lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[hp_ptr];
            halfPtr = hp_ptr- hpbuffer_lgth_half ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            hp_data[hp_ptr] = fdatum ;
            fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
            y = fdatum - derBuff[derI] ;
            derBuff[derI] = fdatum;
            fdatum = y;
            fdatum = fabs(fdatum) ;             // Take the absolute value.
            sum += fdatum - data[ptr] ;
            data[ptr] = fdatum ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            sum_window = sum * window_width_inv;
            if((sum_window) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum_window ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }

            if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr = 0 ;                    // the buffer pointer.
            if(derI == 0)
                derI = 1 ;
            else
                derI = 0 ;
            if(++hp_ptr == HPBUFFER_LGTH)
                hp_ptr = 0 ;
            if(++ptr == WINDOW_WIDTH)
                ptr = 0 ;
            
            // #ifdef OPERATION_COUNTER
            //     float_add_counter += 10;
            //     float_mul_counter++;
            //     float_div_counter += 4;
            // #endif
            output[index] = output_temp;
        }
    }
    for(; i < samples_to_process; i++)
    {
        halfPtr = lp_ptr- lpbuffer_lgth_half ;    // Use halfPtr to index
        if(halfPtr < 0)                         // to x[n-6].
            halfPtr += LPBUFFER_LGTH ;

        y0 = (y1*2.0f) - y2 + input[i] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
        y2 = y1;
        y1 = y0;
        fdatum = y0 * lpbuffer_sqr_div_4;
        lp_data[lp_ptr] = input[i] ;            // Stick most recent sample into
        
        hp_y += fdatum - hp_data[hp_ptr];
        halfPtr = hp_ptr- hpbuffer_lgth_half ;
        if(halfPtr < 0)
            halfPtr += HPBUFFER_LGTH ;
        hp_data[hp_ptr] = fdatum ;
        fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
        y = fdatum - derBuff[derI] ;
        derBuff[derI] = fdatum;
        fdatum = y;
        fdatum = fabs(fdatum) ;             // Take the absolute value.
        sum += fdatum - data[ptr] ;
        data[ptr] = fdatum ;

        // #ifdef OPERATION_COUNTER
        // float_comp_counter++;
        // #endif
        sum_window = sum * window_width_inv;
        if((sum_window) > 32000.f)
        {
            output_temp = 32000.f ;
        } 
        else 
        {
            output_temp = sum_window ;
            // #ifdef OPERATION_COUNTER
            // float_div_counter += 1;
            // #endif
        }

        if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
            lp_ptr = 0 ;                    // the buffer pointer.
        if(derI == 0)
            derI = 1 ;
        else
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
        
        // #ifdef OPERATION_COUNTER
        //     float_add_counter += 10;
        //     float_mul_counter++;
        //     float_div_counter += 4;
        // #endif
        output[i] = output_temp;
    }
}

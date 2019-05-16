#define SAVEFILE 0
#define PRINT 0

#include "stdio.h"
#include "qrsdet.h"		// For sample rate.

#include "ecg_data.h"
#include "bdac.h"

#include "tsc_x86.h"
#include "notch.h"

// External function prototypes.
void ResetBDAC(void) ;
//int BeatDetectAndClassify(int ecgSample, int *beatType, int *beatMatch) ;


// Global variables.

int ADCZero, ADCUnit, InputFileSampleFrequency ;

#ifdef __STDC__
#define MAINTYPE int
#else
#define MAINTYPE void
#endif

MAINTYPE main()
	{	  

	#ifdef OPERATION_COUNTER
	float_add_counter = 0;
	float_mul_counter = 0;
	float_div_counter = 0;
	float_comp_counter = 0;
	#endif

	#ifdef RUNTIME_MEASURE
	start_time = 0;
	end_time = 0;
		#ifdef RUNTIME_QRSDET
		start_QRSDet = 0;
		end_QRSDet = 0;
		#endif
		#ifdef RUNTIME_CLASSIFY
		start_Classify = 0;
		end_Classify = 0;
		#endif
	#endif

	int i, delay;
	float ecg, ecgFilt;
	unsigned char byte ;
	long SampleCount = 0, lTemp, DetectionTime ;
	int beatType, beatMatch ;



		// Initialize beat detection and classification.

		ResetBDAC() ;
		SampleCount = 0 ;
#if SAVEFILE
		FILE *fp;
		fp = fopen("./to_plot/100.csv", "w");
		fprintf(fp, "ecg_data\n");
		fclose(fp);
		fp = fopen("./to_plot/DetectionTime100.csv", "w");
		fprintf(fp, "DetectionTime\n");
		fclose(fp);
#endif
		// Read data from MIT/BIH file until there is none left.
#ifdef FLAME
for (int flame =0; flame < 1000000; flame++)
{
	ResetBDAC() ;
	SampleCount=0;
#endif
		while(SampleCount < N_DATA)
			{
			++SampleCount ;

			// measure only BeatDetectAndClassify and rest not to avoid file opening and closing overhead in performance 
			#ifdef RUNTIME_MEASURE
				start_time = start_tsc();
			#endif

			// Pass sample to beat detection and classification.

			ecg = ecg_data[SampleCount-1];

			slowperformance(&ecg, &ecgFilt, 1); // notch filter 1st implementation

			delay = BeatDetectAndClassify(ecgFilt, &beatType, &beatMatch) ;

			// measure only BeatDetectAndClassify and rest not to avoid file opening and closing overhead in performance 
			#ifdef RUNTIME_MEASURE
				end_time += stop_tsc(start_time);
			#endif

#if SAVEFILE
			fp = fopen("./to_plot/100.csv", "a+");
			fprintf(fp, "%f\n", ecgFilt);
			fclose(fp);
#endif

			// If a beat was detected, annotate the beat location
			// and type.

			if(delay != 0)
				{
				DetectionTime = SampleCount - delay ;
// #if PRINT
// 				printf("DetectionTime %li\n", DetectionTime);
// #endif

#if SAVEFILE
				fp = fopen("./to_plot/DetectionTime100.csv", "a+");
				fprintf(fp, "%ld\n", DetectionTime);
				fclose(fp);
#endif

				}

			}
#ifdef FLAME
}
#endif
	#ifdef OPERATION_COUNTER
		#if PRINT
			printf("float adds:		%li\n", float_add_counter);
			printf("float mult:		%li\n", float_mul_counter);
			printf("float div:		%li\n", float_div_counter);
			printf("float comp:		%li\n", float_comp_counter);
			printf("float total only math:	%li\n", float_div_counter+float_mul_counter+float_add_counter);
			printf("float total with comp:	%li\n", float_div_counter+float_mul_counter+float_add_counter+float_comp_counter);
		#endif
			// TODO: filesave
	#endif

		// TODO if we have finial measurements for one version hadcode the flop values and turn off operation counting to get accurate perormance measurments
	#ifdef RUNTIME_MEASURE
		#if PRINT
			#ifdef RUNTIME_QRSDET
			printf("QRSdet runtime:   %lli\n", end_QRSDet);
			#endif
			#ifdef RUNTIME_CLASSIFY
			printf("Classify runtime: %lli\n", end_Classify);
			#endif
			printf("total runtime:    %lli\n",end_time);
			printf("performance:      %f\n", (double)(float_div_counter+float_mul_counter+float_add_counter)/(double)end_time);
			printf("performance (w/ comp):      %f\n", (double)(float_div_counter+float_mul_counter+float_add_counter+float_comp_counter)/(double)end_time);
		#endif
		// TODO: filesave
	#endif

	}


/*****************************************************************************
FILE:  easytest.cpp
AUTHOR:	Patrick S. Hamilton
REVISED:	5/13/2002 (PSH); 4/10/2003 (GBM)
  ___________________________________________________________________________

easytest.cpp: Use bdac to generate an annotation file.
Copyright (C) 2001 Patrick S. Hamilton
Copyright (C) 1999 George B. Moody

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

Easytest.exe is a simple program to help test the performance of our
beat detection and classification software. Data is read from the
indicated ECG file, the channel 1 signal is fed to bdac.c, and the
resulting detections are saved in the annotation file <record>.ate.
<record>.ate may then be compared to <record>.atr to using bxb to
analyze the performance of the the beat detector and classifier detector.

Note that data in the MIT/BIH Arrythmia database file has been sampled
at 360 samples-per-second, but the beat detection and classification
software has been written for data sampled at 200 samples-per-second.
Date is converterted from 360 sps to 200 sps with the function NextSample.
Code for resampling was copied from George Moody's xform utility.  The beat
locations are then adjusted back to coincide with the original sample
rate of 360 samples/second so that the annotation files generated by
easytest can be compared to the "atruth" annotation files.

This file must be linked with object files produced from:
	wfdb software library (source available at www.physionet.org)
	analbeat.cpp
	match.cpp
	rythmchk.cpp
	classify.cpp
	bdac.cpp
	qrsfilt.cpp
	qrsdet.cpp
	postclass.cpp
	noisechk.cpp
  __________________________________________________________________________

  Revisions
	4/13/02:
		Added conditional define statements that allow MIT/BIH or AHA
			records to be processed.
		Normalize input to 5 mV/LSB (200 A-to-D units/mV).

	4/10/03:
		Moved definitions of Record[] array, ECG_DB_PATH, and REC_COUNT
			into "input.h"
*******************************************************************************/


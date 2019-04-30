/*
 * main.c
 *
 *  Created on: Apr 4, 2019
 *      Author: xiaywang
 */


#include "stdio.h"
#include "qrsdet.h"		// For sample rate.

#include "ecg_data.h"

#include "ecgcodes.h"

// External function prototypes.
void ResetBDAC(void) ;
int BeatDetectAndClassify(int ecgSample, int *beatType, int *beatMatch) ;

// Local Prototypes.
int  NextSample(int *vout,int nosig,int ifreq,
						int ofreq,int init) ;
int gcd(int x, int y);

// Global variables.

int ADCZero, ADCUnit, InputFileSampleFrequency ;

int main(){

	int i, delay, ecg;
	long SampleCount = 0, lTemp, DetectionTime ;
	int beatType, beatMatch ;

	ADCZero = 1024;
	ADCUnit = 200;
	InputFileSampleFrequency = 360;

	// Initialize sampling frequency adjustment.
	//printf("1 here?\n");

	ecg = ecg_data[0];

	//NextSample(&ecg,1,InputFileSampleFrequency,SAMPLE_RATE,1) ;

	//printf("2 here?\n");

	// Initialize beat detection and classification.

	ResetBDAC() ;
	SampleCount = 0 ;

	// Read data from MIT/BIH file until there is none left.

	for(i=1; i<LENGTH; i++)
		{
		++SampleCount ;

		//ecg = ecg_data[i]*1000;

		//printf("%d\n", ecg);

		//NextSample(&ecg,1,InputFileSampleFrequency,SAMPLE_RATE,0);

		//printf("ecg %d\n", ecg*1000);

		// Set baseline to 0 and resolution to 5 mV/lsb (200 units/mV)

		//lTemp = ecg-ADCZero ;
		//lTemp *= 200 ;			lTemp /= ADCUnit ;			ecg = lTemp ;

		// Pass sample to beat detection and classification.
		//ecg = ecg_data[i]*1000;
		//printf("ecg %d\n", ecg);
		ecg = ecg_data[i];
		delay = BeatDetectAndClassify(ecg, &beatType, &beatMatch) ;
		printf("delay %d\n", delay);
		//printf("ecg %d and samplecount %d, delay %d\n",ecg, SampleCount, delay);

		// If a beat was detected, annotate the beat location
		// and type.

		if(delay != 0)
			{
			DetectionTime = SampleCount - delay ;
			
			fp = fopen("DetectionTime100.csv", "a+");
			fprintf(fp, "%ld,\n", DetectionTime);
			fclose(fp);

			// Convert sample count to input file sample
			// rate.

			//DetectionTime *= InputFileSampleFrequency ;
			//DetectionTime /= SAMPLE_RATE ;
			//printf("det time %ld\n", DetectionTime) ;
			//annot.anntyp = beatType ;
			//annot.aux = NULL ;
			//putann(0,&annot) ;
			}
		}

	return 0;



	/*
	int i, delay;
	int beatType, beatMatch ;
//	long SampleCount = 0, lTemp, DetectionTime ;

	ADCZero = 1024;
	ADCUnit = 200;
	InputFileSampleFrequency = 360;

	WFDB_Annotation annot ;

	ResetBDAC() ;

	QRSDet(0,1);
	for(i=0; i<LENGTH; i++){

		//printf("ecg data %d\n", (int)(ecg_data[i]*1000));

		//delay = QRSDet((int)(ecg_data[i]*1000), 0);
		delay = BeatDetectAndClassify(ecg_data[i]*1000, &beatType, &beatMatch) ;
		//printf("ecg data %f\n", ecg_data[i]*1000);
		printf("%d\n", delay);

	}

	return 0;
	*/

	}

/**********************************************************************
	NextSample reads MIT/BIH Arrhythmia data from a file of data
	sampled at ifreq and returns data sampled at ofreq.  Data is
	returned in vout via *vout.  NextSample must be initialized by
	passing in a nonzero value in init.  NextSample returns -1 when
   there is no more data left.
***********************************************************************/

int  NextSample(int *vout,int nosig,int ifreq,
						int ofreq,int init)
	{
	int i ;
	static int m, n, mn, ot, it, vv[WFDB_MAXSIG], v[WFDB_MAXSIG], rval ;
	rval =2;

	if(init)
		{
		i = gcd(ifreq, ofreq);
		m = ifreq/i;
		n = ofreq/i;
		mn = m*n;
		ot = it = 0 ;
		//getvec(vv) ;
		//rval = getvec(v) ;
		}

	else
		{
		while(ot > it)
			{

	    	for(i = 0; i < nosig; ++i)
	    		vv[i] = v[i] ;
			//rval = getvec(v) ;
		    if (it > mn) { it -= mn; ot -= mn; }
		    it += n;
		    }
	    for(i = 0; i < nosig; ++i)
	    	vout[i] = vv[i] + (ot%n)*(v[i]-vv[i])/n;
		ot += m;
		}
	//printf("here?\n");
	return(rval) ;
	}

// Greatest common divisor of x and y (Euclid's algorithm)

int gcd(int x, int y)
	{
	while (x != y) {
		if (x > y) x-=y;
		else y -= x;
		}
	return (x);
	}


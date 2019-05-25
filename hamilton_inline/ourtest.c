#define SAVEFILE 0
#define PRINT 1

#include "stdio.h"
#include "qrsdet.h"		// For sample rate.

#include "ecg_data.h"
#include "bdac.h"

#include "tsc_x86.h"

// External function prototypes.
void ResetBDAC(void) ;
// int BeatDetectAndClassify(int ecgSample, int *beatType, int *beatMatch) ;


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
	float ecg;
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
for (int flame =0; flame < 1; flame++)
{
	SampleCount=0;
	ResetBDAC() ;
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

			delay = BeatDetectAndClassify(ecg, &beatType, &beatMatch) ;

			// measure only BeatDetectAndClassify and rest not to avoid file opening and closing overhead in performance 
			#ifdef RUNTIME_MEASURE
				end_time += stop_tsc(start_time);
			#endif

#if SAVEFILE
			fp = fopen("./to_plot/100.csv", "a+");
			fprintf(fp, "%f\n", ecg);
			fclose(fp);
#endif

			// If a beat was detected, annotate the beat location
			// and type.

			if(delay != 0)
				{
				DetectionTime = SampleCount - delay ;
//#if PRINT
				//printf("DetectionTime %li\n", DetectionTime);
//#endif

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
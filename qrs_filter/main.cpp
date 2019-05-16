/**
*      _________   _____________________  ____  ______
*     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
*    / /_  / /| | \__ \ / / / /   / / / / / / / __/
*   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
*  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
*
*  http://www.inf.ethz.ch/personal/markusp/teaching/
*  How to Write Fast Numerical Code 263-2300 - ETH Zurich
*  Copyright (C) 2017  Alen Stojanov      (astojanov@inf.ethz.ch)
*                      Georg Ofenbeck     (ofenbeck@inf.ethz.ch)
*                      Singh Gagandeep    (gsingh@inf.ethz.ch)
*	                Markus Pueschel    (pueschel@inf.ethz.ch)
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program. If not, see http://www.gnu.org/licenses/.
*/
//#include "stdafx.h"

#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <random>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "tsc_x86.h"
#include "matlab_data.h"

// extern float qrsfilt_input[used_samples];
// extern float qrsfilt_output[used_samples];

/* prototype of the function you need to optimize */
typedef void(*comp_func)(float *, float *, int);

#define cost_analysis 20.0
#define CYCLES_REQUIRED 1e7
#define REP 30
#define MAX_FUNCS 32
#define FLOPS cost_analysis*used_samples //TODO needs cost_analysis
#define EPS (1e-3)

using namespace std;

//headers
double get_perf_score(comp_func f);
void register_functions();
double perf_test(comp_func f, string desc, int flops);


void slowperformance(float* input, float* output, int sample_length);

void add_function(comp_func f, string name, int flop);

/* Global vars, used to keep track of student functions */
vector<comp_func> userFuncs;
vector<string> funcNames;
vector<int> funcFlops;
int numFuncs = 0;


/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions()
{
	add_function(&slowperformance, "Slow Performance", 12);
	// Add your functions here
	// add_function(&your_function, "function: Optimization X", flops per iteration);
	// add_function(&blocking, "Blocking", 12);
}

double nrm_sqr_diff(float *x, float *y, int n) {
    float nrm_sqr = 0.0;
    for(int i = 0; i < n; i++) {
    	// printf("%f matlab %f\n",x[i],y[i] );
        nrm_sqr += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return nrm_sqr;
}

/*
* Main driver routine - calls register_funcs to get student functions, then
* tests all functions registered, and reports the best performance
*/
int main(int argc, char **argv)
{
	cout << "Starting program. ";
	double perf;
	double maxPerf = 0;
	int i;
	int maxInd = 0;
	int verbosity = 2;

	//set n for amount of samples wanted (max is sample_length == 120'000)
	//giving 10min of samples
	int n = used_samples;
	float filter_input[used_samples];
	memcpy(filter_input, qrsfilt_input, used_samples * sizeof(float));


	register_functions();

	if (numFuncs == 0)
	{
        cout << endl;
		cout << "No functions registered - nothing for driver to do" << endl;
		cout << "Register functions by calling register_func(f, name)" << endl;
		cout << "in register_funcs()" << endl;

		return 0;
	}
	cout << numFuncs << " functions registered." << endl;
   
    //Check validity of functions. 
    float output[used_samples];
    float output_old[used_samples];

    //initialize to 0 (first few output samples will be 0, depending on #coefficients)
    for (int i = 0; i < used_samples; i++)
    {
    	output_old[i] = 0.0;
    }

	for (i = 0; i < numFuncs; i++)
	{
		memcpy(output, output_old, used_samples*sizeof(float));
		comp_func f = userFuncs[i];
		f(filter_input, output, i);
		double error = nrm_sqr_diff(output, qrsfilt_output, used_samples);
		if (error > EPS)
			cout << "ERROR!!!!  the results for the " << i << "th function are different to the previous" << std::endl;
	}



	for (i = 0; i < numFuncs; i++)
	{
        cout << endl << "Running: " << funcNames[i] << endl;
		perf = perf_test(userFuncs[i], funcNames[i], funcFlops[i]);
        // cout << perf << " flops per cycle" << endl;
	}

	return 0;
}


/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/
void add_function(comp_func f, string name, int flops)
{
	userFuncs.push_back(f);
	funcNames.emplace_back(name);
	funcFlops.push_back(flops);

	numFuncs++;
}




/*
* Checks the given function for validity. If valid, then computes and
* reports and returns the number of cycles required per iteration
*/
double perf_test(comp_func f, string desc, int flops)
{
	double cycles = 0.;
	double perf = 0.0;
	long num_runs = 16;
	double multiplier = 1;
	myInt64 start, end;

	//new variables, n for #loop iterations
	float output[used_samples];
	int n, i;

	for (i = 500; i<=used_samples-500; i+=500)
	{
		// Warm-up phase: we determine a number of executions that allows
		// the code to be executed for at least CYCLES_REQUIRED cycles.
		// This helps excluding timing overhead when measuring small runtimes.
		do {
			num_runs = num_runs * multiplier;
			start = start_tsc();
			for (size_t i = 0; i < num_runs; i++) {
				f(qrsfilt_input, output , i);
			}
			end = stop_tsc(start);

			cycles = (double)end;
			multiplier = (CYCLES_REQUIRED) / (cycles);
			
		} while (multiplier > 2);

		list< double > cyclesList, perfList;

		// Actual performance measurements repeated REP times.
		// We simply store all results and compute medians during post-processing.
		// Remark Robin: apparently not using median but using best case
		for (size_t j = 0; j < REP; j++) {

			start = start_tsc();
			for (size_t i = 0; i < num_runs; ++i) {
				f(qrsfilt_input, output , i);
			}
			end = stop_tsc(start);

			cycles = ((double)end) / num_runs;

			cyclesList.push_back(cycles);

			//This does not seem to be used
			perfList.push_back(FLOPS / cycles);
		}
		cyclesList.sort();
		cycles = cyclesList.front();
		printf("%f\n", (cost_analysis * i) / cycles);
	}
	
	return  (cost_analysis * i) / cycles;
}



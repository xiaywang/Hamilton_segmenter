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

#define cost_analysis 18.0
#define cost_analysis_blocking_no_division 16.0
#define cost_analysis_blocking_no_division_fact 15.0
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

void qrsfilt_opt_400_50_1(float* datum, float* filtOutput, int sampleLength);


void slowperformance(float* input, float* output, int samples_to_process);
void slowperformance_macro_test(float* input, float* output, int samples_to_process);
void slowperformance_macro_lp(float* input, float* output, int samples_to_process);
void slowperformance_macro_lp_deri(float* input, float* output, int samples_to_process);
void slowperformance_macro_lp_deri_hp(float* input, float* output, int samples_to_process);
void slowperformance_macro_lp_deri_hp_half(float* input, float* output, int samples_to_process);

void slowperformance_macro_lp_deri_hp_half_dependencies(float* input, float* output, int samples_to_process);
void slowperformance_macro_lp_deri_hp_half_dependencies_div(float* input, float* output, int samples_to_process);

void slowperformance_macro_lp_deri_hp_half_div(float* input, float* output, int samples_to_process);
void slowperformance_macro_lp_deri_hp_half_div_const_replace(float* input, float* output, int samples_to_process);


void slowperformance_macro_lp_deri_hp_ptr(float* input, float* output, int samples_to_process);
void slowperformance_macro_lp_deri_hp_ptr_half(float* input, float* output, int samples_to_process);
void slowperformance_macro_lp_deri_hp_ptr_half_div(float* input, float* output, int samples_to_process);


void slowperformance2(float* datum, float* filtOutput, int sampleLength);
void blocking(float* input, float* output, int samples_to_process);
void blocking_no_divisions1(float* input, float* output, int samples_to_process);
void blocking_no_divisions2(float* input, float* output, int samples_to_process);
void blocking_no_divisions2_derI(float* input, float* output, int samples_to_process);
void blocking_no_divisions2_derI_precomp_sum(float* input, float* output, int samples_to_process);
void no_divisions2_derI_precomp_sum(float* input, float* output, int samples_to_process);
void no_division(float* input, float* output, int samples_to_process);
void blocking_no_divisions_factorized(float* input, float* output, int samples_to_process); 
void blocking_no_divisions_unrolled(float* input, float* output, int samples_to_process); 


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
	// Add your functions here
	// add_function(&your_function, "function: Optimization X", flops per iteration);

	add_function(&slowperformance, "Slow Performance", cost_analysis);
	add_function(&qrsfilt_opt_400_50_1, "RENAME", cost_analysis);

	// add_function(&slowperformance_macro_test, "Slow Performance with macro", cost_analysis);
	// add_function(&slowperformance_macro_lp, "Slow Performance with macro lp", cost_analysis);
	// add_function(&slowperformance_macro_lp_deri, "Slow Performance with macro lp derI", cost_analysis);
	// add_function(&slowperformance_macro_lp_deri_hp, "Slow Performance with macro lp derI hp", cost_analysis);

	// add_function(&slowperformance_macro_lp_deri_hp_half, "Slow Performance with macro lp derI hp half", cost_analysis);
	// add_function(&slowperformance_macro_lp_deri_hp_half_dependencies, "Slow Performance with macro lp derI hp half dependencies", cost_analysis);
	add_function(&slowperformance_macro_lp_deri_hp_half_dependencies_div, "Slow Performance with macro lp derI hp half dependencies no div", cost_analysis);
	// add_function(&slowperformance_macro_lp_deri_hp_half_div, "Slow Performance with macro lp derI hp half no div", cost_analysis);
	// add_function(&slowperformance_macro_lp_deri_hp_half_div_const_replace, "Slow Performance with macro lp derI hp half no div and const replaced", cost_analysis_blocking_no_division);
	
	// add_function(&slowperformance_macro_lp_deri_hp_ptr, "Slow Performance with macro lp derI hp ptr", cost_analysis);
	// add_function(&slowperformance_macro_lp_deri_hp_ptr_half, "Slow Performance with macro lp derI hp ptr half", cost_analysis);
	// add_function(&slowperformance_macro_lp_deri_hp_ptr_half_div, "Slow Performance with macro lp derI hp ptr half div", cost_analysis);

	// add_function(&slowperformance2, "Slow Performance2", cost_analysis);
	

	// add_function(&blocking, "Blocking", cost_analysis);
	// add_function(&blocking_no_divisions1, "Blocking precomp div only", cost_analysis);
	// add_function(&blocking_no_divisions2, "Blocking precomp div and constant", cost_analysis_blocking_no_division);
	// add_function(&blocking_no_divisions2_derI, "Blocking precomp div and constant, derI short", cost_analysis_blocking_no_division);
	// add_function(&blocking_no_divisions2_derI_precomp_sum, "Blocking precomp div and constant, derI short and precomp sum", cost_analysis_blocking_no_division);
	// add_function(&no_divisions2_derI_precomp_sum, "Precomp div and constant, derI short and precomp sum", cost_analysis_blocking_no_division);
	// add_function(&no_division, "Precomp div", cost_analysis_blocking_no_division);
	// add_function(&blocking_no_divisions_factorized, "Blocking precomp div and x*2 -y*2 -> (x-y)*2", cost_analysis_blocking_no_division_fact);
	// add_function(&blocking_no_divisions_unrolled, "Blocking precomp div and unrolled loop by 2", cost_analysis_blocking_no_division);
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
    cout << "testing functions for correctness"<<endl;
	for (i = 0; i < numFuncs; i++)
	{
		memcpy(output, output_old, used_samples*sizeof(float));
		comp_func f = userFuncs[i];
		f(filter_input, output, used_samples);
		double error = nrm_sqr_diff(output, qrsfilt_output, used_samples);
		if (error > EPS)
			cout <<"WRONG! "<< funcNames[i] << " function is WRONG! "<< error << endl;
		else
			cout<<"CORRECT! " << funcNames[i] << " function is CORRECT! "<< error << endl;
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
	double average = 0.0;
	double total_cycles = 0.0;
	double average_count = 0.0;

	//new variables, n for #loop iterations
	float output[used_samples];
	int n, i, input_length;

	for (input_length = 400; input_length<=used_samples-400; input_length+=400)
	{
		// cout << "input_length = " << input_length << endl;
		// Warm-up phase: we determine a number of executions that allows
		// the code to be executed for at least CYCLES_REQUIRED cycles.
		// This helps excluding timing overhead when measuring small runtimes.
		do {
			num_runs = num_runs * multiplier;
			start = start_tsc();
			for (size_t i = 0; i < num_runs; i++) {
				f(qrsfilt_input, output , input_length);
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
				f(qrsfilt_input, output , input_length);
			}
			end = stop_tsc(start);

			cycles = ((double)end) / num_runs;

			cyclesList.push_back(cycles);

			//This does not seem to be used
			perfList.push_back(FLOPS / cycles);
		}
		cyclesList.sort();
		cycles = cyclesList.front();
		// printf("%f\n", (flops * input_length) / cycles);
		total_cycles += cycles;
		average += (flops * input_length) / cycles;
		average_count+= 1;
		// printf("%f cycles\n", cycles);
	}
	printf("average %f flops/cycle\ntotal of %f cycles\n", average/average_count, total_cycles);
	return  (flops * input_length) / cycles;
}



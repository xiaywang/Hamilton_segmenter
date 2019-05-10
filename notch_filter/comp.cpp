#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matlab_data.h"

//
//The slow base version
//
void slowperformance(double* input, double* output, double* filter_coef, int number_of_samples) {
    double x1,x2 = 0; //z-1 buffers
    double y1,y2 = 0; //z-2 buffers
    double temp_x; //temporary store of the "fall-through" value
    double z; //temporary output from first SOS-stage
    for (int i = number_filter_coefficients; i < number_of_samples; i++)
    {
        temp_x = input[i] - x1*filter_coef[4] - y1*filter_coef[5];
        z = temp_x*filter_coef[0] + x1*filter_coef[1] + y1*filter_coef[2];
        y1 = x1;
        x1 = temp_x;
        
        temp_x = z - x2*filter_coef[10] - y2*filter_coef[11];
        output[i] = temp_x*filter_coef[6] + x2*filter_coef[7] + y2*filter_coef[8];
        y2 = x2;
        x2 = temp_x;
    }

}
#include <math.h>
#include "notch.h"
//
//The slow base version
//

#define NUM_COEFFS 12
const float filter_coefficients[NUM_COEFFS] = {1.0, 0.000000228016561187872, 0.999999772567283, 1.0, 0.0230135827512062,0.978067688426724, 1.0, -0.000000228016561854005, 0.999999772566322, 1.0, -0.0230135827512065, 0.978067688426719};

void slowperformance(float* input, float* output, int number_of_samples) {
    static double x1,x2 = 0; //z-1 buffers
    static double y1,y2 = 0; //z-2 buffers
    double temp_x = 0; //temporary store of the "fall-through" value
    double z; //temporary output from first SOS-stage
    for (int i = 0; i < number_of_samples; i++)
    {
        temp_x = input[i] - x1*filter_coefficients[4] - y1*filter_coefficients[5];
        z = temp_x*filter_coefficients[0] + x1*filter_coefficients[1] + y1*filter_coefficients[2];
        y1 = x1;
        x1 = temp_x;
        
        temp_x = z - x2*filter_coefficients[10] - y2*filter_coefficients[11];
        output[i] = temp_x*filter_coefficients[6] + x2*filter_coefficients[7] + y2*filter_coefficients[8];
        y2 = x2;
        x2 = temp_x;
    }

}
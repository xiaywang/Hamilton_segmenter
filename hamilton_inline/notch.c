#include <math.h>
#include "notch.h"
#include "config.h"

// #define NUM_COEFFS 96
// #define NUM_STAGES 16
// const float filter_coefficients[NUM_COEFFS] = {
// 1, 0.0116635379651156, 0.998929556477087, 1, 0.0315388160231767, 0.999288769851840,
// 1, -0.0116635379651149, 0.998929556477085, 1, -0.0315388160231763, 0.999288769851839,
// 1, 0.0113147663094372, 0.996801694892653, 1, 0.0326326934096043, 0.997733506584902, 
// 1, -0.0113147663094371, 0.996801694892653, 1, -0.0326326934096039, 0.997733506584903,
// 1, 0.0106118045250604, 0.994714562887238, 1, 0.0350181380978616, 0.995714490539365,
// 1, -0.0106118045250605, 0.994714562887240, 1, -0.0350181380978617, 0.995714490539365,
// 1, 0.00952491926220556, 0.992716011835832, 1, 0.0391057341173189, 0.992652365085149,
// 1, -0.00952491926220578, 0.992716011835834, 1, -0.0391057341173192, 0.992652365085153,
// 1, 0.00802751779060013, 0.990885296073555, 1, 0.0456435762019372, 0.987267799410590,
// 1, -0.00802751779060035, 0.990885296073555, 1, -0.0456435762019371, 0.987267799410588,
// 1, 0.00611797174053946, 0.989334407309108, 1, 0.0556244306522128, 0.976180883105444,
// 1, -0.00611797174053963, 0.989334407309110, 1, -0.0556244306522122, 0.976180883105447,
// 1, 0.00384306197085294, 0.988194829866335, 1, 0.0671800863601312, 0.949333973614906,
// 1, -0.00384306197085349, 0.988194829866330, 1, -0.0671800863601304, 0.949333973614906,
// 1, 0.00131191621174243, 0.987587999296514, 1, 0.0473559385484363, 0.890118468950657,
// 1, -0.00131191621174276, 0.987587999296512, 1, -0.0473559385484362, 0.890118468950658
// };

// #define NUM_COEFFS 48
// #define NUM_STAGES 8
// const float filter_coefficients[NUM_COEFFS] = {
// 1, 0.00241173321321325, 0.999521469585446, 1, 0.0317461503265100, 0.997252410671320, 
// 1, -0.00241173321321353, 0.999521469585447, 1, -0.0317461503265096, 0.997252410671322, 
// 1, 0.00204641495771973, 0.998636030109813, 1, 0.0346585315136606, 0.989924060772599, 
// 1, -0.00204641495771957, 0.998636030109814, 1, -0.0346585315136610, 0.989924060772603, 
// 1, 0.00136951290519086, 0.997955476831014, 1, 0.0391954839498502, 0.974480285471430, 
// 1, -0.00136951290519094, 0.997955476831010, 1, -0.0391954839498498, 0.974480285471429, 
// 1, 0.000481469430242645, 0.997585509914671, 1, 0.0270191429967540, 0.940922700423501, 
// 1, -0.000481469430242187, 0.997585509914671, 1, -0.0270191429967548, 0.940922700423502
// };

#define NUM_COEFFS 24
#define NUM_STAGES 4
const float filter_coefficients[NUM_COEFFS] = {
1, 0.000105999737667783, 0.999956093608071, 1, 0.0309895216876636, 0.990608187607165,
1, -0.000105999737667672, 0.999956093608070, 1, -0.0309895216876636, 0.990608187607168,
1, 4.39053630995161e-05, 0.999894003419193, 1, 0.0195762207185172, 0.965423306433757,
1, -4.39053631002551e-05, 0.999894003419208, 1, -0.0195762207185178, 0.965423306433758
};

// #define NUM_COEFFS 12
// #define NUM_STAGES NUM_COEFFS/6
// const float filter_coefficients[NUM_COEFFS] = {1.0, 0.000000228016561187872, 0.999999772567283, 1.0, 0.0230135827512062,0.978067688426724, 1.0, -0.000000228016561854005, 0.999999772566322, 1.0, -0.0230135827512065, 0.978067688426719};
#if fastNotch == 1

const double new_order1[16] = {
    filter_coefficients[4], filter_coefficients[5], filter_coefficients[1], filter_coefficients[2],
    filter_coefficients[10], filter_coefficients[11], filter_coefficients[7], filter_coefficients[8],
    filter_coefficients[16], filter_coefficients[17], filter_coefficients[13], filter_coefficients[14],
    filter_coefficients[22], filter_coefficients[23], filter_coefficients[19], filter_coefficients[20]
};

const double new_order2[16] = {
    new_order1[2]-new_order1[0], new_order1[6]-new_order1[4], new_order1[10]-new_order1[8], new_order1[14]-new_order1[12],
    new_order1[3]-new_order1[1], new_order1[7]-new_order1[5], new_order1[11]-new_order1[9], new_order1[15]-new_order1[13],
    filter_coefficients[4], filter_coefficients[10], filter_coefficients[16], filter_coefficients[22],
    filter_coefficients[5], filter_coefficients[11], filter_coefficients[17], filter_coefficients[23],
};

void slowperformance(double* input, double* output, int number_of_samples) {
    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[2*NUM_STAGES] = {0};
    double z[2*NUM_STAGES+1];
    double inp_1, outp_1, inp_2, outp_2;

    int i;

    for(i = 0; i < number_of_samples; i++)
    {
        inp_1 = input[i];
        inp_2 = input[i+1];

        double intermed_0 = x[0]*new_order2[0] + y[0]*new_order2[4];
        double intermed_1 = x[1]*new_order2[1] + y[1]*new_order2[5];
        double intermed_2 = x[2]*new_order2[2] + y[2]*new_order2[6];
        double intermed_3 = x[3]*new_order2[3] + y[3]*new_order2[7];

        temp[0] = inp_1 - x[0]*new_order1[0] - y[0]*new_order1[1];
        temp[1] = inp_1 + intermed_0 - x[1]*new_order1[4] - y[1]*new_order1[5];
        temp[2] = inp_1 + intermed_0 + intermed_1 - x[2]*new_order1[8] - y[2]*new_order1[9];
        temp[3] = inp_1 + intermed_0 + intermed_1 + intermed_2 - x[3]*new_order1[12] - y[3]*new_order1[13];

        outp_1 = inp_1 + intermed_0 + intermed_1 + intermed_2 + intermed_3;

        output[i] = outp_1;

        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
        y[3] = x[3];
        x[0] = temp[0];
        x[1] = temp[1];
        x[2] = temp[2];
        x[3] = temp[3];
    }

}
#else
void slowperformance(float* input, float* output, int number_of_samples) {
    static double x[NUM_STAGES] = {0}; //z-1 buffers
    static double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp_x = 0; //temporary store of the "fall-through" value
    double z; //temporary output from first SOS-stage
    for (int i = 0; i < number_of_samples; i++)
    {
        z = input[i];
        for(int j = 0; j < NUM_STAGES; j++){
            temp_x = z*filter_coefficients[6*j+3] - x[j]*filter_coefficients[6*j+4] - y[j]*filter_coefficients[6*j+5];
            z = temp_x*filter_coefficients[6*j] + x[j]*filter_coefficients[6*j+1] + y[j]*filter_coefficients[6*j+2];
            y[j] = x[j];
            x[j] = temp_x;
        }
        output[i] = z;
    }

}
#endif
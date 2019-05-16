#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
// #include "matlab_data.h"

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

const double new_order[16] = {
    0.0309895216876636, -0.0309895216876636, 0.0195762207185172, -0.0195762207185178, // values 4, 10, 16, 22
    0.990608187607165, 0.990608187607168, 0.965423306433757, 0.965423306433758, // values 5, 11, 17, 23
    0.000105999737667783, -0.000105999737667672, 4.39053630995161e-05, -4.39053631002551e-05, // values 1, 7, 13, 19
    0.999956093608070, 0.999956093608070, 0.999894003419193, 0.999894003419208 // values 2, 8, 14, 20
};

// #define NUM_COEFFS 12
// #define NUM_STAGES NUM_COEFFS/6
// const float filter_coefficients[NUM_COEFFS] = {1.0, 0.000000228016561187872, 0.999999772567283, 1.0, 0.0230135827512062,0.978067688426724, 1.0, -0.000000228016561854005, 0.999999772566322, 1.0, -0.0230135827512065, 0.978067688426719};

// inplementation close to system representation of filter
void slowperformance4(double* input, double* output, int number_of_samples) {
    static double x[NUM_STAGES] = {0}; //z-1 buffers
    static double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[NUM_STAGES] = {0};

    for(int i = 0; i < number_of_samples; i++)
    {
        // temp only for readability not really part of algorithm
        x[0] = input[i] - filter_coefficients[4]*x[0] - filter_coefficients[5]*y[0] ;
        temp[0] = (-filter_coefficients[4]+filter_coefficients[1])*x[0] + (-filter_coefficients[5]+filter_coefficients[2])*y[0];
        x[1] = input[i] + temp[0] + - filter_coefficients[10]*x[1] - filter_coefficients[11]*y[1];
        temp[1] = (-filter_coefficients[10]+filter_coefficients[7])*x[1] + (-filter_coefficients[11]+filter_coefficients[8])*y[1];
        x[2] = input[i] + temp[0] + temp[1] - filter_coefficients[16]*x[1] - filter_coefficients[17]*y[1];
        temp[2] = (-filter_coefficients[16]+filter_coefficients[13])*x[2] + (-filter_coefficients[17]+filter_coefficients[14])*y[2];
        x[3] = input[i] + temp[0] + temp[1] + temp[2] - filter_coefficients[22]*x[1] - filter_coefficients[23]*y[1];
        temp[4] = (-filter_coefficients[22]+filter_coefficients[19])*x[3] + (-filter_coefficients[23]+filter_coefficients[20])*y[3];

        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
        y[3] = x[3];

        output[i] = input[i] + temp[0] + temp[1] + temp[2] + temp[3];
    }

}

// pipelined
void pipelined1(double* input, double* output, int number_of_samples) {
    static double x[NUM_STAGES] = {0}; //z-1 buffers
    static double y[NUM_STAGES] = {0}; //z-2 buffers
    static double z[NUM_STAGES+1] = {0}; // buffers
    static double temp[NUM_STAGES] = {0}; //z-2 buffers
    double temp_0 = 0, temp_1 = 0, temp_2 = 0, temp_3 = 0; //temporary store of the "fall-through" value
    double z_in, z_0, z_1, z_2, z_3; //temporary output from first SOS-stage

    int i;

    __m256d vec_x = _mm256_load_pd(x);
    __m256d vec_y = _mm256_load_pd(y);
    __m256d vec_z0 = _mm256_load_pd(z);
    __m256d vec_z1 = _mm256_load_pd(z);
    __m256d vec_temp0 = _mm256_load_pd(temp);
    __m256d vec_temp1 = _mm256_load_pd(temp);
    __m256d vec_fx1 = _mm256_load_pd(new_order);
    __m256d vec_fx2 = _mm256_load_pd(new_order+8);
    __m256d vec_fy1 = _mm256_load_pd(new_order+4);
    __m256d vec_fy2 = _mm256_load_pd(new_order+12);

    for (i = 0; i < number_of_samples; i++)
    {
        z[0] = input[i];

        vec_z0 = _mm256_load_pd(z);

        // temp[0] = z[0] - x[0]*new_order[0] - y[0]*new_order[4];
        // temp[1] = z[1] - x[1]*new_order[1] - y[1]*new_order[5];
        // temp[2] = z[2] - x[2]*new_order[2] - y[2]*new_order[6];
        // temp[3] = z[3] - x[3]*new_order[3] - y[3]*new_order[7];
        vec_temp0 = _mm256_fnmadd_pd(vec_x, vec_fx1, vec_z0);
        vec_temp1 = _mm256_fnmadd_pd(vec_y, vec_fy1, vec_temp0);

        // z[1] = temp[0] + x[0]*new_order[8] + y[0]*new_order[12];
        // z[2] = temp[1] + x[1]*new_order[9] + y[1]*new_order[13];
        // z[3] = temp[2] + x[2]*new_order[10] + y[2]*new_order[14];
        // z[0] = temp[3] + x[3]*new_order[11] + y[3]*new_order[15];
        vec_z0 = _mm256_fmadd_pd(vec_x, vec_fx2, vec_temp1);
        vec_z1 = _mm256_fmadd_pd(vec_y, vec_fy2, vec_z0);
        vec_z0 = vec_z1;
        _mm256_permute4x64_pd(vec_z0,0x90);

        // x[0] = temp[0];
        // x[1] = temp[1];
        // x[2] = temp[2];
        // x[3] = temp[3];
        vec_x = vec_temp1;

        // y[0] = x[0];
        // y[1] = x[1];
        // y[2] = x[2];
        // y[3] = x[3];
        vec_y = vec_x;

        _mm256_store_pd(z, vec_z0);

        output[i] = z[0];
    }
}

// not using coeficients that are 1 anymore, unrolled for scalar replacement
void slowperformance3(double* input, double* output, int number_of_samples) {
    static double x[NUM_STAGES] = {0}; //z-1 buffers
    static double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[NUM_STAGES] = {0};
    double z[NUM_STAGES+1];

    for(int i = 0; i < number_of_samples; i++)
    {
        z[0] = input[i];

        temp[0] = z[0] - x[0]*new_order[0] - y[0]*new_order[4];
        temp[1] = z[1] - x[1]*new_order[1] - y[1]*new_order[5];
        temp[2] = z[2] - x[2]*new_order[2] - y[2]*new_order[6];
        temp[3] = z[3] - x[3]*new_order[3] - y[3]*new_order[7];

        z[1] = temp[0] + x[0]*new_order[8] + y[0]*new_order[12];
        z[2] = temp[1] + x[1]*new_order[9] + y[1]*new_order[13];
        z[3] = temp[2] + x[2]*new_order[10] + y[2]*new_order[14];
        z[0] = temp[3] + x[3]*new_order[11] + y[3]*new_order[15];

        x[0] = temp[0];
        x[1] = temp[1];
        x[2] = temp[2];
        x[3] = temp[3];

        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
        y[3] = x[3];

        output[i] = z[0];
    }

}

// not using coeficients that are 1 anymore, unrolled for scalar replacement
void slowperformance2(double* input, double* output, int number_of_samples) {
    static double x[NUM_STAGES] = {0}; //z-1 buffers
    static double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp_0 = 0, temp_1 = 0, temp_2 = 0, temp_3 = 0; //temporary store of the "fall-through" value
    double z, z_0, z_1, z_2, z_3; //temporary output from first SOS-stage

    for (int i = 0; i < number_of_samples; i++)
    {
        z = input[i];

        temp_0 = z - x[0]*filter_coefficients[4] - y[0]*filter_coefficients[5];
        z_0 = temp_0 + x[0]*filter_coefficients[1] + y[0]*filter_coefficients[2];
        y[0] = x[0];
        x[0] = temp_0;

        temp_1 = z_0 - x[1]*filter_coefficients[10] - y[1]*filter_coefficients[11];
        z_1 = temp_1 + x[1]*filter_coefficients[7] + y[1]*filter_coefficients[8];
        y[1] = x[1];
        x[1] = temp_1;

        temp_2 = z_1 - x[2]*filter_coefficients[16] - y[2]*filter_coefficients[17];
        z_2 = temp_2 + x[2]*filter_coefficients[13] + y[2]*filter_coefficients[14];
        y[2] = x[2];
        x[2] = temp_2;

        temp_3 = z_2 - x[3]*filter_coefficients[22] - y[3]*filter_coefficients[23];
        z_3 = temp_3 + x[3]*filter_coefficients[19] + y[3]*filter_coefficients[20];
        y[3] = x[3];
        x[3] = temp_3;

        output[i] = z_3;
    }

}


//dumb base implementation
void slowperformance(double* input, double* output, int number_of_samples) {
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
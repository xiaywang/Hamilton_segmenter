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

// #define NUM_COEFFS 12
// #define NUM_STAGES NUM_COEFFS/6
// const float filter_coefficients[NUM_COEFFS] = {1.0, 0.000000228016561187872, 0.999999772567283, 1.0, 0.0230135827512062,0.978067688426724, 1.0, -0.000000228016561854005, 0.999999772566322, 1.0, -0.0230135827512065, 0.978067688426719};

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

const double matrixCoeffs2[32] = {
filter_coefficients[4] , // 0
filter_coefficients[10],
filter_coefficients[16],
filter_coefficients[22],
filter_coefficients[5] ,
filter_coefficients[11],
filter_coefficients[17], // 6
filter_coefficients[23],
0, 
new_order1[2]-new_order1[0], 
0, 
new_order1[10]-new_order1[8], 
0, // 12
new_order1[3]-new_order1[1], 
0,
new_order1[11]-new_order1[9], 
0, 
0, 
new_order1[2]-new_order1[0], // 18
new_order1[6]-new_order1[4],
0, 
0, 
new_order1[3]-new_order1[1], 
new_order1[7]-new_order1[5],
0, 
0, 
new_order1[6]-new_order1[4], 
new_order1[2]-new_order1[0], 
0,
0, 
new_order1[7]-new_order1[5],
new_order1[3]-new_order1[1],
};

void split(double* input, double* output, int number_of_samples){

    double temp[NUM_STAGES] = {0}; //temporary store of the "fall-through" value
    double* z = (double*) aligned_alloc(4*sizeof(double), number_of_samples*sizeof(double)); //temporary output from first SOS-stage
    double* a = (double*) aligned_alloc(4*sizeof(double), number_of_samples*sizeof(double)); //temporary output from first SOS-stage

    double y = 0, x = 0;
    int i;

    for(i = 0; i < number_of_samples; i++)
    {
        a[i] = input[i] - x*new_order1[0] - y*new_order1[1];
        y = x;
        x = a[i];
    }

    z[0] = a[0];
    z[1] = a[1] + a[0]*new_order1[2];
    z[2] = a[2] + a[1]*new_order1[2] + a[0]*new_order1[3];
    z[3] = a[3] + a[2]*new_order1[2] + a[1]*new_order1[3];
    __m256d coef_1 = _mm256_set1_pd(new_order1[2]);
    __m256d coef_2 = _mm256_set1_pd(new_order1[3]);
    __m256d vec2V;
    __m256d vec3V =  _mm256_setr_pd(a[2], a[3], a[0], a[1]);
    // double vec2[4];
    // double vec3[4] = {a[2],a[3],a[0],a[2]};
    // double vec1[4];
    for(i = 4; i < number_of_samples -3; i+=4)
    {
        // double vec1[4] = {a[i],a[i+1],a[i+2],a[i+3]}; // 0 1 2 3
        // double vec1[4] = {a[i],a[i],a[i],a[i]}; // 0 1 2 3
        // double vec0[4] = {a[i-4],a[i-3],a[i-2],a[i-1]}; // -4 -3 -2 -1
        // double vec0[4] = {a[i],a[i],a[i],a[i]};
        // double out[4];
        // vec2[0] = vec3[0];
        // vec2[1] = vec3[1];
        // vec2[2] = vec3[2];
        // vec2[3] = vec3[3]; // -2 -1 -4 -3
        // vec3[0] = vec1[2]; // 2 3 0 1 same as above, but from last run
        // vec3[1] = vec1[3];
        // vec3[2] = vec1[0];
        // vec3[3] = vec1[1];
        // double vec4[4] = {vec2[0],vec2[1],vec3[2],vec3[3]}; // -2 -1 0 1 needed for the second mult
        // double vec5[4] = {vec1[0], vec4[1], vec1[2],vec4[3]}; // 0 -1 2 1
        // double vec6[4] = {vec5[1],vec5[0],vec5[3], vec5[2]}; // -1 0 1 2

        __m256d vec1V = _mm256_load_pd(a+i);
        vec2V = vec3V;
        vec3V = _mm256_permute2f128_pd (vec1V, vec1V, 0x01);
        __m256d vec4V = _mm256_blend_pd(vec2V,vec3V, 0b1100);
        __m256d vec5V = _mm256_blend_pd(vec1V,vec4V, 0b1010);
        __m256d vec6V = _mm256_permute_pd(vec5V,0b0101);
        __m256d intermed = _mm256_mul_pd(vec4V, coef_2);
        __m256d intermed2 = _mm256_mul_pd(vec6V, coef_1);
        __m256d intermed3 = _mm256_add_pd(intermed, vec1V);
        __m256d outV = _mm256_add_pd(intermed3, intermed2);
        _mm256_store_pd(z+i, outV);
        
        // out[0] = vec1[0] + vec6[0]*new_order1[2] + vec4[0]*new_order1[3]; // 0 -1 -2
        // out[1] = vec1[1] + vec6[1]*new_order1[2] + vec4[1]*new_order1[3]; // 1 0 -1
        // out[2] = vec1[2] + vec6[2]*new_order1[2] + vec4[2]*new_order1[3]; // 2 1 0
        // out[3] = vec1[3] + vec6[3]*new_order1[2] + vec4[3]*new_order1[3]; // 3 2 1

        // vec2V = vec3V;
        // z[i] = out[0]; // could also be different
        // z[i+1] = out[1]; // could also be different
        // z[i+2] = out[2]; // could also be different
        // z[i+3] = out[3]; // could also be different
    }

    for(; i < number_of_samples; i++)
    {
        double out = a[i] + a[i-1]*new_order1[2] + a[i-2]*new_order1[3];
        z[i] = out; // could also be different
    }

    x = 0, y = 0;
    for(i = 0; i < number_of_samples; i++)
    {
        double out = z[i] - x*new_order1[4] - y*new_order1[5];
        y = x;
        x = out;
        a[i] = out;
    }

    z[0] = a[0];
    z[1] = a[1] + a[0]*new_order1[6];
    z[2] = a[2] + a[1]*new_order1[6] + a[0]*new_order1[7];
    z[3] = a[3] + a[2]*new_order1[6] + a[1]*new_order1[7];
    coef_1 = _mm256_set1_pd(new_order1[6]);
    coef_2 = _mm256_set1_pd(new_order1[7]);
    vec3V =  _mm256_setr_pd(a[2], a[3], a[0], a[1]);
    for(i = 4; i < number_of_samples - 3; i+=4){
        __m256d vec1V = _mm256_load_pd(a+i);
        vec2V = vec3V;
        vec3V = _mm256_permute2f128_pd (vec1V, vec1V, 0x01);
        __m256d vec4V = _mm256_blend_pd(vec2V,vec3V, 0b1100);
        __m256d vec5V = _mm256_blend_pd(vec1V,vec4V, 0b1010);
        __m256d vec6V = _mm256_permute_pd(vec5V,0b0101);
        __m256d intermed = _mm256_mul_pd(vec4V, coef_2);
        __m256d intermed2 = _mm256_mul_pd(vec6V, coef_1);
        __m256d intermed3 = _mm256_add_pd(intermed, vec1V);
        __m256d outV = _mm256_add_pd(intermed3, intermed2);
        _mm256_store_pd(z+i, outV);
    }
    for(; i < number_of_samples; i++)
    {
        double out = a[i] + a[i-1]*new_order1[6] + a[i-2]*new_order1[7];
        z[i] = out; // could also be different
    }

    x = 0, y = 0;
    for(i = 0; i < number_of_samples; i++)
    {
        double out = z[i] - x*new_order1[8] - y*new_order1[9];
        y = x;
        x = out;
        a[i] = out;
    }

    z[0] = a[0];
    z[1] = a[1] + a[0]*new_order1[10];
    z[2] = a[2] + a[1]*new_order1[10] + a[0]*new_order1[11];
    z[3] = a[3] + a[2]*new_order1[10] + a[1]*new_order1[11];
    coef_1 = _mm256_set1_pd(new_order1[10]);
    coef_2 = _mm256_set1_pd(new_order1[11]);
    vec3V =  _mm256_setr_pd(a[2], a[3], a[0], a[1]);
    for(i = 4; i < number_of_samples - 3; i+=4){
        __m256d vec1V = _mm256_load_pd(a+i);
        vec2V = vec3V;
        vec3V = _mm256_permute2f128_pd (vec1V, vec1V, 0x01);
        __m256d vec4V = _mm256_blend_pd(vec2V,vec3V, 0b1100);
        __m256d vec5V = _mm256_blend_pd(vec1V,vec4V, 0b1010);
        __m256d vec6V = _mm256_permute_pd(vec5V,0b0101);
        __m256d intermed = _mm256_mul_pd(vec4V, coef_2);
        __m256d intermed2 = _mm256_mul_pd(vec6V, coef_1);
        __m256d intermed3 = _mm256_add_pd(intermed, vec1V);
        __m256d outV = _mm256_add_pd(intermed3, intermed2);
        _mm256_store_pd(z+i, outV);
    }
    for(; i < number_of_samples; i++)
    {
        double out = a[i] + a[i-1]*new_order1[10] + a[i-2]*new_order1[11];
        z[i] = out; // could also be different
    }

    x = 0, y = 0;
    for(i = 0; i < number_of_samples; i++)
    {
        double out = z[i] - x*new_order1[12] - y*new_order1[13];
        y = x;
        x = out;
        a[i] = out;
    }

    z[0] = a[0];
    z[1] = a[1] + a[0]*new_order1[14];
    z[2] = a[2] + a[1]*new_order1[14] + a[0]*new_order1[15];
    z[3] = a[3] + a[2]*new_order1[14] + a[1]*new_order1[15];
    coef_1 = _mm256_set1_pd(new_order1[14]);
    coef_2 = _mm256_set1_pd(new_order1[15]);
    vec3V =  _mm256_setr_pd(a[2], a[3], a[0], a[1]);
    for(i = 4; i < number_of_samples - 3; i+=4){
        __m256d vec1V = _mm256_load_pd(a+i);
        vec2V = vec3V;
        vec3V = _mm256_permute2f128_pd (vec1V, vec1V, 0x01);
        __m256d vec4V = _mm256_blend_pd(vec2V,vec3V, 0b1100);
        __m256d vec5V = _mm256_blend_pd(vec1V,vec4V, 0b1010);
        __m256d vec6V = _mm256_permute_pd(vec5V,0b0101);
        __m256d intermed = _mm256_mul_pd(vec4V, coef_2);
        __m256d intermed2 = _mm256_mul_pd(vec6V, coef_1);
        __m256d intermed3 = _mm256_add_pd(intermed, vec1V);
        __m256d outV = _mm256_add_pd(intermed3, intermed2);
        _mm256_store_pd(output+i, outV);
    }
    for(; i < number_of_samples; i++)
    {
        double out = a[i] + a[i-1]*new_order1[14] + a[i-2]*new_order1[15];
        z[i] = out; // could also be different
    }

    free(z);
    free(a);
}

void newHope0(double* input, double* output, int number_of_samples){
    static double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    static double b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    static double c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    static double d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    static double ab0 = 0, ab1 = 0, ab2 = 0, ab3 = 0;
    static double cd0 = 0, cd1 = 0, cd2 = 0, cd3 = 0;
    static double p0 = 0, p1 = 0, p2 = 0, p3 = 0;
    static double q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    static double m0 = 0, m1 = 0, m2 = 0, m3 = 0;
    static double n0 = 0, n1 = 0, n2 = 0, n3 = 0;
    static double o0 = 0, o1 = 0, o2 = 0, o3 = 0;

    for(int i = 0; i < number_of_samples; i++){

        double inp = input[i];
        n0 = m0;
        n1 = m1;
        n2 = m2;
        n3 = m3;

        ab0 = a0 + b0;
        ab1 = a1 + b1;
        ab2 = a2 + b2;
        ab3 = a3 + b3;

        cd0 = c0 + d0;
        cd1 = c1 + d1;
        cd2 = c2 + d2;
        cd3 = c3 + d3;

        p0 = inp - ab0;
        p1 = cd0 - ab1;
        p2 = cd1 - ab2;
        p3 = cd2 - ab3;

        //q1 = p1 + p2;
        //q2 = p3 + p0;
        //q3 = p0 + cd3;

        m0 = p0 ;
        m1 = p0 + p1;
        q3 = p2 + p3;
        m2 = m1 + p2;
        m3 = m1 + q3;

        o3 = m3 + cd3;
        //m0 = p0;
        //m1 = p0 + p1;
        //m1 = p1 + m0;
        //m2 = p0 + q1;
        //m3 = q1 + q2;

        // o0 = m0 + cd0;
        // o1 = m1 + cd1;
        // o2 = m2 + cd2;
        //o3 = q1 + q2 + cd3;

        // here to buffer so results of ms can get ready
        b0 = n0*filter_coefficients[5];
        b1 = n1*filter_coefficients[11];
        b2 = n2*filter_coefficients[17];
        b3 = n3*filter_coefficients[23];

        d0 = n0*filter_coefficients[2];
        d1 = n1*filter_coefficients[8];
        d2 = n2*filter_coefficients[14];
        d3 = n3*filter_coefficients[20];

        a0 = m0*filter_coefficients[4];
        a1 = m1*filter_coefficients[10];
        a2 = m2*filter_coefficients[16];
        a3 = m3*filter_coefficients[22];

        c0 = m0*filter_coefficients[1];
        c1 = m1*filter_coefficients[7];
        c2 = m2*filter_coefficients[13];
        c3 = m3*filter_coefficients[19];

        output[i] = o3;      
    }
}

void matrixStyle2(double* input, double* output, int number_of_samples) {

    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[2*NUM_STAGES] = {0};
    double inp_1, outp_1;
    int i;

    __m256d vec_x = _mm256_setzero_pd();
    __m256d vec_y = _mm256_setzero_pd();

    const __m256d coef_x0 = _mm256_load_pd(matrixCoeffs2);
    const __m256d coef_y0 = _mm256_load_pd(matrixCoeffs2 + 4);
    const __m256d coef_x1 = _mm256_load_pd(matrixCoeffs2 + 8);
    const __m256d coef_y1 = _mm256_load_pd(matrixCoeffs2 +12);
    const __m256d coef_x2 = _mm256_load_pd(matrixCoeffs2 +16);
    const __m256d coef_y2 = _mm256_load_pd(matrixCoeffs2 +20);
    const __m256d coef_x3 = _mm256_load_pd(matrixCoeffs2 +24);
    const __m256d coef_y3 = _mm256_load_pd(matrixCoeffs2 +28);

    for(i = 0; i < number_of_samples-1; i+=2)
    {
        inp_1 = input[i];
        __m256d vinp_1 = _mm256_broadcast_sd(input+i);

        _mm256_store_pd(x, vec_x);
        _mm256_store_pd(y, vec_y);

        __m256d perm_x1 = _mm256_permute_pd(vec_x, 0b0000); // make 0 0 2 2 copy
        __m256d perm_y1 = _mm256_permute_pd(vec_y, 0b0000); // make 0 0 2 2 copy
        __m256d perm_x2 = _mm256_permute2f128_pd(vec_x, vec_x, 0b00100000); // make 0 1 0 1 copy
        __m256d perm_y2 = _mm256_permute2f128_pd(vec_y, vec_y, 0b00100000); // make 0 1 0 1 copy
        __m256d perm_x3 = _mm256_permute_pd(perm_x2, 0b0100); // make 0 0 1 0 copy
        __m256d perm_y3 = _mm256_permute_pd(perm_y2, 0b0100); // make 0 0 1 0 copy

        __m256d intermed_0 = _mm256_mul_pd(vec_x, coef_x0);
        __m256d intermed_1 = _mm256_mul_pd(vec_y, coef_y0);
        __m256d intermed_2 = _mm256_add_pd(intermed_0, intermed_1);
        __m256d intermed_3 = _mm256_sub_pd(vinp_1, intermed_2);

        __m256d intermed_4 = _mm256_mul_pd(perm_x1, coef_x1);
        __m256d intermed_5 = _mm256_mul_pd(perm_y1, coef_y1);
        __m256d intermed_6 = _mm256_add_pd(intermed_4, intermed_5);
        __m256d intermed_7 = _mm256_add_pd(intermed_3, intermed_6);

        __m256d intermed_8 = _mm256_mul_pd(perm_x2, coef_x2);
        __m256d intermed_9 = _mm256_mul_pd(perm_y2, coef_y2);
        __m256d intermed_10= _mm256_add_pd(intermed_8, intermed_9);
        __m256d intermed_11= _mm256_add_pd(intermed_7, intermed_10);

        __m256d intermed_12= _mm256_mul_pd(perm_x3, coef_x3);
        __m256d intermed_13= _mm256_mul_pd(perm_y3, coef_y3);
        __m256d intermed_14= _mm256_add_pd(intermed_12, intermed_13);
        __m256d intermed_15= _mm256_add_pd(intermed_11, intermed_14);


        outp_1 = inp_1 + x[0]*new_order2[0] + y[0]*new_order2[4] + x[1]*new_order2[1] + y[1]*new_order2[5] + x[2]*new_order2[2] + y[2]*new_order2[6] + x[3]*new_order2[3] + y[3]*new_order2[7];

        output[i] = outp_1;

        inp_1 = input[i+1];
        vinp_1 = _mm256_broadcast_sd(input+i+1);

        _mm256_store_pd(x, intermed_15);
        _mm256_store_pd(y, vec_x);

        __m256d perm_x5 = _mm256_permute_pd(intermed_15, 0b0000); // make 0 0 2 2 copy
        __m256d perm_y5 = _mm256_permute_pd(vec_x, 0b0000); // make 0 0 2 2 copy
        __m256d perm_x6 = _mm256_permute2f128_pd(intermed_15, intermed_15, 0b00100000); // make 0 1 0 1 copy
        __m256d perm_y6 = _mm256_permute2f128_pd(vec_x, vec_x, 0b00100000); // make 0 1 0 1 copy
        __m256d perm_x7 = _mm256_permute_pd(perm_x6, 0b0100); // make 0 0 1 0 copy
        __m256d perm_y7 = _mm256_permute_pd(perm_y6, 0b0100); // make 0 0 1 0 copy

        __m256d intermed_16 = _mm256_mul_pd(intermed_15, coef_x0);
        __m256d intermed_17 = _mm256_mul_pd(vec_x, coef_y0);
        __m256d intermed_18 = _mm256_add_pd(intermed_16, intermed_17);
        __m256d intermed_19 = _mm256_sub_pd(vinp_1, intermed_18);

        __m256d intermed_20 = _mm256_mul_pd(perm_x5, coef_x1);
        __m256d intermed_21 = _mm256_mul_pd(perm_y5, coef_y1);
        __m256d intermed_22 = _mm256_add_pd(intermed_20, intermed_21);
        __m256d intermed_23 = _mm256_add_pd(intermed_19, intermed_22);

        __m256d intermed_24 = _mm256_mul_pd(perm_x6, coef_x2);
        __m256d intermed_25 = _mm256_mul_pd(perm_y6, coef_y2);
        __m256d intermed_26= _mm256_add_pd(intermed_24, intermed_25);
        __m256d intermed_27= _mm256_add_pd(intermed_23, intermed_26);

        __m256d intermed_28= _mm256_mul_pd(perm_x7, coef_x3);
        __m256d intermed_29= _mm256_mul_pd(perm_y7, coef_y3);
        __m256d intermed_30= _mm256_add_pd(intermed_28, intermed_29);
        __m256d intermed_31= _mm256_add_pd(intermed_27, intermed_30);

        outp_1 = inp_1 + x[0]*new_order2[0] + y[0]*new_order2[4] + x[1]*new_order2[1] + y[1]*new_order2[5] + x[2]*new_order2[2] + y[2]*new_order2[6] + x[3]*new_order2[3] + y[3]*new_order2[7];

        output[i+1] = outp_1;

        vec_y = intermed_15;
        vec_x = intermed_31;
    }

    for(; i < number_of_samples; i++){
        i++;
        inp_1 = input[i];
        __m256d vinp_1 = _mm256_broadcast_sd(input+i);

        _mm256_store_pd(x, vec_x);
        _mm256_store_pd(y, vec_y);

        __m256d perm_x1 = _mm256_permute_pd(vec_x, 0b0000); // make 0 0 2 2 copy
        __m256d perm_y1 = _mm256_permute_pd(vec_y, 0b0000); // make 0 0 2 2 copy
        __m256d perm_x2 = _mm256_permute2f128_pd(vec_x, vec_x, 0b00100000); // make 0 1 0 1 copy
        __m256d perm_y2 = _mm256_permute2f128_pd(vec_y, vec_y, 0b00100000); // make 0 1 0 1 copy
        __m256d perm_x3 = _mm256_permute_pd(perm_x2, 0b0100); // make 0 0 1 0 copy
        __m256d perm_y3 = _mm256_permute_pd(perm_y2, 0b0100); // make 0 0 1 0 copy

        __m256d intermed_0 = _mm256_mul_pd(vec_x, coef_x0);
        __m256d intermed_1 = _mm256_mul_pd(vec_y, coef_y0);
        __m256d intermed_2 = _mm256_add_pd(intermed_0, intermed_1);
        __m256d intermed_3 = _mm256_sub_pd(vinp_1, intermed_2);

        __m256d intermed_4 = _mm256_mul_pd(perm_x1, coef_x1);
        __m256d intermed_5 = _mm256_mul_pd(perm_y1, coef_y1);
        __m256d intermed_6 = _mm256_add_pd(intermed_4, intermed_5);
        __m256d intermed_7 = _mm256_add_pd(intermed_3, intermed_6);

        __m256d intermed_8 = _mm256_mul_pd(perm_x2, coef_x2);
        __m256d intermed_9 = _mm256_mul_pd(perm_y2, coef_y2);
        __m256d intermed_10= _mm256_add_pd(intermed_8, intermed_9);
        __m256d intermed_11= _mm256_add_pd(intermed_7, intermed_10);

        __m256d intermed_12= _mm256_mul_pd(perm_x3, coef_x3);
        __m256d intermed_13= _mm256_mul_pd(perm_y3, coef_y3);
        __m256d intermed_14= _mm256_add_pd(intermed_12, intermed_13);
        __m256d intermed_15= _mm256_add_pd(intermed_11, intermed_14);


        outp_1 = inp_1 + x[0]*new_order2[0] + y[0]*new_order2[4] + x[1]*new_order2[1] + y[1]*new_order2[5] + x[2]*new_order2[2] + y[2]*new_order2[6] + x[3]*new_order2[3] + y[3]*new_order2[7];

        output[i] = outp_1;

        vec_y = vec_x;
        vec_x = intermed_15;
    }
}

const double matrixCoeffs[32] = {
filter_coefficients[4] , // 0
filter_coefficients[10],
filter_coefficients[16],
filter_coefficients[22],
filter_coefficients[5] ,
filter_coefficients[11],
filter_coefficients[17], // 6
filter_coefficients[23],
0, 
new_order1[2]-new_order1[0], 
0, 
new_order1[10]-new_order1[8], 
0, // 12
new_order1[3]-new_order1[1], 
0,
new_order1[11]-new_order1[9], 
0, 
0, 
new_order1[2]-new_order1[0], // 18
new_order1[6]-new_order1[4],
0, 
0, 
new_order1[3]-new_order1[1], 
new_order1[7]-new_order1[5],
0, 
0, 
new_order1[6]-new_order1[4], 
new_order1[2]-new_order1[0], 
0,
0, 
new_order1[7]-new_order1[5],
new_order1[3]-new_order1[1],
};

void matrixStyle(double* input, double* output, int number_of_samples) {

    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[2*NUM_STAGES] = {0};
    double inp_1, outp_1;

    __m256d vec_x = _mm256_setzero_pd();
    __m256d vec_y = _mm256_setzero_pd();

    const __m256d coef_x0 = _mm256_load_pd(matrixCoeffs);
    const __m256d coef_y0 = _mm256_load_pd(matrixCoeffs + 4);
    const __m256d coef_x1 = _mm256_load_pd(matrixCoeffs + 8);
    const __m256d coef_y1 = _mm256_load_pd(matrixCoeffs +12);
    const __m256d coef_x2 = _mm256_load_pd(matrixCoeffs +16);
    const __m256d coef_y2 = _mm256_load_pd(matrixCoeffs +20);
    const __m256d coef_x3 = _mm256_load_pd(matrixCoeffs +24);
    const __m256d coef_y3 = _mm256_load_pd(matrixCoeffs +28);

    for(int i = 0; i < number_of_samples; i++)
    {
        inp_1 = input[i];
        __m256d vinp_1 = _mm256_broadcast_sd(input+i);

        //vec_x = _mm256_load_pd(x);
        //vec_y = _mm256_load_pd(y);
        _mm256_store_pd(x, vec_x);
        _mm256_store_pd(y, vec_y);

        __m256d perm_x1 = _mm256_permute_pd(vec_x, 0b0000); // make 0 0 2 2 copy
        __m256d perm_y1 = _mm256_permute_pd(vec_y, 0b0000); // make 0 0 2 2 copy
        __m256d perm_x2 = _mm256_permute2f128_pd(vec_x, vec_x, 0b00100000); // make 0 1 0 1 copy
        __m256d perm_y2 = _mm256_permute2f128_pd(vec_y, vec_y, 0b00100000); // make 0 1 0 1 copy
        __m256d perm_x3 = _mm256_permute_pd(perm_x2, 0b0100); // make 0 0 1 0 copy
        __m256d perm_y3 = _mm256_permute_pd(perm_y2, 0b0100); // make 0 0 1 0 copy

        //temp[0] = inp_1 - x[0]*matrixCoeffs[0] - y[0]*matrixCoeffs[4];
        //temp[1] = inp_1 - x[1]*matrixCoeffs[1] - y[1]*matrixCoeffs[5];
        //temp[2] = inp_1 - x[2]*matrixCoeffs[2] - y[2]*matrixCoeffs[6];
        //temp[3] = inp_1 - x[3]*matrixCoeffs[3] - y[3]*matrixCoeffs[7];
        __m256d intermed_0 = _mm256_mul_pd(vec_x, coef_x0);
        __m256d intermed_1 = _mm256_mul_pd(vec_y, coef_y0);
        __m256d intermed_2 = _mm256_add_pd(intermed_0, intermed_1);
        __m256d intermed_3 = _mm256_sub_pd(vinp_1, intermed_2);

        //temp[0] += 0;
        //temp[1] += x[0]*matrixCoeffs[9]  + y[0]*matrixCoeffs[13];
        //temp[2] += 0;
        //temp[3] += x[2]*matrixCoeffs[11] + y[2]*matrixCoeffs[15];
        __m256d intermed_4 = _mm256_mul_pd(perm_x1, coef_x1);
        __m256d intermed_5 = _mm256_mul_pd(perm_y1, coef_y1);
        __m256d intermed_6 = _mm256_add_pd(intermed_4, intermed_5);
        __m256d intermed_7 = _mm256_add_pd(intermed_3, intermed_6);

        //temp[0] += 0;
        //temp[1] += 0;
        //temp[2] += x[0]*matrixCoeffs[18] + y[0]*matrixCoeffs[22];
        //temp[3] += x[1]*matrixCoeffs[19] + y[1]*matrixCoeffs[23];
        __m256d intermed_8 = _mm256_mul_pd(perm_x2, coef_x2);
        __m256d intermed_9 = _mm256_mul_pd(perm_y2, coef_y2);
        __m256d intermed_10= _mm256_add_pd(intermed_8, intermed_9);
        __m256d intermed_11= _mm256_add_pd(intermed_7, intermed_10);


        //temp[0] += 0;
        //temp[1] += 0;
        //temp[2] += x[1]*matrixCoeffs[26] + y[1]*matrixCoeffs[30];
        //temp[3] += x[0]*matrixCoeffs[27] + y[0]*matrixCoeffs[31];
        __m256d intermed_12= _mm256_mul_pd(perm_x3, coef_x3);
        __m256d intermed_13= _mm256_mul_pd(perm_y3, coef_y3);
        __m256d intermed_14= _mm256_add_pd(intermed_12, intermed_13);
        __m256d intermed_15= _mm256_add_pd(intermed_11, intermed_14);


        outp_1 = inp_1 + x[0]*new_order2[0] + y[0]*new_order2[4] + x[1]*new_order2[1] + y[1]*new_order2[5] + x[2]*new_order2[2] + y[2]*new_order2[6] + x[3]*new_order2[3] + y[3]*new_order2[7];

        output[i] = outp_1;

        //y[0] = x[0];
        //y[1] = x[1];
        //y[2] = x[2];
        //y[3] = x[3];
        //x[0] = temp[0];
        //x[1] = temp[1];
        //x[2] = temp[2];
        //x[3] = temp[3];
        vec_y = vec_x;
        vec_x = intermed_15;
    }

}

// not using coeficients that are 1 anymore, unrolled for scalar replacement
void pipelined2(double* input, double* output, int number_of_samples) {

    double x[NUM_STAGES];
    double y[NUM_STAGES];
    int i;

    const __m256d coef_x1 = _mm256_load_pd(new_order2);
    const __m256d coef_y1 = _mm256_load_pd(new_order2 +4);
    const __m256d coef_x2 = _mm256_load_pd(new_order2 +8);
    const __m256d coef_y2 = _mm256_load_pd(new_order2 +12);
    const __m256d mask1 = _mm256_set_pd(0xffffffffffffffff, 0, 0xffffffffffffffff, 0);
    const __m256d mask2 = _mm256_set_pd(0xffffffffffffffff, 0xffffffffffffffff, 0, 0);

    static __m256d vec_x = _mm256_setzero_pd();
    static __m256d vec_y = _mm256_setzero_pd();

    for(i = 0; i < number_of_samples; i++)
    {
        __m256d inp_1 = _mm256_broadcast_sd(input+i);

        _mm256_store_pd(y, vec_y); // is needed later
        _mm256_store_pd(x, vec_x); // is needed later

        double outpart = x[3]*new_order1[14] + y[3]*new_order1[15];

        // double intermed_0 = x[0]*new_order2[0] + y[0]*new_order2[4];
        // double intermed_1 = x[1]*new_order2[1] + y[1]*new_order2[5];
        // double intermed_2 = x[2]*new_order2[2] + y[2]*new_order2[6];
        // double intermed_3 = x[3]*new_order2[3] + y[3]*new_order2[7];
        // double intermed_4 = x[0]*new_order2[8] + y[0]*new_order2[12];
        // double intermed_5 = x[1]*new_order2[9] + y[1]*new_order2[13];
        // double intermed_6 = x[2]*new_order2[10] + y[2]*new_order2[14];
        // double intermed_7 = x[3]*new_order2[11] + y[3]*new_order2[15];
        __m256d intermed_0 = _mm256_mul_pd(vec_x, coef_x1);
        __m256d intermed_1 = _mm256_mul_pd(vec_y, coef_y1);
        __m256d intermed_2 = _mm256_mul_pd(vec_x, coef_x2);
        __m256d intermed_3 = _mm256_mul_pd(vec_y, coef_y2);
        __m256d intermed_4 = _mm256_add_pd(intermed_0, intermed_1); // intermed 0-3
        __m256d intermed_5 = _mm256_add_pd(intermed_2, intermed_3); // intermed 4-7

        //y[0] = x[0];
        //y[1] = x[1];
        //y[2] = x[2];
        //y[3] = x[3];
        vec_y = vec_x;

        //x[0] = inp_1 - intermed_4;
        //x[1] = inp_1 - intermed_5 + intermed_0;
        //x[2] = inp_1 - intermed_6 + intermed_0 + intermed_1;
        //x[3] = inp_1 - intermed_7 + intermed_0 + intermed_1 + intermed_2;
        __m256d intermed_6 = _mm256_sub_pd(inp_1, intermed_5); // inp - intermed
        __m256d intermed_7 = _mm256_hadd_pd(intermed_4, intermed_4); // produces  0+1, 0+1, 2, 2+3
        __m256d intermed_8 = _mm256_permute2f128_pd(intermed_7, intermed_7, 0); // produces 0+1, 0+1, 0+1, 0+1
        __m256d intermed_9 = _mm256_permute_pd(intermed_4, 0x0000); // should be 0 0 2 2
        __m256d intermed_10 = _mm256_and_pd(intermed_9, mask1); // zero 0 zero 2
        __m256d intermed_11 = _mm256_and_pd(intermed_8, mask2); // zero zero 0+1 0+1
        vec_x = _mm256_add_pd(intermed_6, intermed_10);
        vec_x = _mm256_add_pd(vec_x, intermed_11);
        
        _mm256_store_pd(x, vec_x); // is needed later
        double outp_1 = x[3] + outpart;

        output[i] = outp_1;

    }

}

// not using coeficients that are 1 anymore, unrolled for scalar replacement
void pipelined1(double* input, double* output, int number_of_samples) {
    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[2*NUM_STAGES] = {0};
    double z[2*NUM_STAGES+1];
    double inp_1, outp_1, inp_2, outp_2;

    int i;

    for(i = 0; i < number_of_samples; i++)
    {
        inp_1 = input[i];

        double intermed_0 = x[0]*new_order2[0] + y[0]*new_order2[4];
        double intermed_1 = x[1]*new_order2[1] + y[1]*new_order2[5];
        double intermed_2 = x[2]*new_order2[2] + y[2]*new_order2[6];
        double intermed_3 = x[3]*new_order2[3] + y[3]*new_order2[7];

        double intermed_4 = x[0]*new_order2[8] + y[0]*new_order2[12];
        double intermed_5 = x[1]*new_order2[9] + y[1]*new_order2[13];
        double intermed_6 = x[2]*new_order2[10] + y[2]*new_order2[14];
        double intermed_7 = x[3]*new_order2[11] + y[3]*new_order2[15];

        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
        y[3] = x[3];

        x[0] = inp_1 - intermed_4;
        x[1] = inp_1 - intermed_5 + intermed_0;
        x[2] = inp_1 - intermed_6 + intermed_0 + intermed_1;
        x[3] = inp_1 - intermed_7 + intermed_0 + intermed_1 + intermed_2;

        outp_1 = inp_1 + intermed_0 + intermed_1 + intermed_2 + intermed_3;

        output[i] = outp_1;

    }

}

// not using coeficients that are 1 anymore, unrolled for scalar replacement
void pipelined0(double* input, double* output, int number_of_samples) {
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

//dumb base implementation
void reversed3(double* input, double* output, int number_of_samples) {
    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[NUM_STAGES] = {0}; //z-2 buffers
    double* z = (double*) calloc(NUM_STAGES*number_of_samples, sizeof(double)); //temporary output from first SOS-stage

    int i = 0;

    z[0] = input[0];
    temp[0] = z[0] - x[0]*new_order1[0] - y[0]*new_order1[1];
    z[number_of_samples] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
    y[0] = x[0];
    x[0] = temp[0];

    temp[1] = z[number_of_samples] - x[1]*new_order1[4] - y[1]*new_order1[5];
    z[2*number_of_samples] = temp[1] + x[1]*new_order1[6] + y[1]*new_order1[7];
    y[1] = x[1];
    x[1] = temp[1];

    temp[2] = z[2*number_of_samples] - x[2]*new_order1[8] - y[2]*new_order1[9];
    z[3*number_of_samples] = temp[2] + x[2]*new_order1[10] + y[2]*new_order1[11];
    y[2] = x[2];
    x[2] = temp[2];

    z[1] = input[1];
    temp[0] = z[1] - x[0]*new_order1[0] - y[0]*new_order1[1];
    z[number_of_samples+1] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
    y[0] = x[0];
    x[0] = temp[0];

    temp[1] = z[number_of_samples+1] - x[1]*new_order1[4] - y[1]*new_order1[5];
    z[2*number_of_samples+1] = temp[1] + x[1]*new_order1[6] + y[1]*new_order1[7];
    y[1] = x[1];
    x[1] = temp[1];

    z[2] = input[2];
    temp[0] = z[2] - x[0]*new_order1[0] - y[0]*new_order1[1];
    z[number_of_samples+2] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
    y[0] = x[0];
    x[0] = temp[0];

    for(i = 0; i < number_of_samples; i++)
    {
        if(i+3 < number_of_samples){
            z[(i+3)] = input[(i+3)];
            temp[0] = z[(i+3)] - x[0]*new_order1[0] - y[0]*new_order1[1];
            z[number_of_samples+(i+3)] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
            y[0] = x[0];
            x[0] = temp[0];
        }

        if(i+2 < number_of_samples){
            temp[1] = z[number_of_samples+(i+2)] - x[1]*new_order1[4] - y[1]*new_order1[5];
            z[2*number_of_samples+(i+2)] = temp[1] + x[1]*new_order1[6] + y[1]*new_order1[7];
            y[1] = x[1];
            x[1] = temp[1];
        }

        if(i+1 < number_of_samples){
            temp[2] = z[2*number_of_samples+(i+1)] - x[2]*new_order1[8] - y[2]*new_order1[9];
            z[3*number_of_samples+(i+1)] = temp[2] + x[2]*new_order1[10] + y[2]*new_order1[11];
            y[2] = x[2];
            x[2] = temp[2];
        }

        temp[3] = z[3*number_of_samples+i] - x[3]*new_order1[12] - y[3]*new_order1[13];
        output[i] = temp[3] + x[3]*new_order1[14] + y[3]*new_order1[15];
        y[3] = x[3];
        x[3] = temp[3];
    }

    free(z);
}

//dumb base implementation
void reversed2(double* input, double* output, int number_of_samples) {
    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[NUM_STAGES] = {0}; //z-2 buffers
    double* z = (double*) calloc(NUM_STAGES*number_of_samples, sizeof(double)); //temporary output from first SOS-stage

    int i = 0;

    z[0] = input[0];
    temp[0] = z[0] - x[0]*new_order1[0] - y[0]*new_order1[1];
    z[1] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
    y[0] = x[0];
    x[0] = temp[0];

    temp[1] = z[1] - x[1]*new_order1[4] - y[1]*new_order1[5];
    z[2] = temp[1] + x[1]*new_order1[6] + y[1]*new_order1[7];
    y[1] = x[1];
    x[1] = temp[1];

    temp[2] = z[2] - x[2]*new_order1[8] - y[2]*new_order1[9];
    z[3] = temp[2] + x[2]*new_order1[10] + y[2]*new_order1[11];
    y[2] = x[2];
    x[2] = temp[2];

    z[NUM_STAGES] = input[1];
    temp[0] = z[NUM_STAGES] - x[0]*new_order1[0] - y[0]*new_order1[1];
    z[NUM_STAGES+1] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
    y[0] = x[0];
    x[0] = temp[0];

    temp[1] = z[NUM_STAGES+1] - x[1]*new_order1[4] - y[1]*new_order1[5];
    z[NUM_STAGES+2] = temp[1] + x[1]*new_order1[6] + y[1]*new_order1[7];
    y[1] = x[1];
    x[1] = temp[1];

    z[2*NUM_STAGES] = input[2];
    temp[0] = z[2*NUM_STAGES] - x[0]*new_order1[0] - y[0]*new_order1[1];
    z[2*NUM_STAGES+1] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
    y[0] = x[0];
    x[0] = temp[0];

    for(i = 0; i < number_of_samples -3; i++)
    {
        z[(i+3)*NUM_STAGES] = input[(i+3)];
        temp[0] = z[(i+3)*NUM_STAGES] - x[0]*new_order1[0] - y[0]*new_order1[1];
        z[(i+3)*NUM_STAGES+1] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
        y[0] = x[0];
        x[0] = temp[0];

        temp[1] = z[(i+2)*NUM_STAGES+1] - x[1]*new_order1[4] - y[1]*new_order1[5];
        z[(i+2)*NUM_STAGES+2] = temp[1] + x[1]*new_order1[6] + y[1]*new_order1[7];
        y[1] = x[1];
        x[1] = temp[1];    
    
        temp[2] = z[(i+1)*NUM_STAGES+2] - x[2]*new_order1[8] - y[2]*new_order1[9];
        z[(i+1)*NUM_STAGES+3] = temp[2] + x[2]*new_order1[10] + y[2]*new_order1[11];
        y[2] = x[2];
        x[2] = temp[2];    

        temp[3] = z[i*NUM_STAGES+3] - x[3]*new_order1[12] - y[3]*new_order1[13];
        output[i] = temp[3] + x[3]*new_order1[14] + y[3]*new_order1[15];
        y[3] = x[3];
        x[3] = temp[3];
    }

    temp[1] = z[(i+2)*NUM_STAGES+1] - x[1]*new_order1[4] - y[1]*new_order1[5];
    z[(i+2)*NUM_STAGES+2] = temp[1] + x[1]*new_order1[6] + y[1]*new_order1[7];
    y[1] = x[1];
    x[1] = temp[1];    

    temp[2] = z[(i+1)*NUM_STAGES+2] - x[2]*new_order1[8] - y[2]*new_order1[9];
    z[(i+1)*NUM_STAGES+3] = temp[2] + x[2]*new_order1[10] + y[2]*new_order1[11];
    y[2] = x[2];
    x[2] = temp[2];    

    temp[3] = z[i*NUM_STAGES+3] - x[3]*new_order1[12] - y[3]*new_order1[13];
    output[i] = temp[3] + x[3]*new_order1[14] + y[3]*new_order1[15];
    y[3] = x[3];
    x[3] = temp[3];

    i = number_of_samples -2;
    temp[2] = z[(i+1)*NUM_STAGES+2] - x[2]*new_order1[8] - y[2]*new_order1[9];
    z[(i+1)*NUM_STAGES+3] = temp[2] + x[2]*new_order1[10] + y[2]*new_order1[11];
    y[2] = x[2];
    x[2] = temp[2];    

    temp[3] = z[i*NUM_STAGES+3] - x[3]*new_order1[12] - y[3]*new_order1[13];
    output[i] = temp[3] + x[3]*new_order1[14] + y[3]*new_order1[15];
    y[3] = x[3];
    x[3] = temp[3];

    i = number_of_samples -1;
    temp[3] = z[i*NUM_STAGES+3] - x[3]*new_order1[12] - y[3]*new_order1[13];
    output[i] = temp[3] + x[3]*new_order1[14] + y[3]*new_order1[15];
    y[3] = x[3];
    x[3] = temp[3];

    free(z);
}

//dumb base implementation
void reversed(double* input, double* output, int number_of_samples) {
    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[NUM_STAGES] = {0}; //temporary store of the "fall-through" value
    double* z = (double*) calloc(NUM_STAGES*number_of_samples, sizeof(double)); //temporary output from first SOS-stage

    for(int i = 0; i < number_of_samples; i++)
    {
        z[i*NUM_STAGES] = input[i];
        temp[0] = z[i*NUM_STAGES] - x[0]*new_order1[0] - y[0]*new_order1[1];
        z[i*NUM_STAGES+1] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
        y[0] = x[0];
        x[0] = temp[0];
    }

    for(int i = 0; i < number_of_samples; i++)
    {
        temp[1] = z[i*NUM_STAGES+1] - x[1]*new_order1[4] - y[1]*new_order1[5];
        z[i*NUM_STAGES+2] = temp[1] + x[1]*new_order1[6] + y[1]*new_order1[7];
        y[1] = x[1];
        x[1] = temp[1];
    }
    
    for(int i = 0; i < number_of_samples; i++)
    {
        temp[2] = z[i*NUM_STAGES+2] - x[2]*new_order1[8] - y[2]*new_order1[9];
        z[i*NUM_STAGES+3] = temp[2] + x[2]*new_order1[10] + y[2]*new_order1[11];
        y[2] = x[2];
        x[2] = temp[2];
    }
    
    for(int i = 0; i < number_of_samples; i++)
    {
        temp[3] = z[i*NUM_STAGES+3] - x[3]*new_order1[12] - y[3]*new_order1[13];
        output[i] = temp[3] + x[3]*new_order1[14] + y[3]*new_order1[15];
        y[3] = x[3];
        x[3] = temp[3];
    }

    free(z);
}

// not using coeficients that are 1 anymore, unrolled for scalar replacement
void slowperformance3(double* input, double* output, int number_of_samples) {
    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
    double temp[NUM_STAGES] = {0};
    double z[NUM_STAGES+1];

    for(int i = 0; i < number_of_samples; i++)
    {
        z[0] = input[i];

        temp[0] = z[0] - x[0]*new_order1[0] - y[0]*new_order1[1];
        z[1] = temp[0] + x[0]*new_order1[2] + y[0]*new_order1[3];
        temp[1] = z[1] - x[1]*new_order1[4] - y[1]*new_order1[5];
        z[2] = temp[1] + x[1]*new_order1[6] + y[1]*new_order1[7];
        temp[2] = z[2] - x[2]*new_order1[8] - y[2]*new_order1[9];
        z[3] = temp[2] + x[2]*new_order1[10] + y[2]*new_order1[11];
        temp[3] = z[3] - x[3]*new_order1[12] - y[3]*new_order1[13];
        z[4] = temp[3] + x[3]*new_order1[14] + y[3]*new_order1[15];


        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2];
        y[3] = x[3];
        x[0] = temp[0];
        x[1] = temp[1];
        x[2] = temp[2];
        x[3] = temp[3];

        output[i] = z[4];
    }

}

// not using coeficients that are 1 anymore, unrolled for scalar replacement
void slowperformance2(double* input, double* output, int number_of_samples) {
    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
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
    double x[NUM_STAGES] = {0}; //z-1 buffers
    double y[NUM_STAGES] = {0}; //z-2 buffers
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
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "matlab_data.h"
#include "../hamilton_inline/qrsdet.h"
#include <emmintrin.h>

//
//The slow base version
//
void slowperformance(float* input, float* output, int samples_to_process) 
{
    // std::cout << "loop length "<< samples_to_process<<"\n";
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;

    for(int index = 0; index < samples_to_process; index++)
    {
        halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;    // Use halfPtr to index
        if(halfPtr < 0)                         // to x[n-6].
            halfPtr += LPBUFFER_LGTH ;

        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
        y2 = y1;
        y1 = y0;
        fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);
        lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
        
        hp_y += fdatum - hp_data[hp_ptr];
        halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;
        if(halfPtr < 0)
            halfPtr += HPBUFFER_LGTH ;
        hp_data[hp_ptr] = fdatum ;
        fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);
        y = fdatum - derBuff[derI] ;
        derBuff[derI] = fdatum;
        fdatum = y;
        fdatum = fabs(fdatum) ;             // Take the absolute value.
        sum += fdatum ;
        sum -= data[ptr] ;
        data[ptr] = fdatum ;

        // #ifdef OPERATION_COUNTER
        // float_comp_counter++;
        // #endif
        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output_temp = 32000.f ;
        } 
        else 
        {
            output_temp = sum / (float)WINDOW_WIDTH ;
            // #ifdef OPERATION_COUNTER
            // float_div_counter += 1;
            // #endif
        }

        if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
            lp_ptr = 0 ;                    // the buffer pointer.
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
        
        // #ifdef OPERATION_COUNTER
        //     float_add_counter += 10;
        //     float_mul_counter++;
        //     float_div_counter += 4;
        // #endif
        output[index] = output_temp;
    }
}

void slowperformance2(float* input, float* output, int samples_to_process) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
    static int ptr_array[4] = {0}; // lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;
    float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

    __m128i maxMask = _mm_set_epi32(WINDOW_WIDTH, DERIV_LENGTH, HPBUFFER_LGTH, LPBUFFER_LGTH);
    __m128i oneVec = _mm_set_epi32(1, 1, 1, 1);

    int i;
    for(i=0; i < samples_to_process - BLOCKING_SIZE+1; i+= BLOCKING_SIZE)
    {
        for(int j=0; j < BLOCKING_SIZE; j++)
        {
            index = i + j;
            halfPtr = ptr_array[0] - lpbuffer_lgth_half ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[ptr_array[0]] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 * lpbuffer_sqr_div_4;
            lp_data[ptr_array[0]] = input[index] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[ptr_array[1]];
            halfPtr = ptr_array[1] - hpbuffer_lgth_half ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            hp_data[ptr_array[1]] = fdatum ;
            fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
            y = fdatum - derBuff[ptr_array[2]] ;
            derBuff[ptr_array[2]] = fdatum;
            fdatum = y;
            fdatum = fabs(fdatum) ;             // Take the absolute value.
            sum += fdatum - data[ptr_array[3]] ;
            data[ptr_array[3]] = fdatum ;

            #ifdef OPERATION_COUNTER
            float_comp_counter++;
            float_mul_counter++;
            #endif
            if((sum * window_width_inv) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum * window_width_inv ;
                #ifdef OPERATION_COUNTER
                float_mul_counter += 1;
                #endif
            }

            __m128 ptr_vecf = _mm_load_ps((float*)ptr_array);
            __m128i ptr_vec = _mm_castps_si128(ptr_vecf);
            __m128i onePlus = _mm_add_epi32(oneVec, ptr_vec);
            __m128i ltmask = _mm_cmplt_epi32(onePlus, maxMask);
            __m128i result =  _mm_and_si128(onePlus, ltmask);
            __m128 resultf = _mm_castsi128_ps(result);
            _mm_store_ps((float*)ptr_array, resultf);
            
            #ifdef OPERATION_COUNTER
                float_add_counter += 10;
                float_mul_counter+=4;
            #endif
            output[index] = output_temp;
        }
    }

    for(i; i<BLOCKING_SIZE; i++)
    {
        halfPtr = ptr_array[0] - lpbuffer_lgth_half ;    // Use halfPtr to index
        if(halfPtr < 0)                         // to x[n-6].
            halfPtr += LPBUFFER_LGTH ;

        y0 = (y1*2.0f) - y2 + input[i] - (lp_data[halfPtr]*2.0f) + lp_data[ptr_array[0]] ;
        y2 = y1;
        y1 = y0;
        fdatum = y0 * lpbuffer_sqr_div_4;
        lp_data[ptr_array[0]] = input[i] ;            // Stick most recent sample into
        
        hp_y += fdatum - hp_data[ptr_array[1]];
        halfPtr = ptr_array[1] - hpbuffer_lgth_half ;
        if(halfPtr < 0)
            halfPtr += HPBUFFER_LGTH ;
        hp_data[ptr_array[1]] = fdatum ;
        fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
        y = fdatum - derBuff[ptr_array[2]] ;
        derBuff[ptr_array[2]] = fdatum;
        fdatum = y;
        fdatum = fabs(fdatum) ;             // Take the absolute value.
        sum += fdatum - data[ptr_array[3]] ;
        data[ptr_array[3]] = fdatum ;

        #ifdef OPERATION_COUNTER
        float_comp_counter++;
        float_mul_counter++;
        #endif
        if((sum * window_width_inv) > 32000.f)
        {
            output_temp = 32000.f ;
        } 
        else 
        {
            output_temp = sum * window_width_inv ;
            #ifdef OPERATION_COUNTER
            float_mul_counter += 1;
            #endif
        }

        __m128 ptr_vecf = _mm_load_ps((float*)ptr_array);
        __m128i ptr_vec = _mm_castps_si128(ptr_vecf);
        __m128i onePlus = _mm_add_epi32(oneVec, ptr_vec);
        __m128i ltmask = _mm_cmplt_epi32(onePlus, maxMask);
        __m128i result =  _mm_and_si128(onePlus, ltmask);
        __m128 resultf = _mm_castsi128_ps(result);
        _mm_store_ps((float*)ptr_array, resultf);
        
        #ifdef OPERATION_COUNTER
            float_add_counter += 10;
            float_mul_counter+=4;
        #endif
        output[i] = output_temp;
    }
}


void blocking(float* input, float* output, int samples_to_process) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;

    for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
    {
        for(int j=0; j < BLOCKING_SIZE; j++)
        {
            index = i * BLOCKING_SIZE + j;
            halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);
            lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[hp_ptr];
            halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            hp_data[hp_ptr] = fdatum ;
            fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);
            y = fdatum - derBuff[derI] ;
            derBuff[derI] = fdatum;
            fdatum = y;
            fdatum = fabs(fdatum) ;             // Take the absolute value.
            sum += fdatum - data[ptr] ;
            data[ptr] = fdatum ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            if((sum / (float)WINDOW_WIDTH) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum / (float)WINDOW_WIDTH ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }

            if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr = 0 ;                    // the buffer pointer.
            if(++derI == DERIV_LENGTH)
                derI = 0 ;
            if(++hp_ptr == HPBUFFER_LGTH)
                hp_ptr = 0 ;
            if(++ptr == WINDOW_WIDTH)
                ptr = 0 ;
            
            // #ifdef OPERATION_COUNTER
            //     float_add_counter += 10;
            //     float_mul_counter++;
            //     float_div_counter += 4;
            // #endif
            output[index] = output_temp;
        }
    }
}


void blocking_no_divisions1(float* input, float* output, int samples_to_process) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;
    float lpbuffer_inv = 1/(float)LPBUFFER_LGTH;
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

    for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
    {
        for(int j=0; j < BLOCKING_SIZE; j++)
        {
            index = i * BLOCKING_SIZE + j;
            halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 * lpbuffer_inv * lpbuffer_inv * 4;
            lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[hp_ptr];
            halfPtr = hp_ptr - hpbuffer_lgth_half ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            hp_data[hp_ptr] = fdatum ;
            fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
            y = fdatum - derBuff[derI] ;
            derBuff[derI] = fdatum;
            fdatum = y;
            fdatum = fabs(fdatum) ;             // Take the absolute value.
            sum += fdatum - data[ptr] ;
            data[ptr] = fdatum ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            if((sum * window_width_inv) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum * window_width_inv ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }

            if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr = 0 ;                    // the buffer pointer.
            if(++derI == DERIV_LENGTH)
                derI = 0 ;
            if(++hp_ptr == HPBUFFER_LGTH)
                hp_ptr = 0 ;
            if(++ptr == WINDOW_WIDTH)
                ptr = 0 ;
            
            // #ifdef OPERATION_COUNTER
            //     float_add_counter += 10;
            //     float_mul_counter++;
            //     float_div_counter += 4;
            // #endif
            output[index] = output_temp;
        }
    }
}

void blocking_no_divisions2(float* input, float* output, int samples_to_process) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;
    float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

    for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
    {
        for(int j=0; j < BLOCKING_SIZE; j++)
        {
            index = i * BLOCKING_SIZE + j;
            halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 * lpbuffer_sqr_div_4;
            lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[hp_ptr];
            halfPtr = hp_ptr- hpbuffer_lgth_half ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            hp_data[hp_ptr] = fdatum ;
            fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
            y = fdatum - derBuff[derI] ;
            derBuff[derI] = fdatum;
            fdatum = y;
            fdatum = fabs(fdatum) ;             // Take the absolute value.
            sum += fdatum - data[ptr] ;
            data[ptr] = fdatum ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            if((sum * window_width_inv) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum * window_width_inv ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }

            if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr = 0 ;                    // the buffer pointer.
            if(++derI == DERIV_LENGTH)
                derI = 0 ;
            if(++hp_ptr == HPBUFFER_LGTH)
                hp_ptr = 0 ;
            if(++ptr == WINDOW_WIDTH)
                ptr = 0 ;
            
            // #ifdef OPERATION_COUNTER
            //     float_add_counter += 10;
            //     float_mul_counter++;
            //     float_div_counter += 4;
            // #endif
            output[index] = output_temp;
        }
    }
}

void blocking_no_divisions2_derI(float* input, float* output, int samples_to_process) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;
    float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

    for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
    {
        for(int j=0; j < BLOCKING_SIZE; j++)
        {
            index = i * BLOCKING_SIZE + j;
            halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 * lpbuffer_sqr_div_4;
            lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[hp_ptr];
            halfPtr = hp_ptr - hpbuffer_lgth_half ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            hp_data[hp_ptr] = fdatum ;
            fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
            y = fdatum - derBuff[derI] ;
            derBuff[derI] = fdatum;
            fdatum = y;
            fdatum = fabs(fdatum) ;             // Take the absolute value.
            sum += fdatum - data[ptr] ;
            data[ptr] = fdatum ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            if((sum * window_width_inv) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum * window_width_inv ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }

            if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr = 0 ;                    // the buffer pointer.
            if(derI == 0)
                derI = 1 ;
            else
                derI = 0 ;
            if(++hp_ptr == HPBUFFER_LGTH)
                hp_ptr = 0 ;
            if(++ptr == WINDOW_WIDTH)
                ptr = 0 ;
            
            // #ifdef OPERATION_COUNTER
            //     float_add_counter += 10;
            //     float_mul_counter++;
            //     float_div_counter += 4;
            // #endif
            output[index] = output_temp;
        }
    }
}

void blocking_no_divisions2_derI_precomp_sum(float* input, float* output, int samples_to_process) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0, sum_window = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;
    float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);
    int i;
    for(i=0; i < samples_to_process - BLOCKING_SIZE + 1; i+= BLOCKING_SIZE)
    {
        for(int j=0; j < BLOCKING_SIZE; j++)
        {
            index = i + j;
            halfPtr = lp_ptr- lpbuffer_lgth_half ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 * lpbuffer_sqr_div_4;
            lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[hp_ptr];
            halfPtr = hp_ptr- hpbuffer_lgth_half ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            hp_data[hp_ptr] = fdatum ;
            fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
            y = fdatum - derBuff[derI] ;
            derBuff[derI] = fdatum;
            fdatum = y;
            fdatum = fabs(fdatum) ;             // Take the absolute value.
            sum += fdatum - data[ptr] ;
            data[ptr] = fdatum ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            sum_window = sum * window_width_inv;
            if((sum_window) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum_window ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }

            if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr = 0 ;                    // the buffer pointer.
            if(derI == 0)
                derI = 1 ;
            else
                derI = 0 ;
            if(++hp_ptr == HPBUFFER_LGTH)
                hp_ptr = 0 ;
            if(++ptr == WINDOW_WIDTH)
                ptr = 0 ;
            
            // #ifdef OPERATION_COUNTER
            //     float_add_counter += 10;
            //     float_mul_counter++;
            //     float_div_counter += 4;
            // #endif
            output[index] = output_temp;
        }
    }
    for(; i < samples_to_process; i++)
    {
        halfPtr = lp_ptr- lpbuffer_lgth_half ;    // Use halfPtr to index
        if(halfPtr < 0)                         // to x[n-6].
            halfPtr += LPBUFFER_LGTH ;

        y0 = (y1*2.0f) - y2 + input[i] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
        y2 = y1;
        y1 = y0;
        fdatum = y0 * lpbuffer_sqr_div_4;
        lp_data[lp_ptr] = input[i] ;            // Stick most recent sample into
        
        hp_y += fdatum - hp_data[hp_ptr];
        halfPtr = hp_ptr- hpbuffer_lgth_half ;
        if(halfPtr < 0)
            halfPtr += HPBUFFER_LGTH ;
        hp_data[hp_ptr] = fdatum ;
        fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
        y = fdatum - derBuff[derI] ;
        derBuff[derI] = fdatum;
        fdatum = y;
        fdatum = fabs(fdatum) ;             // Take the absolute value.
        sum += fdatum - data[ptr] ;
        data[ptr] = fdatum ;

        // #ifdef OPERATION_COUNTER
        // float_comp_counter++;
        // #endif
        sum_window = sum * window_width_inv;
        if((sum_window) > 32000.f)
        {
            output_temp = 32000.f ;
        } 
        else 
        {
            output_temp = sum_window ;
            // #ifdef OPERATION_COUNTER
            // float_div_counter += 1;
            // #endif
        }

        if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
            lp_ptr = 0 ;                    // the buffer pointer.
        if(derI == 0)
            derI = 1 ;
        else
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
        
        // #ifdef OPERATION_COUNTER
        //     float_add_counter += 10;
        //     float_mul_counter++;
        //     float_div_counter += 4;
        // #endif
        output[i] = output_temp;
    }
}

void no_divisions2_derI_precomp_sum(float* input, float* output, int samples_to_process) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0, sum_window = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;
    float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

    for(int index=0; index < samples_to_process; index++)
    {
            halfPtr = lp_ptr- lpbuffer_lgth_half ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 * lpbuffer_sqr_div_4;
            lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[hp_ptr];
            halfPtr = hp_ptr- hpbuffer_lgth_half ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            hp_data[hp_ptr] = fdatum ;
            fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
            y = fdatum - derBuff[derI] ;
            derBuff[derI] = fdatum;
            fdatum = y;
            fdatum = fabs(fdatum) ;             // Take the absolute value.
            sum += fdatum - data[ptr] ;
            data[ptr] = fdatum ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            sum_window = sum * window_width_inv;
            if((sum_window) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum_window ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }

            if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr = 0 ;                    // the buffer pointer.
            if(derI == 0)
                derI = 1 ;
            else
                derI = 0 ;
            if(++hp_ptr == HPBUFFER_LGTH)
                hp_ptr = 0 ;
            if(++ptr == WINDOW_WIDTH)
                ptr = 0 ;
            
            // #ifdef OPERATION_COUNTER
            //     float_add_counter += 10;
            //     float_mul_counter++;
            //     float_div_counter += 4;
            // #endif
            output[index] = output_temp;
    }
}


void no_division(float* input, float* output, int samples_to_process) 
{
    // std::cout << "loop length "<< samples_to_process<<"\n";
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;
    float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

    for(int index = 0; index < samples_to_process; index++)
    {
        halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
        if(halfPtr < 0)                         // to x[n-6].
            halfPtr += LPBUFFER_LGTH ;

        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
        y2 = y1;
        y1 = y0;
        fdatum = y0 * lpbuffer_sqr_div_4;
        lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
        
        hp_y += fdatum - hp_data[hp_ptr];
        halfPtr = hp_ptr - hpbuffer_lgth_half ;
        if(halfPtr < 0)
            halfPtr += HPBUFFER_LGTH ;
        hp_data[hp_ptr] = fdatum ;
        fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
        y = fdatum - derBuff[derI] ;
        derBuff[derI] = fdatum;
        fdatum = y;
        fdatum = fabs(fdatum) ;             // Take the absolute value.
        sum += fdatum - data[ptr] ;
        data[ptr] = fdatum ;

        // #ifdef OPERATION_COUNTER
        // float_comp_counter++;
        // #endif
        if((sum * window_width_inv) > 32000.f)
        {
            output_temp = 32000.f ;
        } 
        else 
        {
            output_temp = sum * window_width_inv;
            // #ifdef OPERATION_COUNTER
            // float_div_counter += 1;
            // #endif
        }

        if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
            lp_ptr = 0 ;                    // the buffer pointer.
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
        
        // #ifdef OPERATION_COUNTER
        //     float_add_counter += 10;
        //     float_mul_counter++;
        //     float_div_counter += 4;
        // #endif
        output[index] = output_temp;
    }
}

void blocking_no_divisions_factorized(float* input, float* output, int samples_to_process) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, index;
    float fdatum, y0, z, y, output_temp;
    float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

    for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
    {
        for(int j=0; j < BLOCKING_SIZE; j++)
        {
            index = i * BLOCKING_SIZE + j;
            halfPtr = lp_ptr - lpbuffer_lgth_half;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = ((y1- lp_data[halfPtr])*2.0f) - y2 + input[index] + lp_data[lp_ptr] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 * lpbuffer_sqr_div_4;
            lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[hp_ptr];
            halfPtr = hp_ptr - hpbuffer_lgth_half ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            hp_data[hp_ptr] = fdatum ;
            fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
            y = fdatum - derBuff[derI] ;
            derBuff[derI] = fdatum;
            fdatum = y;
            fdatum = fabs(fdatum) ;             // Take the absolute value.
            sum += fdatum - data[ptr] ;
            data[ptr] = fdatum ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            if((sum * window_width_inv) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum * window_width_inv ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }

            if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr = 0 ;                    // the buffer pointer.
            if(++derI == DERIV_LENGTH)
                derI = 0 ;
            if(++hp_ptr == HPBUFFER_LGTH)
                hp_ptr = 0 ;
            if(++ptr == WINDOW_WIDTH)
                ptr = 0 ;
            
            // #ifdef OPERATION_COUNTER
            //     float_add_counter += 10;
            //     float_mul_counter++;
            //     float_div_counter += 4;
            // #endif
            output[index] = output_temp;
        }
    }
}

void blocking_no_divisions_unrolled(float* input, float* output, int samples_to_process) 
{
    // data buffer for lpfilt
    static float lp_data[LPBUFFER_LGTH];
    // data buffer for hpfilt
    static float hp_data[HPBUFFER_LGTH];
    // data buffer for derivative
    static float derBuff[DERIV_LENGTH] ;
    // data buffer for moving window average
    static float data[WINDOW_WIDTH];
        
    // ------- initialize filters ------- //
    //lpfilt
    for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
        lp_data[i_init] = 0.f;
    //hpfilt
    for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
        hp_data[i_init] = 0.f;
    //derivative
    for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
        derBuff[i_init] = 0 ;
    //movint window integration
    for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
        data[i_init] = 0 ;
        
    static float y1 = 0.0, y2 = 0.0, y1_2 = 0.0, y2_2 = 0.0, hp_y = 0.0, hp_y2 = 0.0, sum = 0.0;
    static int lp_ptr = 0, lp_ptr2 = 1, hp_ptr = 0, hp_ptr2 = 1, derI = 0, derI2 = 1, ptr = 0, ptr2 = 1;
    int halfPtr, halfPtr2, index, index2;
    float fdatum, fdatum2, y0, y0_2, z, y, output_temp, output_temp2;
    float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    float window_width_inv = 1/ (float)WINDOW_WIDTH;
    int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

    for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
    {
        for(int j=0; j < BLOCKING_SIZE-1; j+=2)
        {
            index = i * BLOCKING_SIZE + j;
            index2 = index + 1;

            halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;
 
            halfPtr2 = lp_ptr2 - lpbuffer_lgth_half ;    // Use halfPtr to index 
            if(halfPtr2 < 0)                         // to x[n-6].
                halfPtr2 += LPBUFFER_LGTH ;

            //lp_data beein overwritten later should not be a problem since halfPtr and lp_ptr are 5 (lpbuff_lgth = 10) apart
            y0 = (y1_2*2.0f) - y2_2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
            y2 = y1_2;
            y1 = y0;
            fdatum = y0 * lpbuffer_sqr_div_4;
            y0_2 = (y1*2.0f) - y2 + input[index2] - (lp_data[halfPtr2]*2.0f) + lp_data[lp_ptr2] ;
            y2_2 = y1;
            y1_2 = y0_2;
            fdatum2 = y0_2 * lpbuffer_sqr_div_4;

            lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            lp_data[lp_ptr2] = input[index2] ;            // Stick most recent sample into
            
            hp_y += fdatum - hp_data[hp_ptr];
            hp_y2 += fdatum2 - hp_data[hp_ptr2];

            halfPtr = hp_ptr - hpbuffer_lgth_half ;
            if(halfPtr < 0)
                halfPtr += HPBUFFER_LGTH ;
            halfPtr2 = hp_ptr2 - hpbuffer_lgth_half ;
            if(halfPtr2 < 0)
                halfPtr2 += HPBUFFER_LGTH ;

            hp_data[hp_ptr] = fdatum ;
            hp_data[hp_ptr2] = fdatum2 ;
            fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
            fdatum2 = hp_data[halfPtr2] - (hp_y2 * hpbuffer_lgth_inv);
            y = fdatum - derBuff[derI] ;
            y2 = fdatum2 - derBuff[derI2] ;

            derBuff[derI] = fdatum;
            derBuff[derI2] = fdatum2;
            fdatum = y;
            fdatum2 = y2;

            fdatum = fabs(fdatum) ;             // Take the absolute value.
            fdatum2 = fabs(fdatum2) ;             // Take the absolute value.


            sum += fdatum - data[ptr] ;
            data[ptr] = fdatum ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            if((sum * window_width_inv) > 32000.f)
            {
                output_temp = 32000.f ;
            } 
            else 
            {
                output_temp = sum * window_width_inv ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }
            output[index] = output_temp;


            sum += fdatum2 - data[ptr2] ;
            data[ptr2] = fdatum2 ;

            // #ifdef OPERATION_COUNTER
            // float_comp_counter++;
            // #endif
            if((sum * window_width_inv) > 32000.f)
            {
                output_temp2 = 32000.f ;
            } 
            else 
            {
                output_temp2 = sum * window_width_inv ;
                // #ifdef OPERATION_COUNTER
                // float_div_counter += 1;
                // #endif
            }
            output[index2] = output_temp2;



            //DERIV_LENGTH is 2, therefore derI and derI2 stay 0 and 1
            lp_ptr += 2;
            lp_ptr2 += 2;
            hp_ptr += 2;
            hp_ptr2 += 2;
            ptr += 2;
            ptr2 += 2;
            // derI += 2;
            // derI2 += 2;

            // the buffer pointer.
            if(lp_ptr >= LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr -= LPBUFFER_LGTH ;
            if(lp_ptr2 >= LPBUFFER_LGTH)   // the circular buffer and update
                lp_ptr2 -= LPBUFFER_LGTH ; 
            // if(derI >= DERIV_LENGTH)
            //     derI -= DERIV_LENGTH ;
            // if(derI2 >= DERIV_LENGTH)
            //     derI2 -= DERIV_LENGTH ;
            if(hp_ptr >= HPBUFFER_LGTH)
                hp_ptr -= HPBUFFER_LGTH ;
            if(hp_ptr2 >= HPBUFFER_LGTH)
                hp_ptr2 -= HPBUFFER_LGTH ;
            if(ptr >= WINDOW_WIDTH)
                ptr -= WINDOW_WIDTH ;
            if(ptr2 >= WINDOW_WIDTH)
                ptr2 -= WINDOW_WIDTH ;
            
            // #ifdef OPERATION_COUNTER
            //     float_add_counter += 10;
            //     float_mul_counter++;
            //     float_div_counter += 4;
            // #endif
        }
    }
}
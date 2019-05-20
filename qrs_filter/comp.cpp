#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matlab_data.h"
#include "../hamilton_inline/qrsdet.h"

//
//The slow base version
//
void slowperformance(float* input, float* output, int sample_length) 
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
        

    static float y1 = 0.0, y2 = 0.0; // this was long, might need to make it double if precision is off
    static float hp_y = 0.0;
    static float sum = 0.0;
    static int lp_ptr = 0;
    static int hp_ptr = 0;
    static int derI = 0;
    static int ptr = 0;
    int halfPtr;
    float fdatum ;
    float y0;
    float z;
    float y;
    float output_temp;
    int index;

    for(int index = 0; index < used_samples; index++)
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




void blocking(float* input, float* output, int sample_length) 
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
        

    static float y1 = 0.0, y2 = 0.0; // this was long, might need to make it double if precision is off
    static float hp_y = 0.0;
    static float sum = 0.0;
    static int lp_ptr = 0;
    static int hp_ptr = 0;
    static int derI = 0;
    static int ptr = 0;
    int halfPtr;
    float fdatum ;
    float y0;
    float z;
    float y;
    float output_temp;
    int index;

    for(int i=0; i < used_samples/sample_length; i++)
    {
        for(int j=0; j < sample_length; j++)
        {
            index = i * (used_samples/sample_length) + j;
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
}
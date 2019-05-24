#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "matlab_data.h"
#include "../hamilton_inline/qrsdet.h"
#include <emmintrin.h>

#define UNROLL_MACRO_TEST(index) {\
        halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)                       \
            halfPtr += LPBUFFER_LGTH ;\
        \
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)\
            halfPtr += HPBUFFER_LGTH ;\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum += fdatum ;\
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        \
        if((sum / (float)WINDOW_WIDTH) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum / (float)WINDOW_WIDTH ;\
        }\
        \
        if(++lp_ptr == LPBUFFER_LGTH)  \
            lp_ptr = 0 ;    \
        if(++derI == DERIV_LENGTH)\
            derI = 0 ;\
        if(++hp_ptr == HPBUFFER_LGTH)\
            hp_ptr = 0 ;\
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define UNROLL_MACRO_LP_PTR(index,lp_ptr) {\
        halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)                       \
            halfPtr += LPBUFFER_LGTH ;\
        \
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)\
            halfPtr += HPBUFFER_LGTH ;\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum += fdatum ;\
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        \
        if((sum / (float)WINDOW_WIDTH) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum / (float)WINDOW_WIDTH ;\
        }\
        \
        if(++derI == DERIV_LENGTH)\
            derI = 0 ;\
        if(++hp_ptr == HPBUFFER_LGTH)\
            hp_ptr = 0 ;\
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define UNROLL_MACRO_LP_DERI(index,lp_ptr,derI) {\
        halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)                       \
            halfPtr += LPBUFFER_LGTH ;\
        \
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)\
            halfPtr += HPBUFFER_LGTH ;\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum += fdatum ;\
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        \
        if((sum / (float)WINDOW_WIDTH) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum / (float)WINDOW_WIDTH ;\
        }\
        \
        if(++hp_ptr == HPBUFFER_LGTH)\
            hp_ptr = 0 ;\
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define UNROLL_MACRO_LP_DERI_HP(index,lp_ptr,derI,hp_ptr) {\
        halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)                       \
            halfPtr += LPBUFFER_LGTH ;\
        \
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)\
            halfPtr += HPBUFFER_LGTH ;\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        sum += fdatum ;\
        \
        if((sum / (float)WINDOW_WIDTH) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum / (float)WINDOW_WIDTH ;\
        }\
        \
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define UNROLL_MACRO_LP_DERI_HP_PTR(index,lp_ptr,derI,hp_ptr,ptr) {\
        halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)                       \
            halfPtr += LPBUFFER_LGTH ;\
        \
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;\
        if(halfPtr < 0)\
            halfPtr += HPBUFFER_LGTH ;\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        sum += fdatum ;\
        \
        if((sum / (float)WINDOW_WIDTH) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum / (float)WINDOW_WIDTH ;\
        }\
        }

#define UNROLL_MACRO_LP_DERI_HP_HALF(index,lp_ptr,derI,hp_ptr,halfPtr1,halfPtr2) {\
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr1]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr2] - (hp_y / (float)HPBUFFER_LGTH);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        sum += fdatum ;\
        \
        if((sum / (float)WINDOW_WIDTH) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum / (float)WINDOW_WIDTH ;\
        }\
        \
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index,lp_ptr,derI,hp_ptr,halfPtr1,halfPtr2) {\
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr1]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        hp_data[hp_ptr] = fdatum ;\
        fdatum2 = hp_data[halfPtr2] - (hp_y / (float)HPBUFFER_LGTH);\
        y = fdatum2 - derBuff[derI] ;\
        derBuff[derI] = fdatum2;\
        fdatum3 = y;\
        fdatum3 = fabs(fdatum3) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum3 ;\
        sum += fdatum3 ;\
        \
        if((sum / (float)WINDOW_WIDTH) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum / (float)WINDOW_WIDTH ;\
        }\
        \
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index,lp_ptr,derI,hp_ptr,halfPtr1,halfPtr2) {\
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr1]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 * 4.0f * LPBUFFER_LGTH_INV * LPBUFFER_LGTH_INV;\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        hp_data[hp_ptr] = fdatum ;\
        fdatum2 = hp_data[halfPtr2] - (hp_y * HPBUFFER_LGTH_INV);\
        y = fdatum2 - derBuff[derI] ;\
        derBuff[derI] = fdatum2;\
        fdatum3 = y;\
        fdatum3 = fabs(fdatum3) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum3 ;\
        sum += fdatum3 ;\
        sum_temp = sum * WINDOW_WIDTH_INV;\
        if(sum_temp > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum_temp;\
        }\
        \
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index,lp_ptr,derI,hp_ptr,halfPtr1,halfPtr2) {\
        y0 = (y1*2.0f) - y2 + datum[index] - (lp_data[halfPtr1]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 * lpbuffer_sqr_div_4;\
        lp_data[lp_ptr] = datum[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        hp_data[hp_ptr] = fdatum ;\
        fdatum2 = hp_data[halfPtr2] - (hp_y * HPBUFFER_LGTH_INV);\
        y = fdatum2 - derBuff[derI] ;\
        derBuff[derI] = fdatum2;\
        fdatum3 = y;\
        fdatum3 = fabs(fdatum3) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum3 ;\
        sum += fdatum3 ;\
        sum_temp = sum * WINDOW_WIDTH_INV;\
        if(sum_temp > 32000.f)\
        {\
            filtOutput[index] = 32000.f ;\
        } \
        else \
        {\
            filtOutput[index] = sum_temp;\
        }\
        \
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index,lp_ptr,derI,hp_ptr,halfPtr1,halfPtr2) {\
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr1]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 * LPBUFFER_LGTH_INV*LPBUFFER_LGTH_INV*4.0f;\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr2] - (hp_y * HPBUFFER_LGTH_INV);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        sum += fdatum ;\
        \
        if((sum * WINDOW_WIDTH_INV) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum * WINDOW_WIDTH_INV ;\
        }\
        \
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index,lp_ptr,derI,hp_ptr,halfPtr1,halfPtr2) {\
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr1]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 * LPBUFFER_LGTH_INV_SQR;\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr2] - (hp_y * HPBUFFER_LGTH_INV);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        sum += fdatum ;\
        \
        if((sum * WINDOW_WIDTH_INV) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum * WINDOW_WIDTH_INV ;\
        }\
        \
        if(++ptr == WINDOW_WIDTH)\
            ptr = 0 ;\
        }

#define UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index,lp_ptr,derI,hp_ptr,ptr,halfPtr1,halfPtr2) {\
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr1]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr2] - (hp_y / (float)HPBUFFER_LGTH);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        sum += fdatum ;\
        \
        if((sum / (float)WINDOW_WIDTH) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum / (float)WINDOW_WIDTH ;\
        }\
        }

#define UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index,lp_ptr,derI,hp_ptr,ptr,halfPtr1,halfPtr2) {\
        y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr1]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 * LPBUFFER_LGTH_INV * LPBUFFER_LGTH_INV * 4.0f;\
        lp_data[lp_ptr] = input[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr2] - (hp_y * HPBUFFER_LGTH_INV);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        sum += fdatum ;\
        sum_temp = sum * WINDOW_WIDTH_INV;\
        if((sum_temp) > 32000.f)\
        {\
            output[index] = 32000.f ;\
        } \
        else \
        {\
            output[index] = sum_temp ;\
        }\
        }

#define RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index,lp_ptr,derI,hp_ptr,ptr,halfPtr1,halfPtr2) {\
        y0 = (y1*2.0f) - y2 + datum[index] - (lp_data[halfPtr1]*2.0f) + lp_data[lp_ptr] ;\
        y2 = y1;\
        y1 = y0;\
        fdatum = y0 * lpbuffer_sqr_div_4;\
        lp_data[lp_ptr] = datum[index] ;           \
        \
        hp_y += fdatum - hp_data[hp_ptr];\
        hp_data[hp_ptr] = fdatum ;\
        fdatum = hp_data[halfPtr2] - (hp_y * HPBUFFER_LGTH_INV);\
        y = fdatum - derBuff[derI] ;\
        derBuff[derI] = fdatum;\
        fdatum = y;\
        fdatum = fabs(fdatum) ;            \
        sum -= data[ptr] ;\
        data[ptr] = fdatum ;\
        sum += fdatum ;\
        sum_temp = sum * WINDOW_WIDTH_INV;\
        if((sum_temp) > 32000.f)\
        {\
            filtOutput[index] = 32000.f ;\
        } \
        else \
        {\
            filtOutput[index] = sum_temp ;\
        }\
        }



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

void slowperformance_macro_test(float* input, float* output, int samples_to_process) 
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
        UNROLL_MACRO_TEST(index)
    }
}

void slowperformance_macro_lp(float* input, float* output, int samples_to_process) 
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

    for(index = 0; index <= samples_to_process- 10 ; index+=10)
    {
        UNROLL_MACRO_LP_PTR(index   ,0  )
        UNROLL_MACRO_LP_PTR(index+1 ,1  )
        UNROLL_MACRO_LP_PTR(index+2 ,2  )
        UNROLL_MACRO_LP_PTR(index+3 ,3  )
        UNROLL_MACRO_LP_PTR(index+4 ,4  )
        UNROLL_MACRO_LP_PTR(index+5 ,5  )
        UNROLL_MACRO_LP_PTR(index+6 ,6  )
        UNROLL_MACRO_LP_PTR(index+7 ,7  )
        UNROLL_MACRO_LP_PTR(index+8 ,8  )
        UNROLL_MACRO_LP_PTR(index+9 ,9  )
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output_temp = 32000.f ;
        } 
        else 
        {
            output_temp = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
        output[index] = output_temp;
    }
}

void slowperformance_macro_lp_deri(float* input, float* output, int samples_to_process) 
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

    for(index = 0; index <= samples_to_process- 10 ; index+=10)
    {
        UNROLL_MACRO_LP_DERI(index   ,0  ,0)
        UNROLL_MACRO_LP_DERI(index+1 ,1  ,1)
        UNROLL_MACRO_LP_DERI(index+2 ,2  ,0)
        UNROLL_MACRO_LP_DERI(index+3 ,3  ,1)
        UNROLL_MACRO_LP_DERI(index+4 ,4  ,0)
        UNROLL_MACRO_LP_DERI(index+5 ,5  ,1)
        UNROLL_MACRO_LP_DERI(index+6 ,6  ,0)
        UNROLL_MACRO_LP_DERI(index+7 ,7  ,1)
        UNROLL_MACRO_LP_DERI(index+8 ,8  ,0)
        UNROLL_MACRO_LP_DERI(index+9 ,9  ,1)
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output_temp = 32000.f ;
        } 
        else 
        {
            output_temp = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
        output[index] = output_temp;
    }
}

void slowperformance_macro_lp_deri_hp(float* input, float* output, int samples_to_process) 
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
    float fdatum, y0, z, y;

    for(index = 0; index <= samples_to_process - 50 ; index+=50)
    {
        UNROLL_MACRO_LP_DERI_HP(index       ,0  ,0  ,0)
        UNROLL_MACRO_LP_DERI_HP(index+1     ,1  ,1  ,1)
        UNROLL_MACRO_LP_DERI_HP(index+2     ,2  ,0  ,2)
        UNROLL_MACRO_LP_DERI_HP(index+3     ,3  ,1  ,3)
        UNROLL_MACRO_LP_DERI_HP(index+4     ,4  ,0  ,4)
        UNROLL_MACRO_LP_DERI_HP(index+5     ,5  ,1  ,5)
        UNROLL_MACRO_LP_DERI_HP(index+6     ,6  ,0  ,6)
        UNROLL_MACRO_LP_DERI_HP(index+7     ,7  ,1  ,7)
        UNROLL_MACRO_LP_DERI_HP(index+8     ,8  ,0  ,8)
        UNROLL_MACRO_LP_DERI_HP(index+9     ,9  ,1  ,9)
        UNROLL_MACRO_LP_DERI_HP(index+10    ,0  ,0  ,10)
        UNROLL_MACRO_LP_DERI_HP(index+11    ,1  ,1  ,11)
        UNROLL_MACRO_LP_DERI_HP(index+12    ,2  ,0  ,12)
        UNROLL_MACRO_LP_DERI_HP(index+13    ,3  ,1  ,13)
        UNROLL_MACRO_LP_DERI_HP(index+14    ,4  ,0  ,14)
        UNROLL_MACRO_LP_DERI_HP(index+15    ,5  ,1  ,15)
        UNROLL_MACRO_LP_DERI_HP(index+16    ,6  ,0  ,16)
        UNROLL_MACRO_LP_DERI_HP(index+17    ,7  ,1  ,17)
        UNROLL_MACRO_LP_DERI_HP(index+18    ,8  ,0  ,18)
        UNROLL_MACRO_LP_DERI_HP(index+19    ,9  ,1  ,19)
        UNROLL_MACRO_LP_DERI_HP(index+20    ,0  ,0  ,20)
        UNROLL_MACRO_LP_DERI_HP(index+21    ,1  ,1  ,21)
        UNROLL_MACRO_LP_DERI_HP(index+22    ,2  ,0  ,22)
        UNROLL_MACRO_LP_DERI_HP(index+23    ,3  ,1  ,23)
        UNROLL_MACRO_LP_DERI_HP(index+24    ,4  ,0  ,24)
        UNROLL_MACRO_LP_DERI_HP(index+25    ,5  ,1  ,0)
        UNROLL_MACRO_LP_DERI_HP(index+26    ,6  ,0  ,1)
        UNROLL_MACRO_LP_DERI_HP(index+27    ,7  ,1  ,2)
        UNROLL_MACRO_LP_DERI_HP(index+28    ,8  ,0  ,3)
        UNROLL_MACRO_LP_DERI_HP(index+29    ,9  ,1  ,4)
        UNROLL_MACRO_LP_DERI_HP(index+30    ,0  ,0  ,5)
        UNROLL_MACRO_LP_DERI_HP(index+31    ,1  ,1  ,6)
        UNROLL_MACRO_LP_DERI_HP(index+32    ,2  ,0  ,7)
        UNROLL_MACRO_LP_DERI_HP(index+33    ,3  ,1  ,8)
        UNROLL_MACRO_LP_DERI_HP(index+34    ,4  ,0  ,9)
        UNROLL_MACRO_LP_DERI_HP(index+35    ,5  ,1  ,10)
        UNROLL_MACRO_LP_DERI_HP(index+36    ,6  ,0  ,11)
        UNROLL_MACRO_LP_DERI_HP(index+37    ,7  ,1  ,12)
        UNROLL_MACRO_LP_DERI_HP(index+38    ,8  ,0  ,13)
        UNROLL_MACRO_LP_DERI_HP(index+39    ,9  ,1  ,14)
        UNROLL_MACRO_LP_DERI_HP(index+40    ,0  ,0  ,15)
        UNROLL_MACRO_LP_DERI_HP(index+41    ,1  ,1  ,16)
        UNROLL_MACRO_LP_DERI_HP(index+42    ,2  ,0  ,17)
        UNROLL_MACRO_LP_DERI_HP(index+43    ,3  ,1  ,18)
        UNROLL_MACRO_LP_DERI_HP(index+44    ,4  ,0  ,19)
        UNROLL_MACRO_LP_DERI_HP(index+45    ,5  ,1  ,20)
        UNROLL_MACRO_LP_DERI_HP(index+46    ,6  ,0  ,21)
        UNROLL_MACRO_LP_DERI_HP(index+47    ,7  ,1  ,22)
        UNROLL_MACRO_LP_DERI_HP(index+48    ,8  ,0  ,23)
        UNROLL_MACRO_LP_DERI_HP(index+49    ,9  ,1  ,24)
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output[index] = 32000.f ;
        } 
        else 
        {
            output[index] = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
    }
}

void slowperformance_macro_lp_deri_hp_half(float* input, float* output, int samples_to_process) 
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
    int halfPtr, halfPtr1, halfPtr2, index;
    float fdatum, y0, z, y;

    for(index = 0; index <= samples_to_process - 50 ; index+=50)
    {
        UNROLL_MACRO_LP_DERI_HP_HALF(index       ,0  ,0  ,0  ,5  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+1     ,1  ,1  ,1  ,6  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+2     ,2  ,0  ,2  ,7  ,15)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+3     ,3  ,1  ,3  ,8  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+4     ,4  ,0  ,4  ,9  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+5     ,5  ,1  ,5  ,0  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+6     ,6  ,0  ,6  ,1  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+7     ,7  ,1  ,7  ,2  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+8     ,8  ,0  ,8  ,3  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+9     ,9  ,1  ,9  ,4  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+10    ,0  ,0  ,10 ,5  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+11    ,1  ,1  ,11 ,6  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+12    ,2  ,0  ,12 ,7  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+13    ,3  ,1  ,13 ,8  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+14    ,4  ,0  ,14 ,9  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+15    ,5  ,1  ,15 ,0  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+16    ,6  ,0  ,16 ,1  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+17    ,7  ,1  ,17 ,2  ,5) 
        UNROLL_MACRO_LP_DERI_HP_HALF(index+18    ,8  ,0  ,18 ,3  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+19    ,9  ,1  ,19 ,4  ,7)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+20    ,0  ,0  ,20 ,5  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+21    ,1  ,1  ,21 ,6  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+22    ,2  ,0  ,22 ,7  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+23    ,3  ,1  ,23 ,8  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+24    ,4  ,0  ,24 ,9  ,12)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+25    ,5  ,1  ,0  ,0  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+26    ,6  ,0  ,1  ,1  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+27    ,7  ,1  ,2  ,2  ,15) 
        UNROLL_MACRO_LP_DERI_HP_HALF(index+28    ,8  ,0  ,3  ,3  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+29    ,9  ,1  ,4  ,4  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+30    ,0  ,0  ,5  ,5  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+31    ,1  ,1  ,6  ,6  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+32    ,2  ,0  ,7  ,7  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+33    ,3  ,1  ,8  ,8  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+34    ,4  ,0  ,9  ,9  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+35    ,5  ,1  ,10 ,0  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+36    ,6  ,0  ,11 ,1  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+37    ,7  ,1  ,12 ,2  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+38    ,8  ,0  ,13 ,3  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+39    ,9  ,1  ,14 ,4  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+40    ,0  ,0  ,15 ,5  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+41    ,1  ,1  ,16 ,6  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+42    ,2  ,0  ,17 ,7  ,5)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+43    ,3  ,1  ,18 ,8  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+44    ,4  ,0  ,19 ,9  ,7) 
        UNROLL_MACRO_LP_DERI_HP_HALF(index+45    ,5  ,1  ,20 ,0  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+46    ,6  ,0  ,21 ,1  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+47    ,7  ,1  ,22 ,2  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+48    ,8  ,0  ,23 ,3  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF(index+49    ,9  ,1  ,24 ,4  ,12)
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output[index] = 32000.f ;
        } 
        else 
        {
            output[index] = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
    }
}

void slowperformance_macro_lp_deri_hp_half_dependencies(float* input, float* output, int samples_to_process) 
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
    int halfPtr, halfPtr1, halfPtr2, index;
    float fdatum, fdatum2, fdatum3, y0, z, y;

    for(index = 0; index <= samples_to_process - 50 ; index+=50)
    {
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index       ,0  ,0  ,0  ,5  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+1     ,1  ,1  ,1  ,6  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+2     ,2  ,0  ,2  ,7  ,15)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+3     ,3  ,1  ,3  ,8  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+4     ,4  ,0  ,4  ,9  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+5     ,5  ,1  ,5  ,0  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+6     ,6  ,0  ,6  ,1  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+7     ,7  ,1  ,7  ,2  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+8     ,8  ,0  ,8  ,3  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+9     ,9  ,1  ,9  ,4  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+10    ,0  ,0  ,10 ,5  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+11    ,1  ,1  ,11 ,6  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+12    ,2  ,0  ,12 ,7  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+13    ,3  ,1  ,13 ,8  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+14    ,4  ,0  ,14 ,9  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+15    ,5  ,1  ,15 ,0  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+16    ,6  ,0  ,16 ,1  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+17    ,7  ,1  ,17 ,2  ,5) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+18    ,8  ,0  ,18 ,3  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+19    ,9  ,1  ,19 ,4  ,7)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+20    ,0  ,0  ,20 ,5  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+21    ,1  ,1  ,21 ,6  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+22    ,2  ,0  ,22 ,7  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+23    ,3  ,1  ,23 ,8  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+24    ,4  ,0  ,24 ,9  ,12)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+25    ,5  ,1  ,0  ,0  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+26    ,6  ,0  ,1  ,1  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+27    ,7  ,1  ,2  ,2  ,15) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+28    ,8  ,0  ,3  ,3  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+29    ,9  ,1  ,4  ,4  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+30    ,0  ,0  ,5  ,5  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+31    ,1  ,1  ,6  ,6  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+32    ,2  ,0  ,7  ,7  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+33    ,3  ,1  ,8  ,8  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+34    ,4  ,0  ,9  ,9  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+35    ,5  ,1  ,10 ,0  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+36    ,6  ,0  ,11 ,1  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+37    ,7  ,1  ,12 ,2  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+38    ,8  ,0  ,13 ,3  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+39    ,9  ,1  ,14 ,4  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+40    ,0  ,0  ,15 ,5  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+41    ,1  ,1  ,16 ,6  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+42    ,2  ,0  ,17 ,7  ,5)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+43    ,3  ,1  ,18 ,8  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+44    ,4  ,0  ,19 ,9  ,7) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+45    ,5  ,1  ,20 ,0  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+46    ,6  ,0  ,21 ,1  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+47    ,7  ,1  ,22 ,2  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+48    ,8  ,0  ,23 ,3  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES(index+49    ,9  ,1  ,24 ,4  ,12)
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output[index] = 32000.f ;
        } 
        else 
        {
            output[index] = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
    }
}

void slowperformance_macro_lp_deri_hp_half_dependencies_div(float* input, float* output, int samples_to_process) 
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
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0, sum_temp = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, halfPtr1, halfPtr2, index;
    float fdatum, fdatum2, fdatum3, y0, z, y;
    float LPBUFFER_LGTH_INV = 1/((float)LPBUFFER_LGTH);
    float HPBUFFER_LGTH_INV = 1/((float)HPBUFFER_LGTH);
    float WINDOW_WIDTH_INV = 1/((float)WINDOW_WIDTH);

    for(index = 0; index <= samples_to_process - 50 ; index+=50)
    {
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index       ,0  ,0  ,0  ,5  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+1     ,1  ,1  ,1  ,6  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+2     ,2  ,0  ,2  ,7  ,15)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+3     ,3  ,1  ,3  ,8  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+4     ,4  ,0  ,4  ,9  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+5     ,5  ,1  ,5  ,0  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+6     ,6  ,0  ,6  ,1  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+7     ,7  ,1  ,7  ,2  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+8     ,8  ,0  ,8  ,3  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+9     ,9  ,1  ,9  ,4  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+10    ,0  ,0  ,10 ,5  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+11    ,1  ,1  ,11 ,6  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+12    ,2  ,0  ,12 ,7  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+13    ,3  ,1  ,13 ,8  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+14    ,4  ,0  ,14 ,9  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+15    ,5  ,1  ,15 ,0  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+16    ,6  ,0  ,16 ,1  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+17    ,7  ,1  ,17 ,2  ,5) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+18    ,8  ,0  ,18 ,3  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+19    ,9  ,1  ,19 ,4  ,7)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+20    ,0  ,0  ,20 ,5  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+21    ,1  ,1  ,21 ,6  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+22    ,2  ,0  ,22 ,7  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+23    ,3  ,1  ,23 ,8  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+24    ,4  ,0  ,24 ,9  ,12)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+25    ,5  ,1  ,0  ,0  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+26    ,6  ,0  ,1  ,1  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+27    ,7  ,1  ,2  ,2  ,15) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+28    ,8  ,0  ,3  ,3  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+29    ,9  ,1  ,4  ,4  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+30    ,0  ,0  ,5  ,5  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+31    ,1  ,1  ,6  ,6  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+32    ,2  ,0  ,7  ,7  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+33    ,3  ,1  ,8  ,8  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+34    ,4  ,0  ,9  ,9  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+35    ,5  ,1  ,10 ,0  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+36    ,6  ,0  ,11 ,1  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+37    ,7  ,1  ,12 ,2  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+38    ,8  ,0  ,13 ,3  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+39    ,9  ,1  ,14 ,4  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+40    ,0  ,0  ,15 ,5  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+41    ,1  ,1  ,16 ,6  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+42    ,2  ,0  ,17 ,7  ,5)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+43    ,3  ,1  ,18 ,8  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+44    ,4  ,0  ,19 ,9  ,7) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+45    ,5  ,1  ,20 ,0  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+46    ,6  ,0  ,21 ,1  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+47    ,7  ,1  ,22 ,2  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+48    ,8  ,0  ,23 ,3  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+49    ,9  ,1  ,24 ,4  ,12)
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output[index] = 32000.f ;
        } 
        else 
        {
            output[index] = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
    }
}

void slowperformance_macro_lp_deri_hp_half_div(float* input, float* output, int samples_to_process) 
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
    int halfPtr, halfPtr1, halfPtr2, index;
    float fdatum, y0, z, y;
    float LPBUFFER_LGTH_INV = 1/((float)LPBUFFER_LGTH);
    float HPBUFFER_LGTH_INV = 1/((float)HPBUFFER_LGTH);
    float WINDOW_WIDTH_INV = 1/((float)WINDOW_WIDTH);

    for(index = 0; index <= samples_to_process - 50 ; index+=50)
    {
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index       ,0  ,0  ,0  ,5  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+1     ,1  ,1  ,1  ,6  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+2     ,2  ,0  ,2  ,7  ,15)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+3     ,3  ,1  ,3  ,8  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+4     ,4  ,0  ,4  ,9  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+5     ,5  ,1  ,5  ,0  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+6     ,6  ,0  ,6  ,1  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+7     ,7  ,1  ,7  ,2  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+8     ,8  ,0  ,8  ,3  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+9     ,9  ,1  ,9  ,4  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+10    ,0  ,0  ,10 ,5  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+11    ,1  ,1  ,11 ,6  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+12    ,2  ,0  ,12 ,7  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+13    ,3  ,1  ,13 ,8  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+14    ,4  ,0  ,14 ,9  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+15    ,5  ,1  ,15 ,0  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+16    ,6  ,0  ,16 ,1  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+17    ,7  ,1  ,17 ,2  ,5) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+18    ,8  ,0  ,18 ,3  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+19    ,9  ,1  ,19 ,4  ,7)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+20    ,0  ,0  ,20 ,5  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+21    ,1  ,1  ,21 ,6  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+22    ,2  ,0  ,22 ,7  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+23    ,3  ,1  ,23 ,8  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+24    ,4  ,0  ,24 ,9  ,12)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+25    ,5  ,1  ,0  ,0  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+26    ,6  ,0  ,1  ,1  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+27    ,7  ,1  ,2  ,2  ,15) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+28    ,8  ,0  ,3  ,3  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+29    ,9  ,1  ,4  ,4  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+30    ,0  ,0  ,5  ,5  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+31    ,1  ,1  ,6  ,6  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+32    ,2  ,0  ,7  ,7  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+33    ,3  ,1  ,8  ,8  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+34    ,4  ,0  ,9  ,9  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+35    ,5  ,1  ,10 ,0  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+36    ,6  ,0  ,11 ,1  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+37    ,7  ,1  ,12 ,2  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+38    ,8  ,0  ,13 ,3  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+39    ,9  ,1  ,14 ,4  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+40    ,0  ,0  ,15 ,5  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+41    ,1  ,1  ,16 ,6  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+42    ,2  ,0  ,17 ,7  ,5)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+43    ,3  ,1  ,18 ,8  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+44    ,4  ,0  ,19 ,9  ,7) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+45    ,5  ,1  ,20 ,0  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+46    ,6  ,0  ,21 ,1  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+47    ,7  ,1  ,22 ,2  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+48    ,8  ,0  ,23 ,3  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV(index+49    ,9  ,1  ,24 ,4  ,12)
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output[index] = 32000.f ;
        } 
        else 
        {
            output[index] = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
    }
}

void slowperformance_macro_lp_deri_hp_half_div_const_replace(float* input, float* output, int samples_to_process) 
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
    int halfPtr, halfPtr1, halfPtr2, index;
    float fdatum, y0, z, y;
    float LPBUFFER_LGTH_INV_SQR = 1/((float)LPBUFFER_LGTH*(float)LPBUFFER_LGTH*0.25);
    float HPBUFFER_LGTH_INV = 1/((float)HPBUFFER_LGTH);
    float WINDOW_WIDTH_INV = 1/((float)WINDOW_WIDTH);

    for(index = 0; index <= samples_to_process - 50 ; index+=50)
    {
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index       ,0  ,0  ,0  ,5  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+1     ,1  ,1  ,1  ,6  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+2     ,2  ,0  ,2  ,7  ,15)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+3     ,3  ,1  ,3  ,8  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+4     ,4  ,0  ,4  ,9  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+5     ,5  ,1  ,5  ,0  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+6     ,6  ,0  ,6  ,1  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+7     ,7  ,1  ,7  ,2  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+8     ,8  ,0  ,8  ,3  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+9     ,9  ,1  ,9  ,4  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+10    ,0  ,0  ,10 ,5  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+11    ,1  ,1  ,11 ,6  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+12    ,2  ,0  ,12 ,7  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+13    ,3  ,1  ,13 ,8  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+14    ,4  ,0  ,14 ,9  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+15    ,5  ,1  ,15 ,0  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+16    ,6  ,0  ,16 ,1  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+17    ,7  ,1  ,17 ,2  ,5) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+18    ,8  ,0  ,18 ,3  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+19    ,9  ,1  ,19 ,4  ,7)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+20    ,0  ,0  ,20 ,5  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+21    ,1  ,1  ,21 ,6  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+22    ,2  ,0  ,22 ,7  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+23    ,3  ,1  ,23 ,8  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+24    ,4  ,0  ,24 ,9  ,12)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+25    ,5  ,1  ,0  ,0  ,13)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+26    ,6  ,0  ,1  ,1  ,14)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+27    ,7  ,1  ,2  ,2  ,15) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+28    ,8  ,0  ,3  ,3  ,16)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+29    ,9  ,1  ,4  ,4  ,17)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+30    ,0  ,0  ,5  ,5  ,18)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+31    ,1  ,1  ,6  ,6  ,19)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+32    ,2  ,0  ,7  ,7  ,20)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+33    ,3  ,1  ,8  ,8  ,21)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+34    ,4  ,0  ,9  ,9  ,22)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+35    ,5  ,1  ,10 ,0  ,23)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+36    ,6  ,0  ,11 ,1  ,24)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+37    ,7  ,1  ,12 ,2  ,0)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+38    ,8  ,0  ,13 ,3  ,1)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+39    ,9  ,1  ,14 ,4  ,2)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+40    ,0  ,0  ,15 ,5  ,3)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+41    ,1  ,1  ,16 ,6  ,4)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+42    ,2  ,0  ,17 ,7  ,5)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+43    ,3  ,1  ,18 ,8  ,6)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+44    ,4  ,0  ,19 ,9  ,7) 
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+45    ,5  ,1  ,20 ,0  ,8)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+46    ,6  ,0  ,21 ,1  ,9)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+47    ,7  ,1  ,22 ,2  ,10)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+48    ,8  ,0  ,23 ,3  ,11)
        UNROLL_MACRO_LP_DERI_HP_HALF_DIV_CONST_REPLACE(index+49    ,9  ,1  ,24 ,4  ,12)
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output[index] = 32000.f ;
        } 
        else 
        {
            output[index] = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
    }
}


void slowperformance_macro_lp_deri_hp_ptr(float* input, float* output, int samples_to_process) 
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
    float fdatum, y0, z, y;

    for(index = 0; index <= samples_to_process - 400 ; index+=400)
    {
        UNROLL_MACRO_LP_DERI_HP_PTR(index       ,0  ,0  ,0  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+1     ,1  ,1  ,1  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+2     ,2  ,0  ,2  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+3     ,3  ,1  ,3  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+4     ,4  ,0  ,4  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+5     ,5  ,1  ,5  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+6     ,6  ,0  ,6  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+7     ,7  ,1  ,7  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+8     ,8  ,0  ,8  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+9     ,9  ,1  ,9  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+10    ,0  ,0  ,10 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+11    ,1  ,1  ,11 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+12    ,2  ,0  ,12 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+13    ,3  ,1  ,13 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+14    ,4  ,0  ,14 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+15    ,5  ,1  ,15 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+16    ,6  ,0  ,16 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+17    ,7  ,1  ,17 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+18    ,8  ,0  ,18 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+19    ,9  ,1  ,19 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+20    ,0  ,0  ,20 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+21    ,1  ,1  ,21 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+22    ,2  ,0  ,22 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+23    ,3  ,1  ,23 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+24    ,4  ,0  ,24 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+25    ,5  ,1  ,0  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+26    ,6  ,0  ,1  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+27    ,7  ,1  ,2  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+28    ,8  ,0  ,3  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+29    ,9  ,1  ,4  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+30    ,0  ,0  ,5  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+31    ,1  ,1  ,6  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+32    ,2  ,0  ,7  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+33    ,3  ,1  ,8  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+34    ,4  ,0  ,9  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+35    ,5  ,1  ,10 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+36    ,6  ,0  ,11 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+37    ,7  ,1  ,12 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+38    ,8  ,0  ,13 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+39    ,9  ,1  ,14 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+40    ,0  ,0  ,15 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+41    ,1  ,1  ,16 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+42    ,2  ,0  ,17 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+43    ,3  ,1  ,18 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+44    ,4  ,0  ,19 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+45    ,5  ,1  ,20 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+46    ,6  ,0  ,21 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+47    ,7  ,1  ,22 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+48    ,8  ,0  ,23 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+49    ,9  ,1  ,24 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+50    ,0  ,0  ,0  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+51    ,1  ,1  ,1  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+52    ,2  ,0  ,2  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+53    ,3  ,1  ,3  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+54    ,4  ,0  ,4  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+55    ,5  ,1  ,5  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+56    ,6  ,0  ,6  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+57    ,7  ,1  ,7  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+58    ,8  ,0  ,8  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+59    ,9  ,1  ,9  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+60    ,0  ,0  ,10 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+61    ,1  ,1  ,11 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+62    ,2  ,0  ,12 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+63    ,3  ,1  ,13 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+64    ,4  ,0  ,14 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+65    ,5  ,1  ,15 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+66    ,6  ,0  ,16 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+67    ,7  ,1  ,17 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+68    ,8  ,0  ,18 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+69    ,9  ,1  ,19 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+70    ,0  ,0  ,20 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+71    ,1  ,1  ,21 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+72    ,2  ,0  ,22 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+73    ,3  ,1  ,23 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+74    ,4  ,0  ,24 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+75    ,5  ,1  ,0  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+76    ,6  ,0  ,1  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+77    ,7  ,1  ,2  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+78    ,8  ,0  ,3  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+79    ,9  ,1  ,4  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+80    ,0  ,0  ,5  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+81    ,1  ,1  ,6  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+82    ,2  ,0  ,7  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+83    ,3  ,1  ,8  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+84    ,4  ,0  ,9  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+85    ,5  ,1  ,10 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+86    ,6  ,0  ,11 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+87    ,7  ,1  ,12 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+88    ,8  ,0  ,13 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+89    ,9  ,1  ,14 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+90    ,0  ,0  ,15 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+91    ,1  ,1  ,16 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+92    ,2  ,0  ,17 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+93    ,3  ,1  ,18 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+94    ,4  ,0  ,19 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+95    ,5  ,1  ,20 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+96    ,6  ,0  ,21 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+97    ,7  ,1  ,22 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+98    ,8  ,0  ,23 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+99    ,9  ,1  ,24 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+100   ,0  ,0  ,0  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+101   ,1  ,1  ,1  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+102   ,2  ,0  ,2  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+103   ,3  ,1  ,3  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+104   ,4  ,0  ,4  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+105   ,5  ,1  ,5  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+106   ,6  ,0  ,6  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+107   ,7  ,1  ,7  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+108   ,8  ,0  ,8  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+109   ,9  ,1  ,9  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+110   ,0  ,0  ,10 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+111   ,1  ,1  ,11 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+112   ,2  ,0  ,12 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+113   ,3  ,1  ,13 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+114   ,4  ,0  ,14 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+115   ,5  ,1  ,15 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+116   ,6  ,0  ,16 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+117   ,7  ,1  ,17 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+118   ,8  ,0  ,18 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+119   ,9  ,1  ,19 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+120   ,0  ,0  ,20 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+121   ,1  ,1  ,21 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+122   ,2  ,0  ,22 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+123   ,3  ,1  ,23 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+124   ,4  ,0  ,24 ,12 )    
        UNROLL_MACRO_LP_DERI_HP_PTR(index+125   ,5  ,1  ,0  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+126   ,6  ,0  ,1  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+127   ,7  ,1  ,2  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+128   ,8  ,0  ,3  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+129   ,9  ,1  ,4  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+130   ,0  ,0  ,5  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+131   ,1  ,1  ,6  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+132   ,2  ,0  ,7  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+133   ,3  ,1  ,8  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+134   ,4  ,0  ,9  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+135   ,5  ,1  ,10 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+136   ,6  ,0  ,11 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+137   ,7  ,1  ,12 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+138   ,8  ,0  ,13 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+139   ,9  ,1  ,14 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+140   ,0  ,0  ,15 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+141   ,1  ,1  ,16 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+142   ,2  ,0  ,17 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+143   ,3  ,1  ,18 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+144   ,4  ,0  ,19 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+145   ,5  ,1  ,20 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+146   ,6  ,0  ,21 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+147   ,7  ,1  ,22 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+148   ,8  ,0  ,23 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+149   ,9  ,1  ,24 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+150   ,0  ,0  ,0  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+151   ,1  ,1  ,1  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+152   ,2  ,0  ,2  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+153   ,3  ,1  ,3  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+154   ,4  ,0  ,4  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+155   ,5  ,1  ,5  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+156   ,6  ,0  ,6  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+157   ,7  ,1  ,7  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+158   ,8  ,0  ,8  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+159   ,9  ,1  ,9  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+160   ,0  ,0  ,10 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+161   ,1  ,1  ,11 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+162   ,2  ,0  ,12 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+163   ,3  ,1  ,13 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+164   ,4  ,0  ,14 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+165   ,5  ,1  ,15 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+166   ,6  ,0  ,16 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+167   ,7  ,1  ,17 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+168   ,8  ,0  ,18 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+169   ,9  ,1  ,19 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+170   ,0  ,0  ,20 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+171   ,1  ,1  ,21 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+172   ,2  ,0  ,22 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+173   ,3  ,1  ,23 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+174   ,4  ,0  ,24 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+175   ,5  ,1  ,0  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+176   ,6  ,0  ,1  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+177   ,7  ,1  ,2  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+178   ,8  ,0  ,3  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+179   ,9  ,1  ,4  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+180   ,0  ,0  ,5  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+181   ,1  ,1  ,6  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+182   ,2  ,0  ,7  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+183   ,3  ,1  ,8  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+184   ,4  ,0  ,9  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+185   ,5  ,1  ,10 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+186   ,6  ,0  ,11 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+187   ,7  ,1  ,12 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+188   ,8  ,0  ,13 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+189   ,9  ,1  ,14 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+190   ,0  ,0  ,15 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+191   ,1  ,1  ,16 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+192   ,2  ,0  ,17 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+193   ,3  ,1  ,18 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+194   ,4  ,0  ,19 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+195   ,5  ,1  ,20 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+196   ,6  ,0  ,21 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+197   ,7  ,1  ,22 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+198   ,8  ,0  ,23 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+199   ,9  ,1  ,24 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+200   ,0  ,0  ,0  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+201   ,1  ,1  ,1  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+202   ,2  ,0  ,2  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+203   ,3  ,1  ,3  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+204   ,4  ,0  ,4  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+205   ,5  ,1  ,5  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+206   ,6  ,0  ,6  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+207   ,7  ,1  ,7  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+208   ,8  ,0  ,8  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+209   ,9  ,1  ,9  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+210   ,0  ,0  ,10 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+211   ,1  ,1  ,11 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+212   ,2  ,0  ,12 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+213   ,3  ,1  ,13 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+214   ,4  ,0  ,14 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+215   ,5  ,1  ,15 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+216   ,6  ,0  ,16 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+217   ,7  ,1  ,17 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+218   ,8  ,0  ,18 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+219   ,9  ,1  ,19 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+220   ,0  ,0  ,20 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+221   ,1  ,1  ,21 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+222   ,2  ,0  ,22 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+223   ,3  ,1  ,23 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+224   ,4  ,0  ,24 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+225   ,5  ,1  ,0  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+226   ,6  ,0  ,1  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+227   ,7  ,1  ,2  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+228   ,8  ,0  ,3  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+229   ,9  ,1  ,4  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+230   ,0  ,0  ,5  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+231   ,1  ,1  ,6  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+232   ,2  ,0  ,7  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+233   ,3  ,1  ,8  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+234   ,4  ,0  ,9  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+235   ,5  ,1  ,10 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+236   ,6  ,0  ,11 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+237   ,7  ,1  ,12 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+238   ,8  ,0  ,13 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+239   ,9  ,1  ,14 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+240   ,0  ,0  ,15 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+241   ,1  ,1  ,16 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+242   ,2  ,0  ,17 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+243   ,3  ,1  ,18 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+244   ,4  ,0  ,19 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+245   ,5  ,1  ,20 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+246   ,6  ,0  ,21 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+247   ,7  ,1  ,22 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+248   ,8  ,0  ,23 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+249   ,9  ,1  ,24 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+250   ,0  ,0  ,0  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+251   ,1  ,1  ,1  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+252   ,2  ,0  ,2  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+253   ,3  ,1  ,3  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+254   ,4  ,0  ,4  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+255   ,5  ,1  ,5  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+256   ,6  ,0  ,6  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+257   ,7  ,1  ,7  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+258   ,8  ,0  ,8  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+259   ,9  ,1  ,9  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+260   ,0  ,0  ,10 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+261   ,1  ,1  ,11 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+262   ,2  ,0  ,12 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+263   ,3  ,1  ,13 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+264   ,4  ,0  ,14 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+265   ,5  ,1  ,15 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+266   ,6  ,0  ,16 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+267   ,7  ,1  ,17 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+268   ,8  ,0  ,18 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+269   ,9  ,1  ,19 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+270   ,0  ,0  ,20 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+271   ,1  ,1  ,21 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+272   ,2  ,0  ,22 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+273   ,3  ,1  ,23 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+274   ,4  ,0  ,24 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+275   ,5  ,1  ,0  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+276   ,6  ,0  ,1  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+277   ,7  ,1  ,2  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+278   ,8  ,0  ,3  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+279   ,9  ,1  ,4  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+280   ,0  ,0  ,5  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+281   ,1  ,1  ,6  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+282   ,2  ,0  ,7  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+283   ,3  ,1  ,8  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+284   ,4  ,0  ,9  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+285   ,5  ,1  ,10 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+286   ,6  ,0  ,11 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+287   ,7  ,1  ,12 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+288   ,8  ,0  ,13 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+289   ,9  ,1  ,14 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+290   ,0  ,0  ,15 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+291   ,1  ,1  ,16 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+292   ,2  ,0  ,17 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+293   ,3  ,1  ,18 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+294   ,4  ,0  ,19 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+295   ,5  ,1  ,20 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+296   ,6  ,0  ,21 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+297   ,7  ,1  ,22 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+298   ,8  ,0  ,23 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+299   ,9  ,1  ,24 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+300   ,0  ,0  ,0  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+301   ,1  ,1  ,1  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+302   ,2  ,0  ,2  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+303   ,3  ,1  ,3  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+304   ,4  ,0  ,4  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+305   ,5  ,1  ,5  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+306   ,6  ,0  ,6  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+307   ,7  ,1  ,7  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+308   ,8  ,0  ,8  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+309   ,9  ,1  ,9  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+310   ,0  ,0  ,10 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+311   ,1  ,1  ,11 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+312   ,2  ,0  ,12 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+313   ,3  ,1  ,13 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+314   ,4  ,0  ,14 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+315   ,5  ,1  ,15 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+316   ,6  ,0  ,16 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+317   ,7  ,1  ,17 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+318   ,8  ,0  ,18 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+319   ,9  ,1  ,19 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+320   ,0  ,0  ,20 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+321   ,1  ,1  ,21 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+322   ,2  ,0  ,22 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+323   ,3  ,1  ,23 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+324   ,4  ,0  ,24 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+325   ,5  ,1  ,0  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+326   ,6  ,0  ,1  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+327   ,7  ,1  ,2  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+328   ,8  ,0  ,3  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+329   ,9  ,1  ,4  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+330   ,0  ,0  ,5  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+331   ,1  ,1  ,6  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+332   ,2  ,0  ,7  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+333   ,3  ,1  ,8  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+334   ,4  ,0  ,9  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+335   ,5  ,1  ,10 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+336   ,6  ,0  ,11 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+337   ,7  ,1  ,12 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+338   ,8  ,0  ,13 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+339   ,9  ,1  ,14 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+340   ,0  ,0  ,15 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+341   ,1  ,1  ,16 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+342   ,2  ,0  ,17 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+343   ,3  ,1  ,18 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+344   ,4  ,0  ,19 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+345   ,5  ,1  ,20 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+346   ,6  ,0  ,21 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+347   ,7  ,1  ,22 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+348   ,8  ,0  ,23 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+349   ,9  ,1  ,24 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+350   ,0  ,0  ,0  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+351   ,1  ,1  ,1  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+352   ,2  ,0  ,2  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+353   ,3  ,1  ,3  ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+354   ,4  ,0  ,4  ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+355   ,5  ,1  ,5  ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+356   ,6  ,0  ,6  ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+357   ,7  ,1  ,7  ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+358   ,8  ,0  ,8  ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+359   ,9  ,1  ,9  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+360   ,0  ,0  ,10 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+361   ,1  ,1  ,11 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+362   ,2  ,0  ,12 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+363   ,3  ,1  ,13 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+364   ,4  ,0  ,14 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+365   ,5  ,1  ,15 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+366   ,6  ,0  ,16 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+367   ,7  ,1  ,17 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+368   ,8  ,0  ,18 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+369   ,9  ,1  ,19 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+370   ,0  ,0  ,20 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+371   ,1  ,1  ,21 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+372   ,2  ,0  ,22 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+373   ,3  ,1  ,23 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+374   ,4  ,0  ,24 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+375   ,5  ,1  ,0  ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+376   ,6  ,0  ,1  ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+377   ,7  ,1  ,2  ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+378   ,8  ,0  ,3  ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+379   ,9  ,1  ,4  ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+380   ,0  ,0  ,5  ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+381   ,1  ,1  ,6  ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+382   ,2  ,0  ,7  ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+383   ,3  ,1  ,8  ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+384   ,4  ,0  ,9  ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+385   ,5  ,1  ,10 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+386   ,6  ,0  ,11 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+387   ,7  ,1  ,12 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+388   ,8  ,0  ,13 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+389   ,9  ,1  ,14 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+390   ,0  ,0  ,15 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+391   ,1  ,1  ,16 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+392   ,2  ,0  ,17 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+393   ,3  ,1  ,18 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+394   ,4  ,0  ,19 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+395   ,5  ,1  ,20 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+396   ,6  ,0  ,21 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+397   ,7  ,1  ,22 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+398   ,8  ,0  ,23 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR(index+399   ,9  ,1  ,24 ,15 )
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output[index] = 32000.f ;
        } 
        else 
        {
            output[index] = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
    }
}


void slowperformance_macro_lp_deri_hp_ptr_half(float* input, float* output, int samples_to_process) 
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
    int halfPtr, halfPtr1, halfPtr2, index;
    float fdatum, y0, z, y, output_temp;

    for(index = 0; index <= samples_to_process - 400 ; index+=400)
    {
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index       ,0  ,0  ,0  ,0  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+1     ,1  ,1  ,1  ,1  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+2     ,2  ,0  ,2  ,2  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+3     ,3  ,1  ,3  ,3  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+4     ,4  ,0  ,4  ,4  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+5     ,5  ,1  ,5  ,5  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+6     ,6  ,0  ,6  ,6  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+7     ,7  ,1  ,7  ,7  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+8     ,8  ,0  ,8  ,8  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+9     ,9  ,1  ,9  ,9  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+10    ,0  ,0  ,10 ,10 ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+11    ,1  ,1  ,11 ,11 ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+12    ,2  ,0  ,12 ,12 ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+13    ,3  ,1  ,13 ,13 ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+14    ,4  ,0  ,14 ,14 ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+15    ,5  ,1  ,15 ,15 ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+16    ,6  ,0  ,16 ,0  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+17    ,7  ,1  ,17 ,1  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+18    ,8  ,0  ,18 ,2  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+19    ,9  ,1  ,19 ,3  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+20    ,0  ,0  ,20 ,4  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+21    ,1  ,1  ,21 ,5  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+22    ,2  ,0  ,22 ,6  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+23    ,3  ,1  ,23 ,7  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+24    ,4  ,0  ,24 ,8  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+25    ,5  ,1  ,0  ,9  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+26    ,6  ,0  ,1  ,10 ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+27    ,7  ,1  ,2  ,11 ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+28    ,8  ,0  ,3  ,12 ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+29    ,9  ,1  ,4  ,13 ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+30    ,0  ,0  ,5  ,14 ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+31    ,1  ,1  ,6  ,15 ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+32    ,2  ,0  ,7  ,0  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+33    ,3  ,1  ,8  ,1  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+34    ,4  ,0  ,9  ,2  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+35    ,5  ,1  ,10 ,3  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+36    ,6  ,0  ,11 ,4  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+37    ,7  ,1  ,12 ,5  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+38    ,8  ,0  ,13 ,6  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+39    ,9  ,1  ,14 ,7  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+40    ,0  ,0  ,15 ,8  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+41    ,1  ,1  ,16 ,9  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+42    ,2  ,0  ,17 ,10 ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+43    ,3  ,1  ,18 ,11 ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+44    ,4  ,0  ,19 ,12 ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+45    ,5  ,1  ,20 ,13 ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+46    ,6  ,0  ,21 ,14 ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+47    ,7  ,1  ,22 ,15 ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+48    ,8  ,0  ,23 ,0  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+49    ,9  ,1  ,24 ,1  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+50    ,0  ,0  ,0  ,2  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+51    ,1  ,1  ,1  ,3  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+52    ,2  ,0  ,2  ,4  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+53    ,3  ,1  ,3  ,5  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+54    ,4  ,0  ,4  ,6  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+55    ,5  ,1  ,5  ,7  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+56    ,6  ,0  ,6  ,8  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+57    ,7  ,1  ,7  ,9  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+58    ,8  ,0  ,8  ,10 ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+59    ,9  ,1  ,9  ,11 ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+60    ,0  ,0  ,10 ,12 ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+61    ,1  ,1  ,11 ,13 ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+62    ,2  ,0  ,12 ,14 ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+63    ,3  ,1  ,13 ,15 ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+64    ,4  ,0  ,14 ,0  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+65    ,5  ,1  ,15 ,1  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+66    ,6  ,0  ,16 ,2  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+67    ,7  ,1  ,17 ,3  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+68    ,8  ,0  ,18 ,4  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+69    ,9  ,1  ,19 ,5  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+70    ,0  ,0  ,20 ,6  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+71    ,1  ,1  ,21 ,7  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+72    ,2  ,0  ,22 ,8  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+73    ,3  ,1  ,23 ,9  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+74    ,4  ,0  ,24 ,10 ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+75    ,5  ,1  ,0  ,11 ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+76    ,6  ,0  ,1  ,12 ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+77    ,7  ,1  ,2  ,13 ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+78    ,8  ,0  ,3  ,14 ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+79    ,9  ,1  ,4  ,15 ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+80    ,0  ,0  ,5  ,0  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+81    ,1  ,1  ,6  ,1  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+82    ,2  ,0  ,7  ,2  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+83    ,3  ,1  ,8  ,3  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+84    ,4  ,0  ,9  ,4  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+85    ,5  ,1  ,10 ,5  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+86    ,6  ,0  ,11 ,6  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+87    ,7  ,1  ,12 ,7  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+88    ,8  ,0  ,13 ,8  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+89    ,9  ,1  ,14 ,9  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+90    ,0  ,0  ,15 ,10 ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+91    ,1  ,1  ,16 ,11 ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+92    ,2  ,0  ,17 ,12 ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+93    ,3  ,1  ,18 ,13 ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+94    ,4  ,0  ,19 ,14 ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+95    ,5  ,1  ,20 ,15 ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+96    ,6  ,0  ,21 ,0  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+97    ,7  ,1  ,22 ,1  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+98    ,8  ,0  ,23 ,2  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+99    ,9  ,1  ,24 ,3  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+100   ,0  ,0  ,0  ,4  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+101   ,1  ,1  ,1  ,5  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+102   ,2  ,0  ,2  ,6  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+103   ,3  ,1  ,3  ,7  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+104   ,4  ,0  ,4  ,8  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+105   ,5  ,1  ,5  ,9  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+106   ,6  ,0  ,6  ,10 ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+107   ,7  ,1  ,7  ,11 ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+108   ,8  ,0  ,8  ,12 ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+109   ,9  ,1  ,9  ,13 ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+110   ,0  ,0  ,10 ,14 ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+111   ,1  ,1  ,11 ,15 ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+112   ,2  ,0  ,12 ,0  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+113   ,3  ,1  ,13 ,1  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+114   ,4  ,0  ,14 ,2  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+115   ,5  ,1  ,15 ,3  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+116   ,6  ,0  ,16 ,4  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+117   ,7  ,1  ,17 ,5  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+118   ,8  ,0  ,18 ,6  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+119   ,9  ,1  ,19 ,7  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+120   ,0  ,0  ,20 ,8  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+121   ,1  ,1  ,21 ,9  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+122   ,2  ,0  ,22 ,10 ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+123   ,3  ,1  ,23 ,11 ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+124   ,4  ,0  ,24 ,12 ,9 ,12 )      
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+125   ,5  ,1  ,0  ,13 ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+126   ,6  ,0  ,1  ,14 ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+127   ,7  ,1  ,2  ,15 ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+128   ,8  ,0  ,3  ,0  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+129   ,9  ,1  ,4  ,1  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+130   ,0  ,0  ,5  ,2  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+131   ,1  ,1  ,6  ,3  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+132   ,2  ,0  ,7  ,4  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+133   ,3  ,1  ,8  ,5  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+134   ,4  ,0  ,9  ,6  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+135   ,5  ,1  ,10 ,7  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+136   ,6  ,0  ,11 ,8  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+137   ,7  ,1  ,12 ,9  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+138   ,8  ,0  ,13 ,10 ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+139   ,9  ,1  ,14 ,11 ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+140   ,0  ,0  ,15 ,12 ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+141   ,1  ,1  ,16 ,13 ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+142   ,2  ,0  ,17 ,14 ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+143   ,3  ,1  ,18 ,15 ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+144   ,4  ,0  ,19 ,0  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+145   ,5  ,1  ,20 ,1  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+146   ,6  ,0  ,21 ,2  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+147   ,7  ,1  ,22 ,3  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+148   ,8  ,0  ,23 ,4  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+149   ,9  ,1  ,24 ,5  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+150   ,0  ,0  ,0  ,6  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+151   ,1  ,1  ,1  ,7  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+152   ,2  ,0  ,2  ,8  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+153   ,3  ,1  ,3  ,9  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+154   ,4  ,0  ,4  ,10 ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+155   ,5  ,1  ,5  ,11 ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+156   ,6  ,0  ,6  ,12 ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+157   ,7  ,1  ,7  ,13 ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+158   ,8  ,0  ,8  ,14 ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+159   ,9  ,1  ,9  ,15 ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+160   ,0  ,0  ,10 ,0  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+161   ,1  ,1  ,11 ,1  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+162   ,2  ,0  ,12 ,2  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+163   ,3  ,1  ,13 ,3  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+164   ,4  ,0  ,14 ,4  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+165   ,5  ,1  ,15 ,5  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+166   ,6  ,0  ,16 ,6  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+167   ,7  ,1  ,17 ,7  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+168   ,8  ,0  ,18 ,8  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+169   ,9  ,1  ,19 ,9  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+170   ,0  ,0  ,20 ,10 ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+171   ,1  ,1  ,21 ,11 ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+172   ,2  ,0  ,22 ,12 ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+173   ,3  ,1  ,23 ,13 ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+174   ,4  ,0  ,24 ,14 ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+175   ,5  ,1  ,0  ,15 ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+176   ,6  ,0  ,1  ,0  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+177   ,7  ,1  ,2  ,1  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+178   ,8  ,0  ,3  ,2  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+179   ,9  ,1  ,4  ,3  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+180   ,0  ,0  ,5  ,4  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+181   ,1  ,1  ,6  ,5  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+182   ,2  ,0  ,7  ,6  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+183   ,3  ,1  ,8  ,7  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+184   ,4  ,0  ,9  ,8  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+185   ,5  ,1  ,10 ,9  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+186   ,6  ,0  ,11 ,10 ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+187   ,7  ,1  ,12 ,11 ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+188   ,8  ,0  ,13 ,12 ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+189   ,9  ,1  ,14 ,13 ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+190   ,0  ,0  ,15 ,14 ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+191   ,1  ,1  ,16 ,15 ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+192   ,2  ,0  ,17 ,0  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+193   ,3  ,1  ,18 ,1  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+194   ,4  ,0  ,19 ,2  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+195   ,5  ,1  ,20 ,3  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+196   ,6  ,0  ,21 ,4  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+197   ,7  ,1  ,22 ,5  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+198   ,8  ,0  ,23 ,6  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+199   ,9  ,1  ,24 ,7  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+200   ,0  ,0  ,0  ,8  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+201   ,1  ,1  ,1  ,9  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+202   ,2  ,0  ,2  ,10 ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+203   ,3  ,1  ,3  ,11 ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+204   ,4  ,0  ,4  ,12 ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+205   ,5  ,1  ,5  ,13 ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+206   ,6  ,0  ,6  ,14 ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+207   ,7  ,1  ,7  ,15 ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+208   ,8  ,0  ,8  ,0  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+209   ,9  ,1  ,9  ,1  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+210   ,0  ,0  ,10 ,2  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+211   ,1  ,1  ,11 ,3  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+212   ,2  ,0  ,12 ,4  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+213   ,3  ,1  ,13 ,5  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+214   ,4  ,0  ,14 ,6  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+215   ,5  ,1  ,15 ,7  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+216   ,6  ,0  ,16 ,8  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+217   ,7  ,1  ,17 ,9  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+218   ,8  ,0  ,18 ,10 ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+219   ,9  ,1  ,19 ,11 ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+220   ,0  ,0  ,20 ,12 ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+221   ,1  ,1  ,21 ,13 ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+222   ,2  ,0  ,22 ,14 ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+223   ,3  ,1  ,23 ,15 ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+224   ,4  ,0  ,24 ,0  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+225   ,5  ,1  ,0  ,1  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+226   ,6  ,0  ,1  ,2  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+227   ,7  ,1  ,2  ,3  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+228   ,8  ,0  ,3  ,4  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+229   ,9  ,1  ,4  ,5  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+230   ,0  ,0  ,5  ,6  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+231   ,1  ,1  ,6  ,7  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+232   ,2  ,0  ,7  ,8  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+233   ,3  ,1  ,8  ,9  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+234   ,4  ,0  ,9  ,10 ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+235   ,5  ,1  ,10 ,11 ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+236   ,6  ,0  ,11 ,12 ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+237   ,7  ,1  ,12 ,13 ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+238   ,8  ,0  ,13 ,14 ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+239   ,9  ,1  ,14 ,15 ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+240   ,0  ,0  ,15 ,0  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+241   ,1  ,1  ,16 ,1  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+242   ,2  ,0  ,17 ,2  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+243   ,3  ,1  ,18 ,3  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+244   ,4  ,0  ,19 ,4  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+245   ,5  ,1  ,20 ,5  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+246   ,6  ,0  ,21 ,6  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+247   ,7  ,1  ,22 ,7  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+248   ,8  ,0  ,23 ,8  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+249   ,9  ,1  ,24 ,9  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+250   ,0  ,0  ,0  ,10 ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+251   ,1  ,1  ,1  ,11 ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+252   ,2  ,0  ,2  ,12 ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+253   ,3  ,1  ,3  ,13 ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+254   ,4  ,0  ,4  ,14 ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+255   ,5  ,1  ,5  ,15 ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+256   ,6  ,0  ,6  ,0  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+257   ,7  ,1  ,7  ,1  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+258   ,8  ,0  ,8  ,2  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+259   ,9  ,1  ,9  ,3  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+260   ,0  ,0  ,10 ,4  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+261   ,1  ,1  ,11 ,5  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+262   ,2  ,0  ,12 ,6  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+263   ,3  ,1  ,13 ,7  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+264   ,4  ,0  ,14 ,8  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+265   ,5  ,1  ,15 ,9  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+266   ,6  ,0  ,16 ,10 ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+267   ,7  ,1  ,17 ,11 ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+268   ,8  ,0  ,18 ,12 ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+269   ,9  ,1  ,19 ,13 ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+270   ,0  ,0  ,20 ,14 ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+271   ,1  ,1  ,21 ,15 ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+272   ,2  ,0  ,22 ,0  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+273   ,3  ,1  ,23 ,1  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+274   ,4  ,0  ,24 ,2  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+275   ,5  ,1  ,0  ,3  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+276   ,6  ,0  ,1  ,4  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+277   ,7  ,1  ,2  ,5  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+278   ,8  ,0  ,3  ,6  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+279   ,9  ,1  ,4  ,7  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+280   ,0  ,0  ,5  ,8  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+281   ,1  ,1  ,6  ,9  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+282   ,2  ,0  ,7  ,10 ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+283   ,3  ,1  ,8  ,11 ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+284   ,4  ,0  ,9  ,12 ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+285   ,5  ,1  ,10 ,13 ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+286   ,6  ,0  ,11 ,14 ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+287   ,7  ,1  ,12 ,15 ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+288   ,8  ,0  ,13 ,0  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+289   ,9  ,1  ,14 ,1  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+290   ,0  ,0  ,15 ,2  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+291   ,1  ,1  ,16 ,3  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+292   ,2  ,0  ,17 ,4  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+293   ,3  ,1  ,18 ,5  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+294   ,4  ,0  ,19 ,6  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+295   ,5  ,1  ,20 ,7  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+296   ,6  ,0  ,21 ,8  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+297   ,7  ,1  ,22 ,9  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+298   ,8  ,0  ,23 ,10 ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+299   ,9  ,1  ,24 ,11 ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+300   ,0  ,0  ,0  ,12 ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+301   ,1  ,1  ,1  ,13 ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+302   ,2  ,0  ,2  ,14 ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+303   ,3  ,1  ,3  ,15 ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+304   ,4  ,0  ,4  ,0  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+305   ,5  ,1  ,5  ,1  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+306   ,6  ,0  ,6  ,2  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+307   ,7  ,1  ,7  ,3  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+308   ,8  ,0  ,8  ,4  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+309   ,9  ,1  ,9  ,5  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+310   ,0  ,0  ,10 ,6  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+311   ,1  ,1  ,11 ,7  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+312   ,2  ,0  ,12 ,8  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+313   ,3  ,1  ,13 ,9  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+314   ,4  ,0  ,14 ,10 ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+315   ,5  ,1  ,15 ,11 ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+316   ,6  ,0  ,16 ,12 ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+317   ,7  ,1  ,17 ,13 ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+318   ,8  ,0  ,18 ,14 ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+319   ,9  ,1  ,19 ,15 ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+320   ,0  ,0  ,20 ,0  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+321   ,1  ,1  ,21 ,1  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+322   ,2  ,0  ,22 ,2  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+323   ,3  ,1  ,23 ,3  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+324   ,4  ,0  ,24 ,4  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+325   ,5  ,1  ,0  ,5  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+326   ,6  ,0  ,1  ,6  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+327   ,7  ,1  ,2  ,7  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+328   ,8  ,0  ,3  ,8  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+329   ,9  ,1  ,4  ,9  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+330   ,0  ,0  ,5  ,10 ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+331   ,1  ,1  ,6  ,11 ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+332   ,2  ,0  ,7  ,12 ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+333   ,3  ,1  ,8  ,13 ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+334   ,4  ,0  ,9  ,14 ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+335   ,5  ,1  ,10 ,15 ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+336   ,6  ,0  ,11 ,0  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+337   ,7  ,1  ,12 ,1  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+338   ,8  ,0  ,13 ,2  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+339   ,9  ,1  ,14 ,3  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+340   ,0  ,0  ,15 ,4  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+341   ,1  ,1  ,16 ,5  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+342   ,2  ,0  ,17 ,6  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+343   ,3  ,1  ,18 ,7  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+344   ,4  ,0  ,19 ,8  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+345   ,5  ,1  ,20 ,9  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+346   ,6  ,0  ,21 ,10 ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+347   ,7  ,1  ,22 ,11 ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+348   ,8  ,0  ,23 ,12 ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+349   ,9  ,1  ,24 ,13 ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+350   ,0  ,0  ,0  ,14 ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+351   ,1  ,1  ,1  ,15 ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+352   ,2  ,0  ,2  ,0  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+353   ,3  ,1  ,3  ,1  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+354   ,4  ,0  ,4  ,2  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+355   ,5  ,1  ,5  ,3  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+356   ,6  ,0  ,6  ,4  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+357   ,7  ,1  ,7  ,5  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+358   ,8  ,0  ,8  ,6  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+359   ,9  ,1  ,9  ,7  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+360   ,0  ,0  ,10 ,8  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+361   ,1  ,1  ,11 ,9  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+362   ,2  ,0  ,12 ,10 ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+363   ,3  ,1  ,13 ,11 ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+364   ,4  ,0  ,14 ,12 ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+365   ,5  ,1  ,15 ,13 ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+366   ,6  ,0  ,16 ,14 ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+367   ,7  ,1  ,17 ,15 ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+368   ,8  ,0  ,18 ,0  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+369   ,9  ,1  ,19 ,1  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+370   ,0  ,0  ,20 ,2  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+371   ,1  ,1  ,21 ,3  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+372   ,2  ,0  ,22 ,4  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+373   ,3  ,1  ,23 ,5  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+374   ,4  ,0  ,24 ,6  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+375   ,5  ,1  ,0  ,7  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+376   ,6  ,0  ,1  ,8  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+377   ,7  ,1  ,2  ,9  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+378   ,8  ,0  ,3  ,10 ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+379   ,9  ,1  ,4  ,11 ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+380   ,0  ,0  ,5  ,12 ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+381   ,1  ,1  ,6  ,13 ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+382   ,2  ,0  ,7  ,14 ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+383   ,3  ,1  ,8  ,15 ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+384   ,4  ,0  ,9  ,0  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+385   ,5  ,1  ,10 ,1  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+386   ,6  ,0  ,11 ,2  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+387   ,7  ,1  ,12 ,3  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+388   ,8  ,0  ,13 ,4  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+389   ,9  ,1  ,14 ,5  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+390   ,0  ,0  ,15 ,6  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+391   ,1  ,1  ,16 ,7  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+392   ,2  ,0  ,17 ,8  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+393   ,3  ,1  ,18 ,9  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+394   ,4  ,0  ,19 ,10 ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+395   ,5  ,1  ,20 ,11 ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+396   ,6  ,0  ,21 ,12 ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+397   ,7  ,1  ,22 ,13 ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+398   ,8  ,0  ,23 ,14 ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF(index+399   ,9  ,1  ,24 ,15 ,4 ,12 )
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output[index] = 32000.f ;
        } 
        else 
        {
            output[index] = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
    }
}

void slowperformance_macro_lp_deri_hp_ptr_half_div(float* input, float* output, int samples_to_process) 
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
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0, sum_temp = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, halfPtr1, halfPtr2, index;
    float fdatum, y0, z, y, output_temp;
    static float LPBUFFER_LGTH_INV = 1/((float)LPBUFFER_LGTH);
    static float HPBUFFER_LGTH_INV = 1/((float)HPBUFFER_LGTH);
    static float WINDOW_WIDTH_INV = 1/((float)WINDOW_WIDTH);

    for(index = 0; index <= samples_to_process - 400 ; index+=400)
    {
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index       ,0  ,0  ,0  ,0  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+1     ,1  ,1  ,1  ,1  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+2     ,2  ,0  ,2  ,2  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+3     ,3  ,1  ,3  ,3  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+4     ,4  ,0  ,4  ,4  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+5     ,5  ,1  ,5  ,5  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+6     ,6  ,0  ,6  ,6  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+7     ,7  ,1  ,7  ,7  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+8     ,8  ,0  ,8  ,8  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+9     ,9  ,1  ,9  ,9  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+10    ,0  ,0  ,10 ,10 ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+11    ,1  ,1  ,11 ,11 ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+12    ,2  ,0  ,12 ,12 ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+13    ,3  ,1  ,13 ,13 ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+14    ,4  ,0  ,14 ,14 ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+15    ,5  ,1  ,15 ,15 ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+16    ,6  ,0  ,16 ,0  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+17    ,7  ,1  ,17 ,1  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+18    ,8  ,0  ,18 ,2  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+19    ,9  ,1  ,19 ,3  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+20    ,0  ,0  ,20 ,4  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+21    ,1  ,1  ,21 ,5  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+22    ,2  ,0  ,22 ,6  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+23    ,3  ,1  ,23 ,7  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+24    ,4  ,0  ,24 ,8  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+25    ,5  ,1  ,0  ,9  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+26    ,6  ,0  ,1  ,10 ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+27    ,7  ,1  ,2  ,11 ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+28    ,8  ,0  ,3  ,12 ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+29    ,9  ,1  ,4  ,13 ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+30    ,0  ,0  ,5  ,14 ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+31    ,1  ,1  ,6  ,15 ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+32    ,2  ,0  ,7  ,0  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+33    ,3  ,1  ,8  ,1  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+34    ,4  ,0  ,9  ,2  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+35    ,5  ,1  ,10 ,3  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+36    ,6  ,0  ,11 ,4  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+37    ,7  ,1  ,12 ,5  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+38    ,8  ,0  ,13 ,6  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+39    ,9  ,1  ,14 ,7  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+40    ,0  ,0  ,15 ,8  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+41    ,1  ,1  ,16 ,9  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+42    ,2  ,0  ,17 ,10 ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+43    ,3  ,1  ,18 ,11 ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+44    ,4  ,0  ,19 ,12 ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+45    ,5  ,1  ,20 ,13 ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+46    ,6  ,0  ,21 ,14 ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+47    ,7  ,1  ,22 ,15 ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+48    ,8  ,0  ,23 ,0  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+49    ,9  ,1  ,24 ,1  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+50    ,0  ,0  ,0  ,2  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+51    ,1  ,1  ,1  ,3  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+52    ,2  ,0  ,2  ,4  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+53    ,3  ,1  ,3  ,5  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+54    ,4  ,0  ,4  ,6  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+55    ,5  ,1  ,5  ,7  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+56    ,6  ,0  ,6  ,8  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+57    ,7  ,1  ,7  ,9  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+58    ,8  ,0  ,8  ,10 ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+59    ,9  ,1  ,9  ,11 ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+60    ,0  ,0  ,10 ,12 ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+61    ,1  ,1  ,11 ,13 ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+62    ,2  ,0  ,12 ,14 ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+63    ,3  ,1  ,13 ,15 ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+64    ,4  ,0  ,14 ,0  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+65    ,5  ,1  ,15 ,1  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+66    ,6  ,0  ,16 ,2  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+67    ,7  ,1  ,17 ,3  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+68    ,8  ,0  ,18 ,4  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+69    ,9  ,1  ,19 ,5  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+70    ,0  ,0  ,20 ,6  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+71    ,1  ,1  ,21 ,7  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+72    ,2  ,0  ,22 ,8  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+73    ,3  ,1  ,23 ,9  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+74    ,4  ,0  ,24 ,10 ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+75    ,5  ,1  ,0  ,11 ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+76    ,6  ,0  ,1  ,12 ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+77    ,7  ,1  ,2  ,13 ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+78    ,8  ,0  ,3  ,14 ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+79    ,9  ,1  ,4  ,15 ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+80    ,0  ,0  ,5  ,0  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+81    ,1  ,1  ,6  ,1  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+82    ,2  ,0  ,7  ,2  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+83    ,3  ,1  ,8  ,3  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+84    ,4  ,0  ,9  ,4  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+85    ,5  ,1  ,10 ,5  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+86    ,6  ,0  ,11 ,6  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+87    ,7  ,1  ,12 ,7  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+88    ,8  ,0  ,13 ,8  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+89    ,9  ,1  ,14 ,9  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+90    ,0  ,0  ,15 ,10 ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+91    ,1  ,1  ,16 ,11 ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+92    ,2  ,0  ,17 ,12 ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+93    ,3  ,1  ,18 ,13 ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+94    ,4  ,0  ,19 ,14 ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+95    ,5  ,1  ,20 ,15 ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+96    ,6  ,0  ,21 ,0  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+97    ,7  ,1  ,22 ,1  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+98    ,8  ,0  ,23 ,2  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+99    ,9  ,1  ,24 ,3  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+100   ,0  ,0  ,0  ,4  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+101   ,1  ,1  ,1  ,5  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+102   ,2  ,0  ,2  ,6  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+103   ,3  ,1  ,3  ,7  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+104   ,4  ,0  ,4  ,8  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+105   ,5  ,1  ,5  ,9  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+106   ,6  ,0  ,6  ,10 ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+107   ,7  ,1  ,7  ,11 ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+108   ,8  ,0  ,8  ,12 ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+109   ,9  ,1  ,9  ,13 ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+110   ,0  ,0  ,10 ,14 ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+111   ,1  ,1  ,11 ,15 ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+112   ,2  ,0  ,12 ,0  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+113   ,3  ,1  ,13 ,1  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+114   ,4  ,0  ,14 ,2  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+115   ,5  ,1  ,15 ,3  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+116   ,6  ,0  ,16 ,4  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+117   ,7  ,1  ,17 ,5  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+118   ,8  ,0  ,18 ,6  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+119   ,9  ,1  ,19 ,7  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+120   ,0  ,0  ,20 ,8  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+121   ,1  ,1  ,21 ,9  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+122   ,2  ,0  ,22 ,10 ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+123   ,3  ,1  ,23 ,11 ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+124   ,4  ,0  ,24 ,12 ,9 ,12 )      
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+125   ,5  ,1  ,0  ,13 ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+126   ,6  ,0  ,1  ,14 ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+127   ,7  ,1  ,2  ,15 ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+128   ,8  ,0  ,3  ,0  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+129   ,9  ,1  ,4  ,1  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+130   ,0  ,0  ,5  ,2  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+131   ,1  ,1  ,6  ,3  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+132   ,2  ,0  ,7  ,4  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+133   ,3  ,1  ,8  ,5  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+134   ,4  ,0  ,9  ,6  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+135   ,5  ,1  ,10 ,7  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+136   ,6  ,0  ,11 ,8  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+137   ,7  ,1  ,12 ,9  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+138   ,8  ,0  ,13 ,10 ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+139   ,9  ,1  ,14 ,11 ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+140   ,0  ,0  ,15 ,12 ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+141   ,1  ,1  ,16 ,13 ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+142   ,2  ,0  ,17 ,14 ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+143   ,3  ,1  ,18 ,15 ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+144   ,4  ,0  ,19 ,0  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+145   ,5  ,1  ,20 ,1  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+146   ,6  ,0  ,21 ,2  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+147   ,7  ,1  ,22 ,3  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+148   ,8  ,0  ,23 ,4  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+149   ,9  ,1  ,24 ,5  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+150   ,0  ,0  ,0  ,6  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+151   ,1  ,1  ,1  ,7  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+152   ,2  ,0  ,2  ,8  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+153   ,3  ,1  ,3  ,9  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+154   ,4  ,0  ,4  ,10 ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+155   ,5  ,1  ,5  ,11 ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+156   ,6  ,0  ,6  ,12 ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+157   ,7  ,1  ,7  ,13 ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+158   ,8  ,0  ,8  ,14 ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+159   ,9  ,1  ,9  ,15 ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+160   ,0  ,0  ,10 ,0  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+161   ,1  ,1  ,11 ,1  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+162   ,2  ,0  ,12 ,2  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+163   ,3  ,1  ,13 ,3  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+164   ,4  ,0  ,14 ,4  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+165   ,5  ,1  ,15 ,5  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+166   ,6  ,0  ,16 ,6  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+167   ,7  ,1  ,17 ,7  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+168   ,8  ,0  ,18 ,8  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+169   ,9  ,1  ,19 ,9  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+170   ,0  ,0  ,20 ,10 ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+171   ,1  ,1  ,21 ,11 ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+172   ,2  ,0  ,22 ,12 ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+173   ,3  ,1  ,23 ,13 ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+174   ,4  ,0  ,24 ,14 ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+175   ,5  ,1  ,0  ,15 ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+176   ,6  ,0  ,1  ,0  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+177   ,7  ,1  ,2  ,1  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+178   ,8  ,0  ,3  ,2  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+179   ,9  ,1  ,4  ,3  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+180   ,0  ,0  ,5  ,4  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+181   ,1  ,1  ,6  ,5  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+182   ,2  ,0  ,7  ,6  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+183   ,3  ,1  ,8  ,7  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+184   ,4  ,0  ,9  ,8  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+185   ,5  ,1  ,10 ,9  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+186   ,6  ,0  ,11 ,10 ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+187   ,7  ,1  ,12 ,11 ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+188   ,8  ,0  ,13 ,12 ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+189   ,9  ,1  ,14 ,13 ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+190   ,0  ,0  ,15 ,14 ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+191   ,1  ,1  ,16 ,15 ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+192   ,2  ,0  ,17 ,0  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+193   ,3  ,1  ,18 ,1  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+194   ,4  ,0  ,19 ,2  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+195   ,5  ,1  ,20 ,3  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+196   ,6  ,0  ,21 ,4  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+197   ,7  ,1  ,22 ,5  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+198   ,8  ,0  ,23 ,6  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+199   ,9  ,1  ,24 ,7  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+200   ,0  ,0  ,0  ,8  ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+201   ,1  ,1  ,1  ,9  ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+202   ,2  ,0  ,2  ,10 ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+203   ,3  ,1  ,3  ,11 ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+204   ,4  ,0  ,4  ,12 ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+205   ,5  ,1  ,5  ,13 ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+206   ,6  ,0  ,6  ,14 ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+207   ,7  ,1  ,7  ,15 ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+208   ,8  ,0  ,8  ,0  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+209   ,9  ,1  ,9  ,1  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+210   ,0  ,0  ,10 ,2  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+211   ,1  ,1  ,11 ,3  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+212   ,2  ,0  ,12 ,4  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+213   ,3  ,1  ,13 ,5  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+214   ,4  ,0  ,14 ,6  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+215   ,5  ,1  ,15 ,7  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+216   ,6  ,0  ,16 ,8  ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+217   ,7  ,1  ,17 ,9  ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+218   ,8  ,0  ,18 ,10 ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+219   ,9  ,1  ,19 ,11 ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+220   ,0  ,0  ,20 ,12 ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+221   ,1  ,1  ,21 ,13 ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+222   ,2  ,0  ,22 ,14 ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+223   ,3  ,1  ,23 ,15 ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+224   ,4  ,0  ,24 ,0  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+225   ,5  ,1  ,0  ,1  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+226   ,6  ,0  ,1  ,2  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+227   ,7  ,1  ,2  ,3  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+228   ,8  ,0  ,3  ,4  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+229   ,9  ,1  ,4  ,5  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+230   ,0  ,0  ,5  ,6  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+231   ,1  ,1  ,6  ,7  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+232   ,2  ,0  ,7  ,8  ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+233   ,3  ,1  ,8  ,9  ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+234   ,4  ,0  ,9  ,10 ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+235   ,5  ,1  ,10 ,11 ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+236   ,6  ,0  ,11 ,12 ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+237   ,7  ,1  ,12 ,13 ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+238   ,8  ,0  ,13 ,14 ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+239   ,9  ,1  ,14 ,15 ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+240   ,0  ,0  ,15 ,0  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+241   ,1  ,1  ,16 ,1  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+242   ,2  ,0  ,17 ,2  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+243   ,3  ,1  ,18 ,3  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+244   ,4  ,0  ,19 ,4  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+245   ,5  ,1  ,20 ,5  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+246   ,6  ,0  ,21 ,6  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+247   ,7  ,1  ,22 ,7  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+248   ,8  ,0  ,23 ,8  ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+249   ,9  ,1  ,24 ,9  ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+250   ,0  ,0  ,0  ,10 ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+251   ,1  ,1  ,1  ,11 ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+252   ,2  ,0  ,2  ,12 ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+253   ,3  ,1  ,3  ,13 ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+254   ,4  ,0  ,4  ,14 ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+255   ,5  ,1  ,5  ,15 ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+256   ,6  ,0  ,6  ,0  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+257   ,7  ,1  ,7  ,1  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+258   ,8  ,0  ,8  ,2  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+259   ,9  ,1  ,9  ,3  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+260   ,0  ,0  ,10 ,4  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+261   ,1  ,1  ,11 ,5  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+262   ,2  ,0  ,12 ,6  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+263   ,3  ,1  ,13 ,7  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+264   ,4  ,0  ,14 ,8  ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+265   ,5  ,1  ,15 ,9  ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+266   ,6  ,0  ,16 ,10 ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+267   ,7  ,1  ,17 ,11 ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+268   ,8  ,0  ,18 ,12 ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+269   ,9  ,1  ,19 ,13 ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+270   ,0  ,0  ,20 ,14 ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+271   ,1  ,1  ,21 ,15 ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+272   ,2  ,0  ,22 ,0  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+273   ,3  ,1  ,23 ,1  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+274   ,4  ,0  ,24 ,2  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+275   ,5  ,1  ,0  ,3  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+276   ,6  ,0  ,1  ,4  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+277   ,7  ,1  ,2  ,5  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+278   ,8  ,0  ,3  ,6  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+279   ,9  ,1  ,4  ,7  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+280   ,0  ,0  ,5  ,8  ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+281   ,1  ,1  ,6  ,9  ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+282   ,2  ,0  ,7  ,10 ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+283   ,3  ,1  ,8  ,11 ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+284   ,4  ,0  ,9  ,12 ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+285   ,5  ,1  ,10 ,13 ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+286   ,6  ,0  ,11 ,14 ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+287   ,7  ,1  ,12 ,15 ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+288   ,8  ,0  ,13 ,0  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+289   ,9  ,1  ,14 ,1  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+290   ,0  ,0  ,15 ,2  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+291   ,1  ,1  ,16 ,3  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+292   ,2  ,0  ,17 ,4  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+293   ,3  ,1  ,18 ,5  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+294   ,4  ,0  ,19 ,6  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+295   ,5  ,1  ,20 ,7  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+296   ,6  ,0  ,21 ,8  ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+297   ,7  ,1  ,22 ,9  ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+298   ,8  ,0  ,23 ,10 ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+299   ,9  ,1  ,24 ,11 ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+300   ,0  ,0  ,0  ,12 ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+301   ,1  ,1  ,1  ,13 ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+302   ,2  ,0  ,2  ,14 ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+303   ,3  ,1  ,3  ,15 ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+304   ,4  ,0  ,4  ,0  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+305   ,5  ,1  ,5  ,1  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+306   ,6  ,0  ,6  ,2  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+307   ,7  ,1  ,7  ,3  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+308   ,8  ,0  ,8  ,4  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+309   ,9  ,1  ,9  ,5  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+310   ,0  ,0  ,10 ,6  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+311   ,1  ,1  ,11 ,7  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+312   ,2  ,0  ,12 ,8  ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+313   ,3  ,1  ,13 ,9  ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+314   ,4  ,0  ,14 ,10 ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+315   ,5  ,1  ,15 ,11 ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+316   ,6  ,0  ,16 ,12 ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+317   ,7  ,1  ,17 ,13 ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+318   ,8  ,0  ,18 ,14 ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+319   ,9  ,1  ,19 ,15 ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+320   ,0  ,0  ,20 ,0  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+321   ,1  ,1  ,21 ,1  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+322   ,2  ,0  ,22 ,2  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+323   ,3  ,1  ,23 ,3  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+324   ,4  ,0  ,24 ,4  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+325   ,5  ,1  ,0  ,5  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+326   ,6  ,0  ,1  ,6  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+327   ,7  ,1  ,2  ,7  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+328   ,8  ,0  ,3  ,8  ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+329   ,9  ,1  ,4  ,9  ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+330   ,0  ,0  ,5  ,10 ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+331   ,1  ,1  ,6  ,11 ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+332   ,2  ,0  ,7  ,12 ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+333   ,3  ,1  ,8  ,13 ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+334   ,4  ,0  ,9  ,14 ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+335   ,5  ,1  ,10 ,15 ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+336   ,6  ,0  ,11 ,0  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+337   ,7  ,1  ,12 ,1  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+338   ,8  ,0  ,13 ,2  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+339   ,9  ,1  ,14 ,3  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+340   ,0  ,0  ,15 ,4  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+341   ,1  ,1  ,16 ,5  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+342   ,2  ,0  ,17 ,6  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+343   ,3  ,1  ,18 ,7  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+344   ,4  ,0  ,19 ,8  ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+345   ,5  ,1  ,20 ,9  ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+346   ,6  ,0  ,21 ,10 ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+347   ,7  ,1  ,22 ,11 ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+348   ,8  ,0  ,23 ,12 ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+349   ,9  ,1  ,24 ,13 ,4 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+350   ,0  ,0  ,0  ,14 ,5 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+351   ,1  ,1  ,1  ,15 ,6 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+352   ,2  ,0  ,2  ,0  ,7 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+353   ,3  ,1  ,3  ,1  ,8 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+354   ,4  ,0  ,4  ,2  ,9 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+355   ,5  ,1  ,5  ,3  ,0 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+356   ,6  ,0  ,6  ,4  ,1 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+357   ,7  ,1  ,7  ,5  ,2 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+358   ,8  ,0  ,8  ,6  ,3 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+359   ,9  ,1  ,9  ,7  ,4 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+360   ,0  ,0  ,10 ,8  ,5 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+361   ,1  ,1  ,11 ,9  ,6 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+362   ,2  ,0  ,12 ,10 ,7 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+363   ,3  ,1  ,13 ,11 ,8 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+364   ,4  ,0  ,14 ,12 ,9 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+365   ,5  ,1  ,15 ,13 ,0 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+366   ,6  ,0  ,16 ,14 ,1 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+367   ,7  ,1  ,17 ,15 ,2 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+368   ,8  ,0  ,18 ,0  ,3 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+369   ,9  ,1  ,19 ,1  ,4 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+370   ,0  ,0  ,20 ,2  ,5 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+371   ,1  ,1  ,21 ,3  ,6 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+372   ,2  ,0  ,22 ,4  ,7 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+373   ,3  ,1  ,23 ,5  ,8 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+374   ,4  ,0  ,24 ,6  ,9 ,12 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+375   ,5  ,1  ,0  ,7  ,0 ,13 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+376   ,6  ,0  ,1  ,8  ,1 ,14 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+377   ,7  ,1  ,2  ,9  ,2 ,15 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+378   ,8  ,0  ,3  ,10 ,3 ,16 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+379   ,9  ,1  ,4  ,11 ,4 ,17 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+380   ,0  ,0  ,5  ,12 ,5 ,18 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+381   ,1  ,1  ,6  ,13 ,6 ,19 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+382   ,2  ,0  ,7  ,14 ,7 ,20 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+383   ,3  ,1  ,8  ,15 ,8 ,21 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+384   ,4  ,0  ,9  ,0  ,9 ,22 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+385   ,5  ,1  ,10 ,1  ,0 ,23 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+386   ,6  ,0  ,11 ,2  ,1 ,24 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+387   ,7  ,1  ,12 ,3  ,2 ,0  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+388   ,8  ,0  ,13 ,4  ,3 ,1  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+389   ,9  ,1  ,14 ,5  ,4 ,2  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+390   ,0  ,0  ,15 ,6  ,5 ,3  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+391   ,1  ,1  ,16 ,7  ,6 ,4  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+392   ,2  ,0  ,17 ,8  ,7 ,5  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+393   ,3  ,1  ,18 ,9  ,8 ,6  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+394   ,4  ,0  ,19 ,10 ,9 ,7  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+395   ,5  ,1  ,20 ,11 ,0 ,8  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+396   ,6  ,0  ,21 ,12 ,1 ,9  )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+397   ,7  ,1  ,22 ,13 ,2 ,10 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+398   ,8  ,0  ,23 ,14 ,3 ,11 )
        UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+399   ,9  ,1  ,24 ,15 ,4 ,12 )
    }
    for(; index < samples_to_process; index++)
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

        if((sum / (float)WINDOW_WIDTH) > 32000.f)
        {
            output[index] = 32000.f ;
        } 
        else 
        {
            output[index] = sum / (float)WINDOW_WIDTH ;
        }

        if(++lp_ptr == LPBUFFER_LGTH)
            lp_ptr = 0 ;
        if(++derI == DERIV_LENGTH)
            derI = 0 ;
        if(++hp_ptr == HPBUFFER_LGTH)
            hp_ptr = 0 ;
        if(++ptr == WINDOW_WIDTH)
            ptr = 0 ;
    }
}


void slowperformance2(float* datum, float* filtOutput, int sampleLength) 
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
    static float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
    static float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
    static float window_width_inv = 1/ (float)WINDOW_WIDTH;
    static int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    static int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

    #ifdef OPERATION_COUNTER
    float_div_counter+=3;
    float_mul_counter+=2;
    #endif

    __m128i maxMask = _mm_set_epi32(WINDOW_WIDTH, DERIV_LENGTH, HPBUFFER_LGTH, LPBUFFER_LGTH);
    __m128i oneVec = _mm_set_epi32(1, 1, 1, 1);

    int i;
    for(i=0; i < sampleLength - BLOCKING_SIZE+1; i+= BLOCKING_SIZE)
    {
        for(int j=0; j < BLOCKING_SIZE; j++)
        {
            index = i + j;
            halfPtr = ptr_array[0] - lpbuffer_lgth_half ;    // Use halfPtr to index
            if(halfPtr < 0)                         // to x[n-6].
                halfPtr += LPBUFFER_LGTH ;

            y0 = (y1*2.0f) - y2 + datum[index] - (lp_data[halfPtr]*2.0f) + lp_data[ptr_array[0]] ;
            y2 = y1;
            y1 = y0;
            fdatum = y0 * lpbuffer_sqr_div_4;
            lp_data[ptr_array[0]] = datum[index] ;            // Stick most recent sample into
            
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
            filtOutput[index] = output_temp;
        }
    }

    for(i; i<sampleLength; i++)
    {
        halfPtr = ptr_array[0] - lpbuffer_lgth_half ;    // Use halfPtr to index
        if(halfPtr < 0)                         // to x[n-6].
            halfPtr += LPBUFFER_LGTH ;

        y0 = (y1*2.0f) - y2 + datum[i] - (lp_data[halfPtr]*2.0f) + lp_data[ptr_array[0]] ;
        y2 = y1;
        y1 = y0;
        fdatum = y0 * lpbuffer_sqr_div_4;
        lp_data[ptr_array[0]] = datum[i] ;            // Stick most recent sample into
        
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
        filtOutput[i] = output_temp;
    }
}

// void blocking(float* input, float* output, int samples_to_process) 
// {
//     // data buffer for lpfilt
//     static float lp_data[LPBUFFER_LGTH];
//     // data buffer for hpfilt
//     static float hp_data[HPBUFFER_LGTH];
//     // data buffer for derivative
//     static float derBuff[DERIV_LENGTH] ;
//     // data buffer for moving window average
//     static float data[WINDOW_WIDTH];
        
//     // ------- initialize filters ------- //
//     //lpfilt
//     for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
//         lp_data[i_init] = 0.f;
//     //hpfilt
//     for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
//         hp_data[i_init] = 0.f;
//     //derivative
//     for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
//         derBuff[i_init] = 0 ;
//     //movint window integration
//     for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
//         data[i_init] = 0 ;
        
//     static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
//     static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
//     int halfPtr, index;
//     float fdatum, y0, z, y, output_temp;

//     for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
//     {
//         for(int j=0; j < BLOCKING_SIZE; j++)
//         {
//             index = i * BLOCKING_SIZE + j;
//             halfPtr = lp_ptr-(LPBUFFER_LGTH/2) ;    // Use halfPtr to index
//             if(halfPtr < 0)                         // to x[n-6].
//                 halfPtr += LPBUFFER_LGTH ;

//             y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
//             y2 = y1;
//             y1 = y0;
//             fdatum = y0 / ((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))/4.0f);
//             lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
//             hp_y += fdatum - hp_data[hp_ptr];
//             halfPtr = hp_ptr-(HPBUFFER_LGTH/2) ;
//             if(halfPtr < 0)
//                 halfPtr += HPBUFFER_LGTH ;
//             hp_data[hp_ptr] = fdatum ;
//             fdatum = hp_data[halfPtr] - (hp_y / (float)HPBUFFER_LGTH);
//             y = fdatum - derBuff[derI] ;
//             derBuff[derI] = fdatum;
//             fdatum = y;
//             fdatum = fabs(fdatum) ;             // Take the absolute value.
//             sum += fdatum - data[ptr] ;
//             data[ptr] = fdatum ;

//             // #ifdef OPERATION_COUNTER
//             // float_comp_counter++;
//             // #endif
//             if((sum / (float)WINDOW_WIDTH) > 32000.f)
//             {
//                 output_temp = 32000.f ;
//             } 
//             else 
//             {
//                 output_temp = sum / (float)WINDOW_WIDTH ;
//                 // #ifdef OPERATION_COUNTER
//                 // float_div_counter += 1;
//                 // #endif
//             }

//             if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
//                 lp_ptr = 0 ;                    // the buffer pointer.
//             if(++derI == DERIV_LENGTH)
//                 derI = 0 ;
//             if(++hp_ptr == HPBUFFER_LGTH)
//                 hp_ptr = 0 ;
//             if(++ptr == WINDOW_WIDTH)
//                 ptr = 0 ;
            
//             // #ifdef OPERATION_COUNTER
//             //     float_add_counter += 10;
//             //     float_mul_counter++;
//             //     float_div_counter += 4;
//             // #endif
//             output[index] = output_temp;
//         }
//     }
// }

// void blocking_no_divisions1(float* input, float* output, int samples_to_process) 
// {
//     // data buffer for lpfilt
//     static float lp_data[LPBUFFER_LGTH];
//     // data buffer for hpfilt
//     static float hp_data[HPBUFFER_LGTH];
//     // data buffer for derivative
//     static float derBuff[DERIV_LENGTH] ;
//     // data buffer for moving window average
//     static float data[WINDOW_WIDTH];
        
//     // ------- initialize filters ------- //
//     //lpfilt
//     for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
//         lp_data[i_init] = 0.f;
//     //hpfilt
//     for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
//         hp_data[i_init] = 0.f;
//     //derivative
//     for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
//         derBuff[i_init] = 0 ;
//     //movint window integration
//     for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
//         data[i_init] = 0 ;
        
//     static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
//     static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
//     int halfPtr, index;
//     float fdatum, y0, z, y, output_temp;
//     float lpbuffer_inv = 1/(float)LPBUFFER_LGTH;
//     float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
//     float window_width_inv = 1/ (float)WINDOW_WIDTH;
//     int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
//     int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

//     for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
//     {
//         for(int j=0; j < BLOCKING_SIZE; j++)
//         {
//             index = i * BLOCKING_SIZE + j;
//             halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
//             if(halfPtr < 0)                         // to x[n-6].
//                 halfPtr += LPBUFFER_LGTH ;

//             y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
//             y2 = y1;
//             y1 = y0;
//             fdatum = y0 * lpbuffer_inv * lpbuffer_inv * 4;
//             lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
//             hp_y += fdatum - hp_data[hp_ptr];
//             halfPtr = hp_ptr - hpbuffer_lgth_half ;
//             if(halfPtr < 0)
//                 halfPtr += HPBUFFER_LGTH ;
//             hp_data[hp_ptr] = fdatum ;
//             fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
//             y = fdatum - derBuff[derI] ;
//             derBuff[derI] = fdatum;
//             fdatum = y;
//             fdatum = fabs(fdatum) ;             // Take the absolute value.
//             sum += fdatum - data[ptr] ;
//             data[ptr] = fdatum ;

//             // #ifdef OPERATION_COUNTER
//             // float_comp_counter++;
//             // #endif
//             if((sum * window_width_inv) > 32000.f)
//             {
//                 output_temp = 32000.f ;
//             } 
//             else 
//             {
//                 output_temp = sum * window_width_inv ;
//                 // #ifdef OPERATION_COUNTER
//                 // float_div_counter += 1;
//                 // #endif
//             }

//             if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
//                 lp_ptr = 0 ;                    // the buffer pointer.
//             if(++derI == DERIV_LENGTH)
//                 derI = 0 ;
//             if(++hp_ptr == HPBUFFER_LGTH)
//                 hp_ptr = 0 ;
//             if(++ptr == WINDOW_WIDTH)
//                 ptr = 0 ;
            
//             // #ifdef OPERATION_COUNTER
//             //     float_add_counter += 10;
//             //     float_mul_counter++;
//             //     float_div_counter += 4;
//             // #endif
//             output[index] = output_temp;
//         }
//     }
// }

// void blocking_no_divisions2(float* input, float* output, int samples_to_process) 
// {
//     // data buffer for lpfilt
//     static float lp_data[LPBUFFER_LGTH];
//     // data buffer for hpfilt
//     static float hp_data[HPBUFFER_LGTH];
//     // data buffer for derivative
//     static float derBuff[DERIV_LENGTH] ;
//     // data buffer for moving window average
//     static float data[WINDOW_WIDTH];
        
//     // ------- initialize filters ------- //
//     //lpfilt
//     for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
//         lp_data[i_init] = 0.f;
//     //hpfilt
//     for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
//         hp_data[i_init] = 0.f;
//     //derivative
//     for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
//         derBuff[i_init] = 0 ;
//     //movint window integration
//     for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
//         data[i_init] = 0 ;
        
//     static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
//     static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
//     int halfPtr, index;
//     float fdatum, y0, z, y, output_temp;
//     float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
//     float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
//     float window_width_inv = 1/ (float)WINDOW_WIDTH;
//     int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
//     int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

//     for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
//     {
//         for(int j=0; j < BLOCKING_SIZE; j++)
//         {
//             index = i * BLOCKING_SIZE + j;
//             halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
//             if(halfPtr < 0)                         // to x[n-6].
//                 halfPtr += LPBUFFER_LGTH ;

//             y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
//             y2 = y1;
//             y1 = y0;
//             fdatum = y0 * lpbuffer_sqr_div_4;
//             lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
//             hp_y += fdatum - hp_data[hp_ptr];
//             halfPtr = hp_ptr- hpbuffer_lgth_half ;
//             if(halfPtr < 0)
//                 halfPtr += HPBUFFER_LGTH ;
//             hp_data[hp_ptr] = fdatum ;
//             fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
//             y = fdatum - derBuff[derI] ;
//             derBuff[derI] = fdatum;
//             fdatum = y;
//             fdatum = fabs(fdatum) ;             // Take the absolute value.
//             sum += fdatum - data[ptr] ;
//             data[ptr] = fdatum ;

//             // #ifdef OPERATION_COUNTER
//             // float_comp_counter++;
//             // #endif
//             if((sum * window_width_inv) > 32000.f)
//             {
//                 output_temp = 32000.f ;
//             } 
//             else 
//             {
//                 output_temp = sum * window_width_inv ;
//                 // #ifdef OPERATION_COUNTER
//                 // float_div_counter += 1;
//                 // #endif
//             }

//             if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
//                 lp_ptr = 0 ;                    // the buffer pointer.
//             if(++derI == DERIV_LENGTH)
//                 derI = 0 ;
//             if(++hp_ptr == HPBUFFER_LGTH)
//                 hp_ptr = 0 ;
//             if(++ptr == WINDOW_WIDTH)
//                 ptr = 0 ;
            
//             // #ifdef OPERATION_COUNTER
//             //     float_add_counter += 10;
//             //     float_mul_counter++;
//             //     float_div_counter += 4;
//             // #endif
//             output[index] = output_temp;
//         }
//     }
// }

// void blocking_no_divisions2_derI(float* input, float* output, int samples_to_process) 
// {
//     // data buffer for lpfilt
//     static float lp_data[LPBUFFER_LGTH];
//     // data buffer for hpfilt
//     static float hp_data[HPBUFFER_LGTH];
//     // data buffer for derivative
//     static float derBuff[DERIV_LENGTH] ;
//     // data buffer for moving window average
//     static float data[WINDOW_WIDTH];
        
//     // ------- initialize filters ------- //
//     //lpfilt
//     for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
//         lp_data[i_init] = 0.f;
//     //hpfilt
//     for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
//         hp_data[i_init] = 0.f;
//     //derivative
//     for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
//         derBuff[i_init] = 0 ;
//     //movint window integration
//     for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
//         data[i_init] = 0 ;
        
//     static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
//     static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
//     int halfPtr, index;
//     float fdatum, y0, z, y, output_temp;
//     float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
//     float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
//     float window_width_inv = 1/ (float)WINDOW_WIDTH;
//     int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
//     int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

//     for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
//     {
//         for(int j=0; j < BLOCKING_SIZE; j++)
//         {
//             index = i * BLOCKING_SIZE + j;
//             halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
//             if(halfPtr < 0)                         // to x[n-6].
//                 halfPtr += LPBUFFER_LGTH ;

//             y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
//             y2 = y1;
//             y1 = y0;
//             fdatum = y0 * lpbuffer_sqr_div_4;
//             lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
//             hp_y += fdatum - hp_data[hp_ptr];
//             halfPtr = hp_ptr - hpbuffer_lgth_half ;
//             if(halfPtr < 0)
//                 halfPtr += HPBUFFER_LGTH ;
//             hp_data[hp_ptr] = fdatum ;
//             fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
//             y = fdatum - derBuff[derI] ;
//             derBuff[derI] = fdatum;
//             fdatum = y;
//             fdatum = fabs(fdatum) ;             // Take the absolute value.
//             sum += fdatum - data[ptr] ;
//             data[ptr] = fdatum ;

//             // #ifdef OPERATION_COUNTER
//             // float_comp_counter++;
//             // #endif
//             if((sum * window_width_inv) > 32000.f)
//             {
//                 output_temp = 32000.f ;
//             } 
//             else 
//             {
//                 output_temp = sum * window_width_inv ;
//                 // #ifdef OPERATION_COUNTER
//                 // float_div_counter += 1;
//                 // #endif
//             }

//             if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
//                 lp_ptr = 0 ;                    // the buffer pointer.
//             if(derI == 0)
//                 derI = 1 ;
//             else
//                 derI = 0 ;
//             if(++hp_ptr == HPBUFFER_LGTH)
//                 hp_ptr = 0 ;
//             if(++ptr == WINDOW_WIDTH)
//                 ptr = 0 ;
            
//             // #ifdef OPERATION_COUNTER
//             //     float_add_counter += 10;
//             //     float_mul_counter++;
//             //     float_div_counter += 4;
//             // #endif
//             output[index] = output_temp;
//         }
//     }
// }

// void blocking_no_divisions2_derI_precomp_sum(float* input, float* output, int samples_to_process) 
// {
//     // data buffer for lpfilt
//     static float lp_data[LPBUFFER_LGTH];
//     // data buffer for hpfilt
//     static float hp_data[HPBUFFER_LGTH];
//     // data buffer for derivative
//     static float derBuff[DERIV_LENGTH] ;
//     // data buffer for moving window average
//     static float data[WINDOW_WIDTH];
        
//     // ------- initialize filters ------- //
//     //lpfilt
//     for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
//         lp_data[i_init] = 0.f;
//     //hpfilt
//     for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
//         hp_data[i_init] = 0.f;
//     //derivative
//     for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
//         derBuff[i_init] = 0 ;
//     //movint window integration
//     for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
//         data[i_init] = 0 ;
        
//     static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0, sum_window = 0.0;
//     static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
//     int halfPtr, index;
//     float fdatum, y0, z, y, output_temp;
//     float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
//     float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
//     float window_width_inv = 1/ (float)WINDOW_WIDTH;
//     int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
//     int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);
//     int i;
//     for(i=0; i < samples_to_process - BLOCKING_SIZE + 1; i+= BLOCKING_SIZE)
//     {
//         for(int j=0; j < BLOCKING_SIZE; j++)
//         {
//             index = i + j;
//             halfPtr = lp_ptr- lpbuffer_lgth_half ;    // Use halfPtr to index
//             if(halfPtr < 0)                         // to x[n-6].
//                 halfPtr += LPBUFFER_LGTH ;

//             y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
//             y2 = y1;
//             y1 = y0;
//             fdatum = y0 * lpbuffer_sqr_div_4;
//             lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
//             hp_y += fdatum - hp_data[hp_ptr];
//             halfPtr = hp_ptr- hpbuffer_lgth_half ;
//             if(halfPtr < 0)
//                 halfPtr += HPBUFFER_LGTH ;
//             hp_data[hp_ptr] = fdatum ;
//             fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
//             y = fdatum - derBuff[derI] ;
//             derBuff[derI] = fdatum;
//             fdatum = y;
//             fdatum = fabs(fdatum) ;             // Take the absolute value.
//             sum += fdatum - data[ptr] ;
//             data[ptr] = fdatum ;

//             // #ifdef OPERATION_COUNTER
//             // float_comp_counter++;
//             // #endif
//             sum_window = sum * window_width_inv;
//             if((sum_window) > 32000.f)
//             {
//                 output_temp = 32000.f ;
//             } 
//             else 
//             {
//                 output_temp = sum_window ;
//                 // #ifdef OPERATION_COUNTER
//                 // float_div_counter += 1;
//                 // #endif
//             }

//             if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
//                 lp_ptr = 0 ;                    // the buffer pointer.
//             if(derI == 0)
//                 derI = 1 ;
//             else
//                 derI = 0 ;
//             if(++hp_ptr == HPBUFFER_LGTH)
//                 hp_ptr = 0 ;
//             if(++ptr == WINDOW_WIDTH)
//                 ptr = 0 ;
            
//             // #ifdef OPERATION_COUNTER
//             //     float_add_counter += 10;
//             //     float_mul_counter++;
//             //     float_div_counter += 4;
//             // #endif
//             output[index] = output_temp;
//         }
//     }
//     for(; i < samples_to_process; i++)
//     {
//         halfPtr = lp_ptr- lpbuffer_lgth_half ;    // Use halfPtr to index
//         if(halfPtr < 0)                         // to x[n-6].
//             halfPtr += LPBUFFER_LGTH ;

//         y0 = (y1*2.0f) - y2 + input[i] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
//         y2 = y1;
//         y1 = y0;
//         fdatum = y0 * lpbuffer_sqr_div_4;
//         lp_data[lp_ptr] = input[i] ;            // Stick most recent sample into
        
//         hp_y += fdatum - hp_data[hp_ptr];
//         halfPtr = hp_ptr- hpbuffer_lgth_half ;
//         if(halfPtr < 0)
//             halfPtr += HPBUFFER_LGTH ;
//         hp_data[hp_ptr] = fdatum ;
//         fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
//         y = fdatum - derBuff[derI] ;
//         derBuff[derI] = fdatum;
//         fdatum = y;
//         fdatum = fabs(fdatum) ;             // Take the absolute value.
//         sum += fdatum - data[ptr] ;
//         data[ptr] = fdatum ;

//         // #ifdef OPERATION_COUNTER
//         // float_comp_counter++;
//         // #endif
//         sum_window = sum * window_width_inv;
//         if((sum_window) > 32000.f)
//         {
//             output_temp = 32000.f ;
//         } 
//         else 
//         {
//             output_temp = sum_window ;
//             // #ifdef OPERATION_COUNTER
//             // float_div_counter += 1;
//             // #endif
//         }

//         if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
//             lp_ptr = 0 ;                    // the buffer pointer.
//         if(derI == 0)
//             derI = 1 ;
//         else
//             derI = 0 ;
//         if(++hp_ptr == HPBUFFER_LGTH)
//             hp_ptr = 0 ;
//         if(++ptr == WINDOW_WIDTH)
//             ptr = 0 ;
        
//         // #ifdef OPERATION_COUNTER
//         //     float_add_counter += 10;
//         //     float_mul_counter++;
//         //     float_div_counter += 4;
//         // #endif
//         output[i] = output_temp;
//     }
// }

// void no_divisions2_derI_precomp_sum(float* input, float* output, int samples_to_process) 
// {
//     // data buffer for lpfilt
//     static float lp_data[LPBUFFER_LGTH];
//     // data buffer for hpfilt
//     static float hp_data[HPBUFFER_LGTH];
//     // data buffer for derivative
//     static float derBuff[DERIV_LENGTH] ;
//     // data buffer for moving window average
//     static float data[WINDOW_WIDTH];
        
//     // ------- initialize filters ------- //
//     //lpfilt
//     for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
//         lp_data[i_init] = 0.f;
//     //hpfilt
//     for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
//         hp_data[i_init] = 0.f;
//     //derivative
//     for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
//         derBuff[i_init] = 0 ;
//     //movint window integration
//     for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
//         data[i_init] = 0 ;
        
//     static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0, sum_window = 0.0;
//     static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
//     int halfPtr, index;
//     float fdatum, y0, z, y, output_temp;
//     float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
//     float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
//     float window_width_inv = 1/ (float)WINDOW_WIDTH;
//     int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
//     int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

//     for(int index=0; index < samples_to_process; index++)
//     {
//             halfPtr = lp_ptr- lpbuffer_lgth_half ;    // Use halfPtr to index
//             if(halfPtr < 0)                         // to x[n-6].
//                 halfPtr += LPBUFFER_LGTH ;

//             y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
//             y2 = y1;
//             y1 = y0;
//             fdatum = y0 * lpbuffer_sqr_div_4;
//             lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
//             hp_y += fdatum - hp_data[hp_ptr];
//             halfPtr = hp_ptr- hpbuffer_lgth_half ;
//             if(halfPtr < 0)
//                 halfPtr += HPBUFFER_LGTH ;
//             hp_data[hp_ptr] = fdatum ;
//             fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
//             y = fdatum - derBuff[derI] ;
//             derBuff[derI] = fdatum;
//             fdatum = y;
//             fdatum = fabs(fdatum) ;             // Take the absolute value.
//             sum += fdatum - data[ptr] ;
//             data[ptr] = fdatum ;

//             // #ifdef OPERATION_COUNTER
//             // float_comp_counter++;
//             // #endif
//             sum_window = sum * window_width_inv;
//             if((sum_window) > 32000.f)
//             {
//                 output_temp = 32000.f ;
//             } 
//             else 
//             {
//                 output_temp = sum_window ;
//                 // #ifdef OPERATION_COUNTER
//                 // float_div_counter += 1;
//                 // #endif
//             }

//             if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
//                 lp_ptr = 0 ;                    // the buffer pointer.
//             if(derI == 0)
//                 derI = 1 ;
//             else
//                 derI = 0 ;
//             if(++hp_ptr == HPBUFFER_LGTH)
//                 hp_ptr = 0 ;
//             if(++ptr == WINDOW_WIDTH)
//                 ptr = 0 ;
            
//             // #ifdef OPERATION_COUNTER
//             //     float_add_counter += 10;
//             //     float_mul_counter++;
//             //     float_div_counter += 4;
//             // #endif
//             output[index] = output_temp;
//     }
// }

// void no_division(float* input, float* output, int samples_to_process) 
// {
//     // std::cout << "loop length "<< samples_to_process<<"\n";
//     // data buffer for lpfilt
//     static float lp_data[LPBUFFER_LGTH];
//     // data buffer for hpfilt
//     static float hp_data[HPBUFFER_LGTH];
//     // data buffer for derivative
//     static float derBuff[DERIV_LENGTH] ;
//     // data buffer for moving window average
//     static float data[WINDOW_WIDTH];
        
//     // ------- initialize filters ------- //
//     //lpfilt
//     for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
//         lp_data[i_init] = 0.f;
//     //hpfilt
//     for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
//         hp_data[i_init] = 0.f;
//     //derivative
//     for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
//         derBuff[i_init] = 0 ;
//     //movint window integration
//     for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
//         data[i_init] = 0 ;
        
//     static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
//     static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
//     int halfPtr, index;
//     float fdatum, y0, z, y, output_temp;
//     float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
//     float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
//     float window_width_inv = 1/ (float)WINDOW_WIDTH;
//     int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
//     int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

//     for(int index = 0; index < samples_to_process; index++)
//     {
//         halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
//         if(halfPtr < 0)                         // to x[n-6].
//             halfPtr += LPBUFFER_LGTH ;

//         y0 = (y1*2.0f) - y2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
//         y2 = y1;
//         y1 = y0;
//         fdatum = y0 * lpbuffer_sqr_div_4;
//         lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
        
//         hp_y += fdatum - hp_data[hp_ptr];
//         halfPtr = hp_ptr - hpbuffer_lgth_half ;
//         if(halfPtr < 0)
//             halfPtr += HPBUFFER_LGTH ;
//         hp_data[hp_ptr] = fdatum ;
//         fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
//         y = fdatum - derBuff[derI] ;
//         derBuff[derI] = fdatum;
//         fdatum = y;
//         fdatum = fabs(fdatum) ;             // Take the absolute value.
//         sum += fdatum - data[ptr] ;
//         data[ptr] = fdatum ;

//         // #ifdef OPERATION_COUNTER
//         // float_comp_counter++;
//         // #endif
//         if((sum * window_width_inv) > 32000.f)
//         {
//             output_temp = 32000.f ;
//         } 
//         else 
//         {
//             output_temp = sum * window_width_inv;
//             // #ifdef OPERATION_COUNTER
//             // float_div_counter += 1;
//             // #endif
//         }

//         if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
//             lp_ptr = 0 ;                    // the buffer pointer.
//         if(++derI == DERIV_LENGTH)
//             derI = 0 ;
//         if(++hp_ptr == HPBUFFER_LGTH)
//             hp_ptr = 0 ;
//         if(++ptr == WINDOW_WIDTH)
//             ptr = 0 ;
        
//         // #ifdef OPERATION_COUNTER
//         //     float_add_counter += 10;
//         //     float_mul_counter++;
//         //     float_div_counter += 4;
//         // #endif
//         output[index] = output_temp;
//     }
// }

// void blocking_no_divisions_factorized(float* input, float* output, int samples_to_process) 
// {
//     // data buffer for lpfilt
//     static float lp_data[LPBUFFER_LGTH];
//     // data buffer for hpfilt
//     static float hp_data[HPBUFFER_LGTH];
//     // data buffer for derivative
//     static float derBuff[DERIV_LENGTH] ;
//     // data buffer for moving window average
//     static float data[WINDOW_WIDTH];
        
//     // ------- initialize filters ------- //
//     //lpfilt
//     for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
//         lp_data[i_init] = 0.f;
//     //hpfilt
//     for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
//         hp_data[i_init] = 0.f;
//     //derivative
//     for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
//         derBuff[i_init] = 0 ;
//     //movint window integration
//     for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
//         data[i_init] = 0 ;
        
//     static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0;
//     static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
//     int halfPtr, index;
//     float fdatum, y0, z, y, output_temp;
//     float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
//     float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
//     float window_width_inv = 1/ (float)WINDOW_WIDTH;
//     int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
//     int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

//     for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
//     {
//         for(int j=0; j < BLOCKING_SIZE; j++)
//         {
//             index = i * BLOCKING_SIZE + j;
//             halfPtr = lp_ptr - lpbuffer_lgth_half;    // Use halfPtr to index
//             if(halfPtr < 0)                         // to x[n-6].
//                 halfPtr += LPBUFFER_LGTH ;

//             y0 = ((y1- lp_data[halfPtr])*2.0f) - y2 + input[index] + lp_data[lp_ptr] ;
//             y2 = y1;
//             y1 = y0;
//             fdatum = y0 * lpbuffer_sqr_div_4;
//             lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
            
//             hp_y += fdatum - hp_data[hp_ptr];
//             halfPtr = hp_ptr - hpbuffer_lgth_half ;
//             if(halfPtr < 0)
//                 halfPtr += HPBUFFER_LGTH ;
//             hp_data[hp_ptr] = fdatum ;
//             fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
//             y = fdatum - derBuff[derI] ;
//             derBuff[derI] = fdatum;
//             fdatum = y;
//             fdatum = fabs(fdatum) ;             // Take the absolute value.
//             sum += fdatum - data[ptr] ;
//             data[ptr] = fdatum ;

//             // #ifdef OPERATION_COUNTER
//             // float_comp_counter++;
//             // #endif
//             if((sum * window_width_inv) > 32000.f)
//             {
//                 output_temp = 32000.f ;
//             } 
//             else 
//             {
//                 output_temp = sum * window_width_inv ;
//                 // #ifdef OPERATION_COUNTER
//                 // float_div_counter += 1;
//                 // #endif
//             }

//             if(++lp_ptr == LPBUFFER_LGTH)   // the circular buffer and update
//                 lp_ptr = 0 ;                    // the buffer pointer.
//             if(++derI == DERIV_LENGTH)
//                 derI = 0 ;
//             if(++hp_ptr == HPBUFFER_LGTH)
//                 hp_ptr = 0 ;
//             if(++ptr == WINDOW_WIDTH)
//                 ptr = 0 ;
            
//             // #ifdef OPERATION_COUNTER
//             //     float_add_counter += 10;
//             //     float_mul_counter++;
//             //     float_div_counter += 4;
//             // #endif
//             output[index] = output_temp;
//         }
//     }
// }

// void blocking_no_divisions_unrolled(float* input, float* output, int samples_to_process) 
// {
//     // data buffer for lpfilt
//     static float lp_data[LPBUFFER_LGTH];
//     // data buffer for hpfilt
//     static float hp_data[HPBUFFER_LGTH];
//     // data buffer for derivative
//     static float derBuff[DERIV_LENGTH] ;
//     // data buffer for moving window average
//     static float data[WINDOW_WIDTH];
        
//     // ------- initialize filters ------- //
//     //lpfilt
//     for(int i_init = 0; i_init < LPBUFFER_LGTH; ++i_init)
//         lp_data[i_init] = 0.f;
//     //hpfilt
//     for(int i_init = 0; i_init < HPBUFFER_LGTH; ++i_init)
//         hp_data[i_init] = 0.f;
//     //derivative
//     for(int i_init = 0; i_init < DERIV_LENGTH; ++i_init)
//         derBuff[i_init] = 0 ;
//     //movint window integration
//     for(int i_init = 0; i_init < WINDOW_WIDTH ; ++i_init)
//         data[i_init] = 0 ;
        
//     static float y1 = 0.0, y2 = 0.0, y1_2 = 0.0, y2_2 = 0.0, hp_y = 0.0, hp_y2 = 0.0, sum = 0.0;
//     static int lp_ptr = 0, lp_ptr2 = 1, hp_ptr = 0, hp_ptr2 = 1, derI = 0, derI2 = 1, ptr = 0, ptr2 = 1;
//     int halfPtr, halfPtr2, index, index2;
//     float fdatum, fdatum2, y0, y0_2, z, y, output_temp, output_temp2;
//     float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);
//     float hpbuffer_lgth_inv = 1/(float)HPBUFFER_LGTH;
//     float window_width_inv = 1/ (float)WINDOW_WIDTH;
//     int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
//     int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);

//     for(int i=0; i < samples_to_process/BLOCKING_SIZE; i++)
//     {
//         for(int j=0; j < BLOCKING_SIZE-1; j+=2)
//         {
//             index = i * BLOCKING_SIZE + j;
//             index2 = index + 1;

//             halfPtr = lp_ptr - lpbuffer_lgth_half ;    // Use halfPtr to index
//             if(halfPtr < 0)                         // to x[n-6].
//                 halfPtr += LPBUFFER_LGTH ;
 
//             halfPtr2 = lp_ptr2 - lpbuffer_lgth_half ;    // Use halfPtr to index 
//             if(halfPtr2 < 0)                         // to x[n-6].
//                 halfPtr2 += LPBUFFER_LGTH ;

//             //lp_data beein overwritten later should not be a problem since halfPtr and lp_ptr are 5 (lpbuff_lgth = 10) apart
//             y0 = (y1_2*2.0f) - y2_2 + input[index] - (lp_data[halfPtr]*2.0f) + lp_data[lp_ptr] ;
//             y2 = y1_2;
//             y1 = y0;
//             fdatum = y0 * lpbuffer_sqr_div_4;
//             y0_2 = (y1*2.0f) - y2 + input[index2] - (lp_data[halfPtr2]*2.0f) + lp_data[lp_ptr2] ;
//             y2_2 = y1;
//             y1_2 = y0_2;
//             fdatum2 = y0_2 * lpbuffer_sqr_div_4;

//             lp_data[lp_ptr] = input[index] ;            // Stick most recent sample into
//             lp_data[lp_ptr2] = input[index2] ;            // Stick most recent sample into
            
//             hp_y += fdatum - hp_data[hp_ptr];
//             hp_y2 += fdatum2 - hp_data[hp_ptr2];

//             halfPtr = hp_ptr - hpbuffer_lgth_half ;
//             if(halfPtr < 0)
//                 halfPtr += HPBUFFER_LGTH ;
//             halfPtr2 = hp_ptr2 - hpbuffer_lgth_half ;
//             if(halfPtr2 < 0)
//                 halfPtr2 += HPBUFFER_LGTH ;

//             hp_data[hp_ptr] = fdatum ;
//             hp_data[hp_ptr2] = fdatum2 ;
//             fdatum = hp_data[halfPtr] - (hp_y * hpbuffer_lgth_inv);
//             fdatum2 = hp_data[halfPtr2] - (hp_y2 * hpbuffer_lgth_inv);
//             y = fdatum - derBuff[derI] ;
//             y2 = fdatum2 - derBuff[derI2] ;

//             derBuff[derI] = fdatum;
//             derBuff[derI2] = fdatum2;
//             fdatum = y;
//             fdatum2 = y2;

//             fdatum = fabs(fdatum) ;             // Take the absolute value.
//             fdatum2 = fabs(fdatum2) ;             // Take the absolute value.


//             sum += fdatum - data[ptr] ;
//             data[ptr] = fdatum ;

//             // #ifdef OPERATION_COUNTER
//             // float_comp_counter++;
//             // #endif
//             if((sum * window_width_inv) > 32000.f)
//             {
//                 output_temp = 32000.f ;
//             } 
//             else 
//             {
//                 output_temp = sum * window_width_inv ;
//                 // #ifdef OPERATION_COUNTER
//                 // float_div_counter += 1;
//                 // #endif
//             }
//             output[index] = output_temp;


//             sum += fdatum2 - data[ptr2] ;
//             data[ptr2] = fdatum2 ;

//             // #ifdef OPERATION_COUNTER
//             // float_comp_counter++;
//             // #endif
//             if((sum * window_width_inv) > 32000.f)
//             {
//                 output_temp2 = 32000.f ;
//             } 
//             else 
//             {
//                 output_temp2 = sum * window_width_inv ;
//                 // #ifdef OPERATION_COUNTER
//                 // float_div_counter += 1;
//                 // #endif
//             }
//             output[index2] = output_temp2;



//             //DERIV_LENGTH is 2, therefore derI and derI2 stay 0 and 1
//             lp_ptr += 2;
//             lp_ptr2 += 2;
//             hp_ptr += 2;
//             hp_ptr2 += 2;
//             ptr += 2;
//             ptr2 += 2;
//             // derI += 2;
//             // derI2 += 2;

//             // the buffer pointer.
//             if(lp_ptr >= LPBUFFER_LGTH)   // the circular buffer and update
//                 lp_ptr -= LPBUFFER_LGTH ;
//             if(lp_ptr2 >= LPBUFFER_LGTH)   // the circular buffer and update
//                 lp_ptr2 -= LPBUFFER_LGTH ; 
//             // if(derI >= DERIV_LENGTH)
//             //     derI -= DERIV_LENGTH ;
//             // if(derI2 >= DERIV_LENGTH)
//             //     derI2 -= DERIV_LENGTH ;
//             if(hp_ptr >= HPBUFFER_LGTH)
//                 hp_ptr -= HPBUFFER_LGTH ;
//             if(hp_ptr2 >= HPBUFFER_LGTH)
//                 hp_ptr2 -= HPBUFFER_LGTH ;
//             if(ptr >= WINDOW_WIDTH)
//                 ptr -= WINDOW_WIDTH ;
//             if(ptr2 >= WINDOW_WIDTH)
//                 ptr2 -= WINDOW_WIDTH ;
            
//             // #ifdef OPERATION_COUNTER
//             //     float_add_counter += 10;
//             //     float_mul_counter++;
//             //     float_div_counter += 4;
//             // #endif
//         }
//     }
// }


void qrsfilt_opt_400_50_1(float* datum, float* filtOutput, int sampleLength) 
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
        
    static float y1 = 0.0, y2 = 0.0, hp_y = 0.0, sum = 0.0, sum_temp = 0.0;
    static int lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    int halfPtr, halfPtr1, halfPtr2, index;
    float fdatum, fdatum2, fdatum3, y0, z, y, output_temp;
    static float LPBUFFER_LGTH_INV = 1/((float)LPBUFFER_LGTH);
    static float HPBUFFER_LGTH_INV = 1/((float)HPBUFFER_LGTH);
    static float WINDOW_WIDTH_INV = 1/((float)WINDOW_WIDTH);
    static float lpbuffer_sqr_div_4 = 1/((((float)LPBUFFER_LGTH)*((float)LPBUFFER_LGTH))*0.25);


    for(index = 0; index <= sampleLength - 400 ; index+=400)
    {
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index       ,0  ,0  ,0  ,0  ,5 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+1     ,1  ,1  ,1  ,1  ,6 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+2     ,2  ,0  ,2  ,2  ,7 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+3     ,3  ,1  ,3  ,3  ,8 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+4     ,4  ,0  ,4  ,4  ,9 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+5     ,5  ,1  ,5  ,5  ,0 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+6     ,6  ,0  ,6  ,6  ,1 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+7     ,7  ,1  ,7  ,7  ,2 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+8     ,8  ,0  ,8  ,8  ,3 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+9     ,9  ,1  ,9  ,9  ,4 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+10    ,0  ,0  ,10 ,10 ,5 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+11    ,1  ,1  ,11 ,11 ,6 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+12    ,2  ,0  ,12 ,12 ,7 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+13    ,3  ,1  ,13 ,13 ,8 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+14    ,4  ,0  ,14 ,14 ,9 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+15    ,5  ,1  ,15 ,15 ,0 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+16    ,6  ,0  ,16 ,0  ,1 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+17    ,7  ,1  ,17 ,1  ,2 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+18    ,8  ,0  ,18 ,2  ,3 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+19    ,9  ,1  ,19 ,3  ,4 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+20    ,0  ,0  ,20 ,4  ,5 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+21    ,1  ,1  ,21 ,5  ,6 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+22    ,2  ,0  ,22 ,6  ,7 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+23    ,3  ,1  ,23 ,7  ,8 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+24    ,4  ,0  ,24 ,8  ,9 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+25    ,5  ,1  ,0  ,9  ,0 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+26    ,6  ,0  ,1  ,10 ,1 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+27    ,7  ,1  ,2  ,11 ,2 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+28    ,8  ,0  ,3  ,12 ,3 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+29    ,9  ,1  ,4  ,13 ,4 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+30    ,0  ,0  ,5  ,14 ,5 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+31    ,1  ,1  ,6  ,15 ,6 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+32    ,2  ,0  ,7  ,0  ,7 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+33    ,3  ,1  ,8  ,1  ,8 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+34    ,4  ,0  ,9  ,2  ,9 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+35    ,5  ,1  ,10 ,3  ,0 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+36    ,6  ,0  ,11 ,4  ,1 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+37    ,7  ,1  ,12 ,5  ,2 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+38    ,8  ,0  ,13 ,6  ,3 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+39    ,9  ,1  ,14 ,7  ,4 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+40    ,0  ,0  ,15 ,8  ,5 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+41    ,1  ,1  ,16 ,9  ,6 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+42    ,2  ,0  ,17 ,10 ,7 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+43    ,3  ,1  ,18 ,11 ,8 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+44    ,4  ,0  ,19 ,12 ,9 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+45    ,5  ,1  ,20 ,13 ,0 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+46    ,6  ,0  ,21 ,14 ,1 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+47    ,7  ,1  ,22 ,15 ,2 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+48    ,8  ,0  ,23 ,0  ,3 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+49    ,9  ,1  ,24 ,1  ,4 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+50    ,0  ,0  ,0  ,2  ,5 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+51    ,1  ,1  ,1  ,3  ,6 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+52    ,2  ,0  ,2  ,4  ,7 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+53    ,3  ,1  ,3  ,5  ,8 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+54    ,4  ,0  ,4  ,6  ,9 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+55    ,5  ,1  ,5  ,7  ,0 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+56    ,6  ,0  ,6  ,8  ,1 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+57    ,7  ,1  ,7  ,9  ,2 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+58    ,8  ,0  ,8  ,10 ,3 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+59    ,9  ,1  ,9  ,11 ,4 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+60    ,0  ,0  ,10 ,12 ,5 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+61    ,1  ,1  ,11 ,13 ,6 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+62    ,2  ,0  ,12 ,14 ,7 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+63    ,3  ,1  ,13 ,15 ,8 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+64    ,4  ,0  ,14 ,0  ,9 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+65    ,5  ,1  ,15 ,1  ,0 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+66    ,6  ,0  ,16 ,2  ,1 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+67    ,7  ,1  ,17 ,3  ,2 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+68    ,8  ,0  ,18 ,4  ,3 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+69    ,9  ,1  ,19 ,5  ,4 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+70    ,0  ,0  ,20 ,6  ,5 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+71    ,1  ,1  ,21 ,7  ,6 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+72    ,2  ,0  ,22 ,8  ,7 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+73    ,3  ,1  ,23 ,9  ,8 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+74    ,4  ,0  ,24 ,10 ,9 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+75    ,5  ,1  ,0  ,11 ,0 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+76    ,6  ,0  ,1  ,12 ,1 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+77    ,7  ,1  ,2  ,13 ,2 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+78    ,8  ,0  ,3  ,14 ,3 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+79    ,9  ,1  ,4  ,15 ,4 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+80    ,0  ,0  ,5  ,0  ,5 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+81    ,1  ,1  ,6  ,1  ,6 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+82    ,2  ,0  ,7  ,2  ,7 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+83    ,3  ,1  ,8  ,3  ,8 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+84    ,4  ,0  ,9  ,4  ,9 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+85    ,5  ,1  ,10 ,5  ,0 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+86    ,6  ,0  ,11 ,6  ,1 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+87    ,7  ,1  ,12 ,7  ,2 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+88    ,8  ,0  ,13 ,8  ,3 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+89    ,9  ,1  ,14 ,9  ,4 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+90    ,0  ,0  ,15 ,10 ,5 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+91    ,1  ,1  ,16 ,11 ,6 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+92    ,2  ,0  ,17 ,12 ,7 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+93    ,3  ,1  ,18 ,13 ,8 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+94    ,4  ,0  ,19 ,14 ,9 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+95    ,5  ,1  ,20 ,15 ,0 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+96    ,6  ,0  ,21 ,0  ,1 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+97    ,7  ,1  ,22 ,1  ,2 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+98    ,8  ,0  ,23 ,2  ,3 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+99    ,9  ,1  ,24 ,3  ,4 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+100   ,0  ,0  ,0  ,4  ,5 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+101   ,1  ,1  ,1  ,5  ,6 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+102   ,2  ,0  ,2  ,6  ,7 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+103   ,3  ,1  ,3  ,7  ,8 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+104   ,4  ,0  ,4  ,8  ,9 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+105   ,5  ,1  ,5  ,9  ,0 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+106   ,6  ,0  ,6  ,10 ,1 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+107   ,7  ,1  ,7  ,11 ,2 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+108   ,8  ,0  ,8  ,12 ,3 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+109   ,9  ,1  ,9  ,13 ,4 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+110   ,0  ,0  ,10 ,14 ,5 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+111   ,1  ,1  ,11 ,15 ,6 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+112   ,2  ,0  ,12 ,0  ,7 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+113   ,3  ,1  ,13 ,1  ,8 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+114   ,4  ,0  ,14 ,2  ,9 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+115   ,5  ,1  ,15 ,3  ,0 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+116   ,6  ,0  ,16 ,4  ,1 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+117   ,7  ,1  ,17 ,5  ,2 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+118   ,8  ,0  ,18 ,6  ,3 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+119   ,9  ,1  ,19 ,7  ,4 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+120   ,0  ,0  ,20 ,8  ,5 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+121   ,1  ,1  ,21 ,9  ,6 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+122   ,2  ,0  ,22 ,10 ,7 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+123   ,3  ,1  ,23 ,11 ,8 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+124   ,4  ,0  ,24 ,12 ,9 ,12 )      
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+125   ,5  ,1  ,0  ,13 ,0 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+126   ,6  ,0  ,1  ,14 ,1 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+127   ,7  ,1  ,2  ,15 ,2 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+128   ,8  ,0  ,3  ,0  ,3 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+129   ,9  ,1  ,4  ,1  ,4 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+130   ,0  ,0  ,5  ,2  ,5 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+131   ,1  ,1  ,6  ,3  ,6 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+132   ,2  ,0  ,7  ,4  ,7 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+133   ,3  ,1  ,8  ,5  ,8 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+134   ,4  ,0  ,9  ,6  ,9 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+135   ,5  ,1  ,10 ,7  ,0 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+136   ,6  ,0  ,11 ,8  ,1 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+137   ,7  ,1  ,12 ,9  ,2 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+138   ,8  ,0  ,13 ,10 ,3 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+139   ,9  ,1  ,14 ,11 ,4 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+140   ,0  ,0  ,15 ,12 ,5 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+141   ,1  ,1  ,16 ,13 ,6 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+142   ,2  ,0  ,17 ,14 ,7 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+143   ,3  ,1  ,18 ,15 ,8 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+144   ,4  ,0  ,19 ,0  ,9 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+145   ,5  ,1  ,20 ,1  ,0 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+146   ,6  ,0  ,21 ,2  ,1 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+147   ,7  ,1  ,22 ,3  ,2 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+148   ,8  ,0  ,23 ,4  ,3 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+149   ,9  ,1  ,24 ,5  ,4 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+150   ,0  ,0  ,0  ,6  ,5 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+151   ,1  ,1  ,1  ,7  ,6 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+152   ,2  ,0  ,2  ,8  ,7 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+153   ,3  ,1  ,3  ,9  ,8 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+154   ,4  ,0  ,4  ,10 ,9 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+155   ,5  ,1  ,5  ,11 ,0 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+156   ,6  ,0  ,6  ,12 ,1 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+157   ,7  ,1  ,7  ,13 ,2 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+158   ,8  ,0  ,8  ,14 ,3 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+159   ,9  ,1  ,9  ,15 ,4 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+160   ,0  ,0  ,10 ,0  ,5 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+161   ,1  ,1  ,11 ,1  ,6 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+162   ,2  ,0  ,12 ,2  ,7 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+163   ,3  ,1  ,13 ,3  ,8 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+164   ,4  ,0  ,14 ,4  ,9 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+165   ,5  ,1  ,15 ,5  ,0 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+166   ,6  ,0  ,16 ,6  ,1 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+167   ,7  ,1  ,17 ,7  ,2 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+168   ,8  ,0  ,18 ,8  ,3 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+169   ,9  ,1  ,19 ,9  ,4 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+170   ,0  ,0  ,20 ,10 ,5 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+171   ,1  ,1  ,21 ,11 ,6 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+172   ,2  ,0  ,22 ,12 ,7 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+173   ,3  ,1  ,23 ,13 ,8 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+174   ,4  ,0  ,24 ,14 ,9 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+175   ,5  ,1  ,0  ,15 ,0 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+176   ,6  ,0  ,1  ,0  ,1 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+177   ,7  ,1  ,2  ,1  ,2 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+178   ,8  ,0  ,3  ,2  ,3 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+179   ,9  ,1  ,4  ,3  ,4 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+180   ,0  ,0  ,5  ,4  ,5 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+181   ,1  ,1  ,6  ,5  ,6 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+182   ,2  ,0  ,7  ,6  ,7 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+183   ,3  ,1  ,8  ,7  ,8 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+184   ,4  ,0  ,9  ,8  ,9 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+185   ,5  ,1  ,10 ,9  ,0 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+186   ,6  ,0  ,11 ,10 ,1 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+187   ,7  ,1  ,12 ,11 ,2 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+188   ,8  ,0  ,13 ,12 ,3 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+189   ,9  ,1  ,14 ,13 ,4 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+190   ,0  ,0  ,15 ,14 ,5 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+191   ,1  ,1  ,16 ,15 ,6 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+192   ,2  ,0  ,17 ,0  ,7 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+193   ,3  ,1  ,18 ,1  ,8 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+194   ,4  ,0  ,19 ,2  ,9 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+195   ,5  ,1  ,20 ,3  ,0 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+196   ,6  ,0  ,21 ,4  ,1 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+197   ,7  ,1  ,22 ,5  ,2 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+198   ,8  ,0  ,23 ,6  ,3 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+199   ,9  ,1  ,24 ,7  ,4 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+200   ,0  ,0  ,0  ,8  ,5 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+201   ,1  ,1  ,1  ,9  ,6 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+202   ,2  ,0  ,2  ,10 ,7 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+203   ,3  ,1  ,3  ,11 ,8 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+204   ,4  ,0  ,4  ,12 ,9 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+205   ,5  ,1  ,5  ,13 ,0 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+206   ,6  ,0  ,6  ,14 ,1 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+207   ,7  ,1  ,7  ,15 ,2 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+208   ,8  ,0  ,8  ,0  ,3 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+209   ,9  ,1  ,9  ,1  ,4 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+210   ,0  ,0  ,10 ,2  ,5 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+211   ,1  ,1  ,11 ,3  ,6 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+212   ,2  ,0  ,12 ,4  ,7 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+213   ,3  ,1  ,13 ,5  ,8 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+214   ,4  ,0  ,14 ,6  ,9 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+215   ,5  ,1  ,15 ,7  ,0 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+216   ,6  ,0  ,16 ,8  ,1 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+217   ,7  ,1  ,17 ,9  ,2 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+218   ,8  ,0  ,18 ,10 ,3 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+219   ,9  ,1  ,19 ,11 ,4 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+220   ,0  ,0  ,20 ,12 ,5 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+221   ,1  ,1  ,21 ,13 ,6 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+222   ,2  ,0  ,22 ,14 ,7 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+223   ,3  ,1  ,23 ,15 ,8 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+224   ,4  ,0  ,24 ,0  ,9 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+225   ,5  ,1  ,0  ,1  ,0 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+226   ,6  ,0  ,1  ,2  ,1 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+227   ,7  ,1  ,2  ,3  ,2 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+228   ,8  ,0  ,3  ,4  ,3 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+229   ,9  ,1  ,4  ,5  ,4 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+230   ,0  ,0  ,5  ,6  ,5 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+231   ,1  ,1  ,6  ,7  ,6 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+232   ,2  ,0  ,7  ,8  ,7 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+233   ,3  ,1  ,8  ,9  ,8 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+234   ,4  ,0  ,9  ,10 ,9 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+235   ,5  ,1  ,10 ,11 ,0 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+236   ,6  ,0  ,11 ,12 ,1 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+237   ,7  ,1  ,12 ,13 ,2 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+238   ,8  ,0  ,13 ,14 ,3 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+239   ,9  ,1  ,14 ,15 ,4 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+240   ,0  ,0  ,15 ,0  ,5 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+241   ,1  ,1  ,16 ,1  ,6 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+242   ,2  ,0  ,17 ,2  ,7 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+243   ,3  ,1  ,18 ,3  ,8 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+244   ,4  ,0  ,19 ,4  ,9 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+245   ,5  ,1  ,20 ,5  ,0 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+246   ,6  ,0  ,21 ,6  ,1 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+247   ,7  ,1  ,22 ,7  ,2 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+248   ,8  ,0  ,23 ,8  ,3 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+249   ,9  ,1  ,24 ,9  ,4 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+250   ,0  ,0  ,0  ,10 ,5 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+251   ,1  ,1  ,1  ,11 ,6 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+252   ,2  ,0  ,2  ,12 ,7 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+253   ,3  ,1  ,3  ,13 ,8 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+254   ,4  ,0  ,4  ,14 ,9 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+255   ,5  ,1  ,5  ,15 ,0 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+256   ,6  ,0  ,6  ,0  ,1 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+257   ,7  ,1  ,7  ,1  ,2 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+258   ,8  ,0  ,8  ,2  ,3 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+259   ,9  ,1  ,9  ,3  ,4 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+260   ,0  ,0  ,10 ,4  ,5 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+261   ,1  ,1  ,11 ,5  ,6 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+262   ,2  ,0  ,12 ,6  ,7 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+263   ,3  ,1  ,13 ,7  ,8 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+264   ,4  ,0  ,14 ,8  ,9 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+265   ,5  ,1  ,15 ,9  ,0 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+266   ,6  ,0  ,16 ,10 ,1 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+267   ,7  ,1  ,17 ,11 ,2 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+268   ,8  ,0  ,18 ,12 ,3 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+269   ,9  ,1  ,19 ,13 ,4 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+270   ,0  ,0  ,20 ,14 ,5 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+271   ,1  ,1  ,21 ,15 ,6 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+272   ,2  ,0  ,22 ,0  ,7 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+273   ,3  ,1  ,23 ,1  ,8 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+274   ,4  ,0  ,24 ,2  ,9 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+275   ,5  ,1  ,0  ,3  ,0 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+276   ,6  ,0  ,1  ,4  ,1 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+277   ,7  ,1  ,2  ,5  ,2 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+278   ,8  ,0  ,3  ,6  ,3 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+279   ,9  ,1  ,4  ,7  ,4 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+280   ,0  ,0  ,5  ,8  ,5 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+281   ,1  ,1  ,6  ,9  ,6 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+282   ,2  ,0  ,7  ,10 ,7 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+283   ,3  ,1  ,8  ,11 ,8 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+284   ,4  ,0  ,9  ,12 ,9 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+285   ,5  ,1  ,10 ,13 ,0 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+286   ,6  ,0  ,11 ,14 ,1 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+287   ,7  ,1  ,12 ,15 ,2 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+288   ,8  ,0  ,13 ,0  ,3 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+289   ,9  ,1  ,14 ,1  ,4 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+290   ,0  ,0  ,15 ,2  ,5 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+291   ,1  ,1  ,16 ,3  ,6 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+292   ,2  ,0  ,17 ,4  ,7 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+293   ,3  ,1  ,18 ,5  ,8 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+294   ,4  ,0  ,19 ,6  ,9 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+295   ,5  ,1  ,20 ,7  ,0 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+296   ,6  ,0  ,21 ,8  ,1 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+297   ,7  ,1  ,22 ,9  ,2 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+298   ,8  ,0  ,23 ,10 ,3 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+299   ,9  ,1  ,24 ,11 ,4 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+300   ,0  ,0  ,0  ,12 ,5 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+301   ,1  ,1  ,1  ,13 ,6 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+302   ,2  ,0  ,2  ,14 ,7 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+303   ,3  ,1  ,3  ,15 ,8 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+304   ,4  ,0  ,4  ,0  ,9 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+305   ,5  ,1  ,5  ,1  ,0 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+306   ,6  ,0  ,6  ,2  ,1 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+307   ,7  ,1  ,7  ,3  ,2 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+308   ,8  ,0  ,8  ,4  ,3 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+309   ,9  ,1  ,9  ,5  ,4 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+310   ,0  ,0  ,10 ,6  ,5 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+311   ,1  ,1  ,11 ,7  ,6 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+312   ,2  ,0  ,12 ,8  ,7 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+313   ,3  ,1  ,13 ,9  ,8 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+314   ,4  ,0  ,14 ,10 ,9 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+315   ,5  ,1  ,15 ,11 ,0 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+316   ,6  ,0  ,16 ,12 ,1 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+317   ,7  ,1  ,17 ,13 ,2 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+318   ,8  ,0  ,18 ,14 ,3 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+319   ,9  ,1  ,19 ,15 ,4 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+320   ,0  ,0  ,20 ,0  ,5 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+321   ,1  ,1  ,21 ,1  ,6 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+322   ,2  ,0  ,22 ,2  ,7 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+323   ,3  ,1  ,23 ,3  ,8 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+324   ,4  ,0  ,24 ,4  ,9 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+325   ,5  ,1  ,0  ,5  ,0 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+326   ,6  ,0  ,1  ,6  ,1 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+327   ,7  ,1  ,2  ,7  ,2 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+328   ,8  ,0  ,3  ,8  ,3 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+329   ,9  ,1  ,4  ,9  ,4 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+330   ,0  ,0  ,5  ,10 ,5 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+331   ,1  ,1  ,6  ,11 ,6 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+332   ,2  ,0  ,7  ,12 ,7 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+333   ,3  ,1  ,8  ,13 ,8 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+334   ,4  ,0  ,9  ,14 ,9 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+335   ,5  ,1  ,10 ,15 ,0 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+336   ,6  ,0  ,11 ,0  ,1 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+337   ,7  ,1  ,12 ,1  ,2 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+338   ,8  ,0  ,13 ,2  ,3 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+339   ,9  ,1  ,14 ,3  ,4 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+340   ,0  ,0  ,15 ,4  ,5 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+341   ,1  ,1  ,16 ,5  ,6 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+342   ,2  ,0  ,17 ,6  ,7 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+343   ,3  ,1  ,18 ,7  ,8 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+344   ,4  ,0  ,19 ,8  ,9 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+345   ,5  ,1  ,20 ,9  ,0 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+346   ,6  ,0  ,21 ,10 ,1 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+347   ,7  ,1  ,22 ,11 ,2 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+348   ,8  ,0  ,23 ,12 ,3 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+349   ,9  ,1  ,24 ,13 ,4 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+350   ,0  ,0  ,0  ,14 ,5 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+351   ,1  ,1  ,1  ,15 ,6 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+352   ,2  ,0  ,2  ,0  ,7 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+353   ,3  ,1  ,3  ,1  ,8 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+354   ,4  ,0  ,4  ,2  ,9 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+355   ,5  ,1  ,5  ,3  ,0 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+356   ,6  ,0  ,6  ,4  ,1 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+357   ,7  ,1  ,7  ,5  ,2 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+358   ,8  ,0  ,8  ,6  ,3 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+359   ,9  ,1  ,9  ,7  ,4 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+360   ,0  ,0  ,10 ,8  ,5 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+361   ,1  ,1  ,11 ,9  ,6 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+362   ,2  ,0  ,12 ,10 ,7 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+363   ,3  ,1  ,13 ,11 ,8 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+364   ,4  ,0  ,14 ,12 ,9 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+365   ,5  ,1  ,15 ,13 ,0 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+366   ,6  ,0  ,16 ,14 ,1 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+367   ,7  ,1  ,17 ,15 ,2 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+368   ,8  ,0  ,18 ,0  ,3 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+369   ,9  ,1  ,19 ,1  ,4 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+370   ,0  ,0  ,20 ,2  ,5 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+371   ,1  ,1  ,21 ,3  ,6 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+372   ,2  ,0  ,22 ,4  ,7 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+373   ,3  ,1  ,23 ,5  ,8 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+374   ,4  ,0  ,24 ,6  ,9 ,12 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+375   ,5  ,1  ,0  ,7  ,0 ,13 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+376   ,6  ,0  ,1  ,8  ,1 ,14 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+377   ,7  ,1  ,2  ,9  ,2 ,15 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+378   ,8  ,0  ,3  ,10 ,3 ,16 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+379   ,9  ,1  ,4  ,11 ,4 ,17 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+380   ,0  ,0  ,5  ,12 ,5 ,18 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+381   ,1  ,1  ,6  ,13 ,6 ,19 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+382   ,2  ,0  ,7  ,14 ,7 ,20 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+383   ,3  ,1  ,8  ,15 ,8 ,21 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+384   ,4  ,0  ,9  ,0  ,9 ,22 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+385   ,5  ,1  ,10 ,1  ,0 ,23 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+386   ,6  ,0  ,11 ,2  ,1 ,24 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+387   ,7  ,1  ,12 ,3  ,2 ,0  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+388   ,8  ,0  ,13 ,4  ,3 ,1  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+389   ,9  ,1  ,14 ,5  ,4 ,2  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+390   ,0  ,0  ,15 ,6  ,5 ,3  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+391   ,1  ,1  ,16 ,7  ,6 ,4  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+392   ,2  ,0  ,17 ,8  ,7 ,5  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+393   ,3  ,1  ,18 ,9  ,8 ,6  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+394   ,4  ,0  ,19 ,10 ,9 ,7  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+395   ,5  ,1  ,20 ,11 ,0 ,8  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+396   ,6  ,0  ,21 ,12 ,1 ,9  )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+397   ,7  ,1  ,22 ,13 ,2 ,10 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+398   ,8  ,0  ,23 ,14 ,3 ,11 )
        RENAME_UNROLL_MACRO_LP_DERI_HP_PTR_HALF_DIV(index+399   ,9  ,1  ,24 ,15 ,4 ,12 )
        #ifdef OPERATION_COUNTER
        //fabs() counted as 1mult
        float_mul_counter+=400*6;
        float_add_counter+=400*10;
        float_comp_counter+=400*1;
        #endif    
    }
    for(; index <= sampleLength - 50 ; index+=50)
    {
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index       ,0  ,0  ,0  ,5  ,13)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+1     ,1  ,1  ,1  ,6  ,14)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+2     ,2  ,0  ,2  ,7  ,15)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+3     ,3  ,1  ,3  ,8  ,16)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+4     ,4  ,0  ,4  ,9  ,17)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+5     ,5  ,1  ,5  ,0  ,18)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+6     ,6  ,0  ,6  ,1  ,19)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+7     ,7  ,1  ,7  ,2  ,20)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+8     ,8  ,0  ,8  ,3  ,21)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+9     ,9  ,1  ,9  ,4  ,22)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+10    ,0  ,0  ,10 ,5  ,23)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+11    ,1  ,1  ,11 ,6  ,24)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+12    ,2  ,0  ,12 ,7  ,0)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+13    ,3  ,1  ,13 ,8  ,1)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+14    ,4  ,0  ,14 ,9  ,2)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+15    ,5  ,1  ,15 ,0  ,3)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+16    ,6  ,0  ,16 ,1  ,4)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+17    ,7  ,1  ,17 ,2  ,5) 
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+18    ,8  ,0  ,18 ,3  ,6)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+19    ,9  ,1  ,19 ,4  ,7)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+20    ,0  ,0  ,20 ,5  ,8)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+21    ,1  ,1  ,21 ,6  ,9)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+22    ,2  ,0  ,22 ,7  ,10)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+23    ,3  ,1  ,23 ,8  ,11)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+24    ,4  ,0  ,24 ,9  ,12)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+25    ,5  ,1  ,0  ,0  ,13)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+26    ,6  ,0  ,1  ,1  ,14)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+27    ,7  ,1  ,2  ,2  ,15) 
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+28    ,8  ,0  ,3  ,3  ,16)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+29    ,9  ,1  ,4  ,4  ,17)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+30    ,0  ,0  ,5  ,5  ,18)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+31    ,1  ,1  ,6  ,6  ,19)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+32    ,2  ,0  ,7  ,7  ,20)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+33    ,3  ,1  ,8  ,8  ,21)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+34    ,4  ,0  ,9  ,9  ,22)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+35    ,5  ,1  ,10 ,0  ,23)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+36    ,6  ,0  ,11 ,1  ,24)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+37    ,7  ,1  ,12 ,2  ,0)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+38    ,8  ,0  ,13 ,3  ,1)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+39    ,9  ,1  ,14 ,4  ,2)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+40    ,0  ,0  ,15 ,5  ,3)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+41    ,1  ,1  ,16 ,6  ,4)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+42    ,2  ,0  ,17 ,7  ,5)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+43    ,3  ,1  ,18 ,8  ,6)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+44    ,4  ,0  ,19 ,9  ,7) 
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+45    ,5  ,1  ,20 ,0  ,8)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+46    ,6  ,0  ,21 ,1  ,9)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+47    ,7  ,1  ,22 ,2  ,10)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+48    ,8  ,0  ,23 ,3  ,11)
        RENAME_UNROLL_MACRO_LP_DERI_HP_HALF_DEPENDENCIES_DIV(index+49    ,9  ,1  ,24 ,4  ,12)
        #ifdef OPERATION_COUNTER
        //fabs() counted as 1mult
        float_mul_counter+=50*6;
        float_add_counter+=50*10;
        float_comp_counter+=50*1;
        #endif
    }

    static int lpbuffer_lgth_half = (LPBUFFER_LGTH/2);
    static int hpbuffer_lgth_half = (HPBUFFER_LGTH/2);
    static int ptr_array[4] = {0}; // lp_ptr = 0, hp_ptr = 0, derI = 0, ptr = 0;
    __m128i maxMask = _mm_set_epi32(WINDOW_WIDTH, DERIV_LENGTH, HPBUFFER_LGTH, LPBUFFER_LGTH);
    __m128i oneVec = _mm_set_epi32(1, 1, 1, 1);

    for(index; index<sampleLength; index++)
    {
        halfPtr = ptr_array[0] - lpbuffer_lgth_half ;    // Use halfPtr to index
        if(halfPtr < 0)                         // to x[n-6].
            halfPtr += LPBUFFER_LGTH ;

        y0 = (y1*2.0f) - y2 + datum[index] - (lp_data[halfPtr]*2.0f) + lp_data[ptr_array[0]] ;
        y2 = y1;
        y1 = y0;
        fdatum = y0 * lpbuffer_sqr_div_4;
        lp_data[ptr_array[0]] = datum[index] ;            // Stick most recent sample into
        
        hp_y += fdatum - hp_data[ptr_array[1]];
        halfPtr = ptr_array[1] - hpbuffer_lgth_half ;
        if(halfPtr < 0)
            halfPtr += HPBUFFER_LGTH ;
        hp_data[ptr_array[1]] = fdatum ;
        fdatum = hp_data[halfPtr] - (hp_y * HPBUFFER_LGTH_INV);
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
        if((sum * WINDOW_WIDTH_INV) > 32000.f)
        {
            filtOutput[index] = 32000.f ;
        } 
        else 
        {
            filtOutput[index] = sum * WINDOW_WIDTH_INV ;
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
    }

}

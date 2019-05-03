#include <stdio.h>

#define SAMPLE_RATE 200
#define MS_PER_SAMPLE ((1000/SAMPLE_RATE))

int main(){
  printf("MS10 %d\n", ((int) (10/ MS_PER_SAMPLE + 0.5)));
  printf("MS125 %d\n", ((int) (125/MS_PER_SAMPLE + 0.5)));
  printf("MS25 %d\n", ((int) (25/MS_PER_SAMPLE + 0.5)));
  printf("MS80 %d\n", ((int) (80/MS_PER_SAMPLE + 0.5)));


  
  return 0;
}

#include "tsc_x86.h"

void init_tsc() {
		; // no need to initialize anything for x86
	}

	myInt64 start_tsc(void) {
	    tsc_counter start;
	    CPUID();
	    RDTSC(start);
	    return COUNTER_VAL(start);
	}

	myInt64 stop_tsc(myInt64 start) {
		tsc_counter end;
		RDTSC(end);
		CPUID();
		return COUNTER_VAL(end) - start;
	}
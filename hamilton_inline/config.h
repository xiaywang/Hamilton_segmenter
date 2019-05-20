#ifndef __CONFIG_H__
#define __CONFIG_H__
#include "performance.h"
// how many sambles get passed to the beatDetectAndClassify and subsequent function
// 7200 should be devisible by it, so it runs most smoothly
//#define MAIN_BLOCK_SIZE 1000

// define all version numbers
#define QRSFILT_OPT 1
#define INIT_INLINE 1
#define BLOCKING_SIZE_QRSFILT 5

#define BDAC_OPT 1

#define AVX_OPT 1

#define NOISECHK_OPT 1

#endif

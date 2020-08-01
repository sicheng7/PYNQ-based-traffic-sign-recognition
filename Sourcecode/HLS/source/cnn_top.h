#ifndef __CNN_TOP_H__
#define __CNN_TOP_H__



#include "lib.h"
#include <cmath>
#include "conv.h"
#include "pool.h"
#include "operate.h"
#include "model.h"
#include "parameter.h"

// Uncomment this line to compare TB vs HW C-model and/or RTL
//#define HW_COSIM

//#define MAT_A_ROWS 3
//#define MAT_A_COLS 3
//#define MAT_B_ROWS 3
//#define MAT_B_COLS 3

//typedef char mat_a_t;
//typedef char mat_b_t;
//typedef short result_t;

// Prototype of top level function for C-synthesis
void top_cnn();

#endif

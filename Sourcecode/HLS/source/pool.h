#ifndef _POOL_H_
#define _POOL_H_



#include "lib.h"
#include "parameter.h"

float MAX_float(float a, float b, float c, float d);
BinarySum MAX_BinarySum(BinarySum a, BinarySum b, BinarySum c, BinarySum d);

void maxpool1(
		float In[CH_Layer][Size_CONV1_out][Size_CONV1_out],
		float Out[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out],
		int S);
void maxpool2(
		BinarySum In[CH_Layer][Size_CONV2_out][Size_CONV2_out],
		BinarySum Out[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out],
		int S);
void maxpool3(
		BinarySum In[CH_Layer][Size_CONV3_out][Size_CONV3_out],
		BinarySum Out[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out],
		int S);

#endif

#ifndef _CONV_H_
#define _CONV_H_
#include "stdlib.h"
#include "memory.h"
#include "lib.h"
#include "parameter.h"


void conv1(
		float In[CH_In][Size_In][Size_In],
		float Weight[CH_Layer][CH_In][Conv_K1][Conv_K1],
		float Out[CH_Layer][Size_CONV1_out][Size_CONV1_out]);

void conv2(
		Binary In[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out],
		Binary Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1],
		BinarySum Out[CH_Layer][Size_CONV2_out][Size_CONV2_out]);


void conv3(
		Binary In[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out],
		Binary Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1],
		BinarySum Out[CH_Layer][Size_CONV3_out][Size_CONV3_out]);


void conv4(
		Binary In[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out],
		Binary Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1],
		BinarySum Out[CH_Layer][Size_Classif_out][Size_Classif_out]);

void conv5(
		Binary In[CH_Layer][Size_Classif_out][Size_Classif_out],
		Binary Weight[CH_Layer][CH_Layer][Conv_K2][Conv_K2],
		Binary Out[CH_Layer][Size_Classif_out][Size_Classif_out]);

void conv6(
		Binary In[CH_Layer][Size_Classif_out][Size_Classif_out],
		float Weight[CH_Layer][CH_Layer][Conv_K2][Conv_K2],
		float Out[CH_Out][Size_Classif_out][Size_Classif_out]);

#endif

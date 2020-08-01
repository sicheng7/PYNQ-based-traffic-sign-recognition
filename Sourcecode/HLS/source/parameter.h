#ifndef _PARAMETER_H_
#define _PARAMETER_H_

#include "lib.h"

#define CH_In       3
#define CH_Layer    128
#define CH_Out	    58

#define Size_In		32

#define Conv_K1		3
#define Conv_K2		1

#define MAXPOOL_S1  2
#define MAXPOOL_S2  1

#define Size_CONV1_out 		30
#define Size_MAXPOOL1_out 	15
#define Size_CONV2_out		13
#define Size_MAXPOOL2_out	6
#define Size_CONV3_out      4
#define Size_MAXPOOL3_out	3
#define Size_Classif_out	1

extern float DATA_IN[CH_In][Size_In][Size_In];
extern float DATA_CONV1_Out[CH_Layer][Size_CONV1_out][Size_CONV1_out];
extern float DATA_MAXPOOL1_Out[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out];
extern Binary DATA_MAXPOOL1_OutB[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out];

extern BinarySum DATA_CONV2_Out[CH_Layer][Size_CONV2_out][Size_CONV2_out];
extern BinarySum DATA_MAXPOOL2_Out[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out];
extern Binary DATA_MAXPOOL2_OutB[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out];

extern BinarySum DATA_CONV3_Out[CH_Layer][Size_CONV3_out][Size_CONV3_out];
extern BinarySum DATA_MAXPOOL3_Out[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out];
extern Binary DATA_MAXPOOL3_OutB[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out];

extern BinarySum DATA_CONV4_Out[CH_Layer][Size_Classif_out][Size_Classif_out];
extern Binary DATA_CONV4_OutB[CH_Layer][Size_Classif_out][Size_Classif_out];
extern Binary DATA_CONV5_Out[CH_Layer][Size_Classif_out][Size_Classif_out];
extern Binary DATA_Relu_Out[CH_Layer][Size_Classif_out][Size_Classif_out];
extern float DATA_CONV6_Out[CH_Out][Size_Classif_out][Size_Classif_out];

extern float CONV1_Weight[CH_Layer][CH_In][Conv_K1][Conv_K1];
extern Binary CONV2_Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1];
extern Binary CONV3_Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1];
extern Binary CONV4_Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1];
extern Binary CONV5_Weight[CH_Layer][CH_Layer][Conv_K2][Conv_K2];
extern float CONV6_Weight[CH_Out][CH_Layer][Conv_K2][Conv_K2];


extern int label;

#endif

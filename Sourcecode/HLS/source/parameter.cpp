#include "parameter.h"

float DATA_IN[CH_In][Size_In][Size_In];
float DATA_CONV1_Out[CH_Layer][Size_CONV1_out][Size_CONV1_out];
float DATA_MAXPOOL1_Out[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out];
Binary DATA_MAXPOOL1_OutB[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out];

BinarySum DATA_CONV2_Out[CH_Layer][Size_CONV2_out][Size_CONV2_out];
BinarySum DATA_MAXPOOL2_Out[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out];
Binary DATA_MAXPOOL2_OutB[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out];

BinarySum DATA_CONV3_Out[CH_Layer][Size_CONV3_out][Size_CONV3_out];
BinarySum DATA_MAXPOOL3_Out[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out];
Binary DATA_MAXPOOL3_OutB[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out];

BinarySum DATA_CONV4_Out[CH_Layer][Size_Classif_out][Size_Classif_out];
Binary DATA_CONV4_OutB[CH_Layer][Size_Classif_out][Size_Classif_out];
Binary DATA_CONV5_Out[CH_Layer][Size_Classif_out][Size_Classif_out];
Binary DATA_Relu_Out[CH_Layer][Size_Classif_out][Size_Classif_out];
float DATA_CONV6_Out[CH_Out][Size_Classif_out][Size_Classif_out];

float CONV1_Weight[CH_Layer][CH_In][Conv_K1][Conv_K1];
Binary CONV2_Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1];
Binary CONV3_Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1];
Binary CONV4_Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1];
Binary CONV5_Weight[CH_Layer][CH_Layer][Conv_K2][Conv_K2];
float CONV6_Weight[CH_Out][CH_Layer][Conv_K2][Conv_K2];

int label;

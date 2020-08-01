#ifndef _OPERATE_H_
#define _OPERATE_H_

#include "lib.h"

//���ڵ������룬����������Ƚϵó�����Ȳ���
#include "operate.h"
#include <stdio.h>
#include <fstream>
#include "parameter.h"

#define Rin 64
#define Cin 64

//��������
void Load_In(float* In_ddr, float In[4][Rin][Cin]);

void GetLabel(float In[CH_Out][Size_Classif_out][Size_Classif_out]);

void BinaryOperate1(
		float In[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out],
		Binary Out[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out]);

void BinaryOperate2(
		BinarySum In[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out],
		Binary Out[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out]);


void BinaryOperate3(
		BinarySum In[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out],
		Binary Out[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out]);

void BinaryOperate4(
		BinarySum In[CH_Layer][Size_Classif_out][Size_Classif_out],
		Binary Out[CH_Layer][Size_Classif_out][Size_Classif_out]);


#endif

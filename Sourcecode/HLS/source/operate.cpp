//���ڵ������룬����������Ƚϵó�����Ȳ���
#include "operate.h"
#include <stdio.h>
#include <fstream>
#include "parameter.h"

#define Rin 64
#define Cin 64

//��������
void Load_In(float* In_ddr, float In[4][Rin][Cin])
{
	for(int L_ri=0; L_ri<Rin; L_ri++)
	{
#pragma HLS PIPELINE
		for(int L_ci=0; L_ci<Cin; L_ci++)
		{
			for(int L_chi=0; L_chi<4; L_chi++)
			{
				In[L_chi][L_ri][L_ci] = *In_ddr++;
			}
		}
	}
	return;
}

//��ñ�ǩ
void GetLabel(float In[CH_Out][Size_Classif_out][Size_Classif_out])
{
	float maxData = 0;
	label = 0;
	maxData = In[0][Size_Classif_out-1][Size_Classif_out-1];
	for (int i = 1; i < CH_Out; i++)
	{
		if(maxData > In[i][Size_Classif_out-1][Size_Classif_out-1]){}
		else
		{
			maxData = In[i][Size_Classif_out-1][Size_Classif_out-1];
			label = i;
		}
	}
}

void BinaryOperate1(
		float In[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out],
		Binary Out[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out])
{
	for (int ch=0; ch < CH_Layer; ch++)
	{
		for (int i=0; i < Size_MAXPOOL1_out; i++)
		{
			for (int j=0; j < Size_MAXPOOL1_out; j++)
			{
				if(In[ch][i][j] > 0)
					Out[ch][i][j] = 1;
				else if(In[ch][i][j] == 0)
					Out[ch][i][j] = 0;
				else
					Out[ch][i][j] = -1;
			}
		}
	}
}

void BinaryOperate2(
		BinarySum In[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out],
		Binary Out[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out])
{
	for (int ch=0; ch < CH_Layer; ch++)
	{
		for (int i=0; i < Size_MAXPOOL2_out; i++)
		{
			for (int j=0; j < Size_MAXPOOL2_out; j++)
			{
				if(In[ch][i][j] > 0)
					Out[ch][i][j] = 1;
				else if(In[ch][i][j] == 0)
					Out[ch][i][j] = 0;
				else
					Out[ch][i][j] = -1;
			}
		}
	}
}


void BinaryOperate3(
		BinarySum In[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out],
		Binary Out[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out])
{
	for (int ch=0; ch < CH_Layer; ch++)
	{
		for (int i=0; i < Size_MAXPOOL3_out; i++)
		{
			for (int j=0; j < Size_MAXPOOL3_out; j++)
			{
				if(In[ch][i][j] > 0)
					Out[ch][i][j] = 1;
				else if(In[ch][i][j] == 0)
					Out[ch][i][j] = 0;
				else
					Out[ch][i][j] = -1;
			}
		}
	}
}

void BinaryOperate4(
		BinarySum In[CH_Layer][Size_Classif_out][Size_Classif_out],
		Binary Out[CH_Layer][Size_Classif_out][Size_Classif_out])
{
	for (int ch=0; ch < CH_Layer; ch++)
	{
		for (int i=0; i < Size_Classif_out; i++)
		{
			for (int j=0; j < Size_Classif_out; j++)
			{
				if(In[ch][i][j] > 0)
					Out[ch][i][j] = 1;
				else if(In[ch][i][j] == 0)
					Out[ch][i][j] = 0;
				else
					Out[ch][i][j] = -1;
			}
		}
	}
}


//void rw()
//{
////	//������д���ݣ�������0~9д�뵽data.txt�ļ���
////	FILE *fpWrite=fopen("a.txt","w");
////	if(fpWrite==NULL)
////	{
////	}
////	for(int i=0;i<10;i++)
////		fprintf(fpWrite,"%d ",i);
////	fclose(fpWrite);
////	//�����Ƕ����ݣ������������ݴ浽����a[10]�У����Ҵ�ӡ������̨��
////	int a[10]={0};
////	FILE *fpRead=fopen("a.txt","r");
////	if(fpRead==NULL)
////	{
////	}
////	for(int i=0;i<10;i++)
////	{
////		fscanf(fpRead,"%d ",&a[i]);
////		printf("%d ",a[i]);
////	}
//}

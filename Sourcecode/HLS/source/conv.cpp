#include "conv.h"



//	;
//#pragma HLS array_partition variable=In complete dim=1
//
//#pragma HLS array_partition variable=Out complete dim=1
//	;
//#pragma HLS array_partition variable=W complete dim=1
//#pragma HLS array_partition variable=W complete dim=2

void conv1(
		float In[CH_In][Size_In][Size_In],
		float Weight[CH_Layer][CH_In][Conv_K1][Conv_K1],
		float Out[CH_Layer][Size_CONV1_out][Size_CONV1_out])
{
	Kernel_Row:
	for(int kr=0; kr<Conv_K1; kr++)
	{
		Kernel_Column:
		for(int kc=0; kc<Conv_K1; kc++)
		{
			Row:
			for(int r=0; r<Size_CONV1_out; r++)
			{
				Column:
				for(int c=0; c<Size_CONV1_out; c++)
				{
#pragma HLS PIPELINE
					Output_Channel:
					for(int cho=0; cho<CH_Layer; cho++)
					{
						Input_Channel:
						for(int chi=0; chi<CH_In; chi++)
						{
							Out[cho][r][c] += In[chi][r+kr][c+kc] * Weight[cho][chi][kr][kc];
						}
					}
				}
			}
		}
	}
}

void conv2(
		Binary In[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out],
		Binary Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1],
		BinarySum Out[CH_Layer][Size_CONV2_out][Size_CONV2_out])
{
	Kernel_Row:
	for(int kr=0; kr<Conv_K1; kr++)
	{
		Kernel_Column:
		for(int kc=0; kc<Conv_K1; kc++)
		{
			Row:
			for(int r=0; r<Size_CONV2_out; r++)
			{
				Column:
				for(int c=0; c<Size_CONV2_out; c++)
				{
#pragma HLS PIPELINE
					Output_Channel:
					for(int cho=0; cho<CH_Layer; cho++)
					{
						Input_Channel:
						for(int chi=0; chi<CH_Layer; chi++)
						{
							Out[cho][r][c] += In[chi][r+kr][c+kc] * Weight[cho][chi][kr][kc];
						}
					}
				}
			}
		}
	}
}

void conv3(
		Binary In[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out],
		Binary Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1],
		BinarySum Out[CH_Layer][Size_CONV3_out][Size_CONV3_out])
{
	Kernel_Row:
	for(int kr=0; kr<Conv_K1; kr++)
	{
		Kernel_Column:
		for(int kc=0; kc<Conv_K1; kc++)
		{
			Row:
			for(int r=0; r<Size_CONV3_out; r++)
			{
				Column:
				for(int c=0; c<Size_CONV3_out; c++)
				{
#pragma HLS PIPELINE
					Output_Channel:
					for(int cho=0; cho<CH_Layer; cho++)
					{
						Input_Channel:
						for(int chi=0; chi<CH_Layer; chi++)
						{
							Out[cho][r][c] += In[chi][r+kr][c+kc] * Weight[cho][chi][kr][kc];
						}
					}
				}
			}
		}
	}
}

void conv4(
		Binary In[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out],
		Binary Weight[CH_Layer][CH_Layer][Conv_K1][Conv_K1],
		BinarySum Out[CH_Layer][Size_Classif_out][Size_Classif_out])
{
	Kernel_Row:
	for(int kr=0; kr<Conv_K1; kr++)
	{
		Kernel_Column:
		for(int kc=0; kc<Conv_K1; kc++)
		{
			Row:
			for(int r=0; r<Size_Classif_out; r++)
			{
				Column:
				for(int c=0; c<Size_Classif_out; c++)
				{
#pragma HLS PIPELINE
					Output_Channel:
					for(int cho=0; cho<CH_Layer; cho++)
					{
						Input_Channel:
						for(int chi=0; chi<CH_Layer; chi++)
						{
							Out[cho][r][c] += In[chi][r+kr][c+kc] * Weight[cho][chi][kr][kc];
						}
					}
				}
			}
		}
	}
}

void conv5(
		Binary In[CH_Layer][Size_Classif_out][Size_Classif_out],
		Binary Weight[CH_Layer][CH_Layer][Conv_K2][Conv_K2],
		Binary Out[CH_Layer][Size_Classif_out][Size_Classif_out])
{
	Kernel_Row:
	for(int kr=0; kr<Conv_K2; kr++)
	{
		Kernel_Column:
		for(int kc=0; kc<Conv_K2; kc++)
		{
			Row:
			for(int r=0; r<Size_Classif_out; r++)
			{
				Column:
				for(int c=0; c<Size_Classif_out; c++)
				{
#pragma HLS PIPELINE
					Output_Channel:
					for(int cho=0; cho<CH_Layer; cho++)
					{
						Input_Channel:
						for(int chi=0; chi<CH_Layer; chi++)
						{
							Out[cho][r][c] += In[chi][r+kr][c+kc] * Weight[cho][chi][kr][kc];
						}
					}
				}
			}
		}
	}
}


void conv6(
		Binary In[CH_Layer][Size_Classif_out][Size_Classif_out],
		float Weight[CH_Layer][CH_Layer][Conv_K2][Conv_K2],
		float Out[CH_Out][Size_Classif_out][Size_Classif_out])
{
	Kernel_Row:
	for(int kr=0; kr<Conv_K2; kr++)
	{
		Kernel_Column:
		for(int kc=0; kc<Conv_K2; kc++)
		{
			Row:
			for(int r=0; r<Size_Classif_out; r++)
			{
				Column:
				for(int c=0; c<Size_Classif_out; c++)
				{
#pragma HLS PIPELINE
					Output_Channel:
					for(int cho=0; cho<CH_Out; cho++)
					{
						Input_Channel:
						for(int chi=0; chi<CH_Layer; chi++)
						{
							Out[cho][r][c] += In[chi][r+kr][c+kc] * Weight[cho][chi][kr][kc];
						}
					}
				}
			}
		}
	}
}

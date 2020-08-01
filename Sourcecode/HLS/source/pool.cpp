#include "pool.h"



float MAX_float(float a, float b, float c, float d)
{
#pragma HLS INLINE
	float t1 = a > b ? a : b;
	float t2 = c > d ? c : d;
	return t1 > t2 ? t1 : t2;
}

BinarySum MAX_BinarySum(BinarySum a, BinarySum b, BinarySum c, BinarySum d)
{
#pragma HLS INLINE
	BinarySum t1 = a > b ? a : b;
	BinarySum t2 = c > d ? c : d;
	return t1 > t2 ? t1 : t2;
}

void maxpool1(
		float In[CH_Layer][Size_CONV1_out][Size_CONV1_out],
		float Out[CH_Layer][Size_MAXPOOL1_out][Size_MAXPOOL1_out],
		int S)
{
	int out_w = 0;
	int out_h = 0;
	for (int ch= 0; ch < CH_Layer; ch ++)
	{
		out_w = 0;
		for (int w=0; (w < Size_CONV1_out && (w+S) <= Size_CONV1_out); w+= S)
		{
			out_h = 0;
			for (int h=0; (h < Size_CONV1_out && (h+S) <= Size_CONV1_out); h+= S)
			{
				Out[ch][out_w][out_h] = MAX_float(In[ch][w][h], In[ch][w+1][h], In[ch][w][h+1], In[ch][w+1][h+1]);
				out_h ++;
			}
			out_w ++;
		}
	}
}

void maxpool2(
		BinarySum In[CH_Layer][Size_CONV2_out][Size_CONV2_out],
		BinarySum Out[CH_Layer][Size_MAXPOOL2_out][Size_MAXPOOL2_out],
		int S)
{
	int out_w = 0;
	int out_h = 0;
	for (int ch= 0; ch < CH_Layer; ch ++)
	{
		out_w = 0;
		for (int w=0; (w < Size_CONV2_out && (w+S) <= Size_CONV2_out); w+= S)
		{
			out_h = 0;
			for (int h=0; (h < Size_CONV2_out && (h+S) < Size_CONV2_out); h+= S)
			{
				Out[ch][out_w][out_h] = MAX_BinarySum(In[ch][w][h], In[ch][w+1][h], In[ch][w][h+1], In[ch][w+1][h+1]);
				out_h ++;
			}
			out_w ++;
		}
	}
}

void maxpool3(
		BinarySum In[CH_Layer][Size_CONV3_out][Size_CONV3_out],
		BinarySum Out[CH_Layer][Size_MAXPOOL3_out][Size_MAXPOOL3_out],
		int S)
{
	int out_w = 0;
	int out_h = 0;
	for (int ch= 0; ch < CH_Layer; ch ++)
	{
		out_w = 0;
		for (int w=0; (w < Size_CONV3_out && (w+S) < Size_CONV3_out); w+= S)
		{
			out_h = 0;
			for (int h=0; (h < Size_CONV3_out && (h+S) < Size_CONV3_out); h+= S)
			{
				Out[ch][out_w][out_h] = MAX_BinarySum(In[ch][w][h], In[ch][w+1][h], In[ch][w][h+1], In[ch][w+1][h+1]);
				out_h ++;
			}
			out_w ++;
		}
	}
}

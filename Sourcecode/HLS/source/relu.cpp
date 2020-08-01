#include "relu.h"


Binary relu_sigData(Binary data)
{
#pragma HLS INLINE
	return data > Binary(0) ? data : Binary(0);
}


void relu(
		Binary In[CH_Layer][Size_Classif_out][Size_Classif_out],
		Binary Out[CH_Layer][Size_Classif_out][Size_Classif_out])
{
	for (int ch= 0; ch < CH_Layer; ch ++)
	{
		for (int w=0; w < Size_Classif_out; w++)
		{
			for (int h=0; h < Size_Classif_out; h++)
			{
				Out[ch][w][h] = relu_sigData(In[ch][w][h]);
			}
		}
	}
}

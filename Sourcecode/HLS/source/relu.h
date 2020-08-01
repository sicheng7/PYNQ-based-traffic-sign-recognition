#ifndef _RELU_H_
#define _RELU_H_

#include "lib.h"
#include "parameter.h"

Binary relu_sigData(Binary data);


void relu(
		Binary In[CH_Layer][Size_Classif_out][Size_Classif_out],
		Binary Out[CH_Layer][Size_Classif_out][Size_Classif_out]);

#endif

#include "model.h"


void model()
{
	conv1(
			DATA_IN,
			CONV1_Weight,
			DATA_CONV1_Out);
	maxpool1(
			DATA_CONV1_Out,
			DATA_MAXPOOL1_Out,
			MAXPOOL_S1);
	BinaryOperate1(
			DATA_MAXPOOL1_Out,
			DATA_MAXPOOL1_OutB);
	conv2(
			DATA_MAXPOOL1_OutB,
			CONV2_Weight,
			DATA_CONV2_Out);
	maxpool2(
			DATA_CONV2_Out,
			DATA_MAXPOOL2_Out,
			MAXPOOL_S1);
	BinaryOperate2(
			DATA_MAXPOOL2_Out,
			DATA_MAXPOOL2_OutB);
	conv3(
			DATA_MAXPOOL2_OutB,
			CONV3_Weight,
			DATA_CONV3_Out);
	maxpool3(
			DATA_CONV3_Out,
			DATA_MAXPOOL3_Out,
			MAXPOOL_S2);
	BinaryOperate3(
			DATA_MAXPOOL3_Out,
			DATA_MAXPOOL3_OutB);
	conv4(
			DATA_MAXPOOL3_OutB,
			CONV4_Weight,
			DATA_CONV4_Out);
	BinaryOperate4(
			DATA_CONV4_Out,
			DATA_CONV4_OutB);
	conv5(
			DATA_CONV4_OutB,
			CONV5_Weight,
			DATA_CONV5_Out);
	relu(
			DATA_CONV5_Out,
			DATA_Relu_Out);
	conv6(
			DATA_Relu_Out,
			CONV6_Weight,
			DATA_CONV6_Out);
	GetLabel(
			DATA_CONV6_Out);

}


#include <iostream>
#include "cnn_top.h"
#include <hls_opencv.h>

using namespace std;

#define INPUT_IMG "C:/Users/Administrator/Desktop/XilinxSummer/Pytorch_classification/TSRD-Test/057_0001_j.png"


#define WEIGHT_1 "C:/Users/Administrator/Desktop/XilinxSummer/Pytorch_classification/features.0.weight.txt"
#define WEIGHT_2 "C:/Users/Administrator/Desktop/XilinxSummer/Pytorch_classification/features.2.weight.txt"
#define WEIGHT_3 "C:/Users/Administrator/Desktop/XilinxSummer/Pytorch_classification/features.4.weight.txt"
#define WEIGHT_4 "C:/Users/Administrator/Desktop/XilinxSummer/Pytorch_classification/classifier.0.weight.txt"
#define WEIGHT_5 "C:/Users/Administrator/Desktop/XilinxSummer/Pytorch_classification/classifier.1.weight.txt"
#define WEIGHT_6 "C:/Users/Administrator/Desktop/XilinxSummer/Pytorch_classification/classifier.3.weight.txt"

using namespace cv;

void load_in(Mat Img)
{
	for (int w=0; w < Size_In; w++)
	{
		for (int h=0; h < Size_In; h++)
		{
			for (int ch=0; ch < CH_In; ch++)
			{
				DATA_IN[ch][w][h] = (Img.at<Vec3b>(w, h)[ch] / 255.0);
			}
		}
	}
}


void load_conv1()
{
	FILE *fpRead=fopen(WEIGHT_1,"r");
	if(fpRead==NULL)
	{
	}
    for (int N=0; N < CH_Layer; N++)
    {
    	for (int ch=0; ch < CH_In; ch++)
    	{
    		for (int w=0; w < Conv_K1; w++)
    		{
    			for (int h=0; h< Conv_K1; h++)
    			{
    				fscanf(fpRead,"%f ",&CONV1_Weight[N][ch][w][h]);
    			}
    		}
    	}
    }
    cout << CONV1_Weight[CH_Layer-1][CH_In-1][Conv_K1-1][Conv_K1-1] << endl;
}

void load_conv2()
{
	float a;
	FILE *fpRead=fopen(WEIGHT_2,"r");
	if(fpRead==NULL)
	{
	}
    for (int N=0; N < CH_Layer; N++)
    {
    	for (int ch=0; ch < CH_Layer; ch++)
    	{
    		for (int w=0; w < Conv_K1; w++)
    		{
    			for (int h=0; h< Conv_K1; h++)
    			{
    				fscanf(fpRead,"%f ",&a);
    				//cout << a << endl;
    				CONV2_Weight[N][ch][w][h] = Binary(a);
    			}
    		}
    	}
    }
    cout << CONV2_Weight[CH_Layer-1][CH_Layer-1][Conv_K1-1][Conv_K1-1] << endl;
}

void load_conv3()
{
	float a;
	FILE *fpRead=fopen(WEIGHT_3,"r");
	if(fpRead==NULL)
	{
	}
    for (int N=0; N < CH_Layer; N++)
    {
    	for (int ch=0; ch < CH_Layer; ch++)
    	{
    		for (int w=0; w < Conv_K1; w++)
    		{
    			for (int h=0; h< Conv_K1; h++)
    			{
    				fscanf(fpRead,"%f ",&a);
    				//cout << a << endl;
    				CONV3_Weight[N][ch][w][h] = Binary(a);
    			}
    		}
    	}
    }
    cout << CONV3_Weight[CH_Layer-1][CH_Layer-1][Conv_K1-1][Conv_K1-1] << endl;
}

void load_conv4()
{
	float a;
	FILE *fpRead=fopen(WEIGHT_4,"r");
	if(fpRead==NULL)
	{
	}
    for (int N=0; N < CH_Layer; N++)
    {
    	for (int ch=0; ch < CH_Layer; ch++)
    	{
    		for (int w=0; w < Conv_K1; w++)
    		{
    			for (int h=0; h< Conv_K1; h++)
    			{
    				fscanf(fpRead,"%f ",&a);
    				//cout << a << endl;
    				CONV4_Weight[N][ch][w][h] = Binary(a);
    			}
    		}
    	}
    }
    cout << CONV4_Weight[CH_Layer-1][CH_Layer-1][Conv_K1-1][Conv_K1-1] << endl;
}

void load_conv5()
{
	float a;
	FILE *fpRead=fopen(WEIGHT_5,"r");
	if(fpRead==NULL)
	{
	}
    for (int N=0; N < CH_Layer; N++)
    {
    	for (int ch=0; ch < CH_Layer; ch++)
    	{
    		for (int w=0; w < Conv_K2; w++)
    		{
    			for (int h=0; h< Conv_K2; h++)
    			{
    				fscanf(fpRead,"%f ",&a);
    				//cout << a << endl;
    				CONV5_Weight[N][ch][w][h] = Binary(a);
    			}
    		}
    	}
    }
    cout << CONV5_Weight[CH_Layer-1][CH_Layer-1][Conv_K2-1][Conv_K2-1] << endl;
}

void load_conv6()
{
	FILE *fpRead=fopen(WEIGHT_6,"r");
	if(fpRead==NULL)
	{
	}
    for (int N=0; N < CH_Out; N++)
    {
    	for (int ch=0; ch < CH_Layer; ch++)
    	{
    		for (int w=0; w < Conv_K2; w++)
    		{
    			for (int h=0; h< Conv_K2; h++)
    			{
    				fscanf(fpRead,"%f ",&CONV6_Weight[N][ch][w][h]);
    			}
    		}
    	}
    }
    cout << CONV6_Weight[CH_Out-1][CH_Layer-1][Conv_K2-1][Conv_K2-1] << endl;
}

int main()//int argc, char **argv)
{
	Mat imgSrc;
	imgSrc = imread(INPUT_IMG);//, CV_32FC2);
	printf("r %d, c %d, d %d\n",imgSrc.rows,imgSrc.cols,imgSrc);
	Mat ResImg;
	resize(imgSrc, ResImg, cvSize(32, 32),INTER_AREA);
	printf("r %d, c %d, d %d\n",ResImg.rows,ResImg.cols,ResImg.dims);
	//cout << ResImg << endl;
	cout << float(ResImg.at<Vec3b>(1, 3)[0]) << endl;
	load_in(ResImg);
	cout << DATA_IN[0][1][3] << endl;
	load_conv1();
	load_conv2();
	load_conv3();
	load_conv4();
	load_conv5();
	load_conv6();
	top_cnn();
	cout << "result" << endl;
//	for (int result; result < 58; result++)
//	{
//		cout << DATA_CONV6_Out[result][0][0] << endl;
//	}

	cout << label << endl;
}


//2

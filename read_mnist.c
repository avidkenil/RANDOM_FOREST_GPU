#include <stdio.h>
#include <stdlib.h>
#include "MnistPreProcess.h"

#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define FEATURE 784
#define NUMBER_OF_CLASSES 10



void readData(float* dataset,float*labels,const char* dataPath,const char*labelPath)
{
	FILE* dataFile=fopen(dataPath,"rb");
	FILE* labelFile=fopen(labelPath,"rb");
	int mbs=0,number=0,col=0,row=0;
	fread(&mbs,4,1,dataFile);
	fread(&number,4,1,dataFile);
	fread(&row,4,1,dataFile);
	fread(&col,4,1,dataFile);
	revertInt(&mbs);
	revertInt(&number);
	revertInt(&row);
	revertInt(&col);
	fread(&mbs,4,1,labelFile);
	fread(&number,4,1,labelFile);
	revertInt(&mbs);
	revertInt(&number);
	unsigned char temp;
	for(int i=0;i<number;++i)
	{
		for(int j=0;j<row*col;++j)
		{
			fread(&temp,1,1,dataFile);
			//dataset[i][j]=static_cast<float>(temp);
			dataset[(i*row*col) + j] = (float)temp;
		}
		fread(&temp,1,1,labelFile);
		//printf("%s\n",*temp );
		//labels[i]=static_cast<float>(temp);
		labels[i] = (float)temp;
		//printf("%f\n", labels[i]);
	}
	fclose(dataFile);
	fclose(labelFile);
};

int main(int argc, char * argv[])
{
	float *dataset_train,*dataset_test;
	float *labels_train,*labels_test;
	dataset_train = (float *)malloc(FEATURE * TRAIN_NUM*sizeof(float));
	labels_train = (float *)malloc(TRAIN_NUM*sizeof(float));
	dataset_test = (float *)malloc(FEATURE * TEST_NUM*sizeof(float));
	labels_test = (float *)malloc(TEST_NUM*sizeof(float));

	char file_train_set[] = "data/train-images-idx3-ubyte";
	char file_train_label[] = "data/train-labels-idx1-ubyte";
	char file_test_set[] = "data/t10k-images-idx3-ubyte";
	char file_test_label[] = "data/t10k-labels-idx1-ubyte";
	readData(dataset_train,labels_train,file_train_set,file_train_label);
	//printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n Testttttt ******************************");
	readData(dataset_test,labels_test,file_test_set,file_test_label);
	//for(int i=0;i<FEATURE;i++)
//		printf("%f\n",dataset_train[i] );
//	printf("Feature 1%f\n",labels_train[0] );
	return 0;
    //readData(testset,testlabels,argv[3],argv[4]);
}





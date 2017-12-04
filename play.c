#include <stdio.h>
#include <stdlib.h>
#include "read_mnist.h"

int main(){
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
	readData(dataset_test,labels_test,file_test_set,file_test_label);

	printf("hi\n");
	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#define MAX_FILE_NAME 100
#define TRAIN_NUM 120
#define TEST_NUM 30
#define FEATURE 4
#define NUMBER_OF_CLASSES 3
 
int countNumRows(char *filename)
{
    FILE *fp;
    int count = 0;  // Line counter (result)
    //char filename[MAX_FILE_NAME];
    char c;  // To store a character read from file
 
    // Get file name from user. The file should be
    // either in current folder or complete path should be provided
    //printf("Enter file name: ");
    //scanf("%s", filename);
 
    // Open the file
    fp = fopen(filename, "r");
 
    // Check if file exists
    if (fp == NULL)
    {
        printf("Could not open file %s", filename);
        return -1;
    }
 
    // Extract characters from file and store in character c
    for (c = getc(fp); c != EOF; c = getc(fp))
        if (c == '\n') // Increment count if this character is newline
            count = count + 1;
 
    // Close the file
    fclose(fp);
    //printf("The file %s has %d lines\n ", filename, count);
 
    return count;
}

const char* getfield(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}
/*
Labels for IRIS:
Iris-setosa - 0
Iris-versicolor - 1
Iris-virginica - 2

*/
void read_csv_iris(float *data, float *label, int row_count, char *filename)
{
    //data = (float *)malloc(row_count*4*sizeof(float));
    //label = (int *)malloc(row_count*sizeof(int));
    FILE *fp = fopen(filename,"r");
    char line[1024];
    int count = 0;
    int idx = 0;
    for(int iter = 0;iter<row_count;iter++)
    {
        fgets(line,1024,fp);
        const char *temp_field;
        for(int i=0;i<5;i++)
        {
            float temp_num;
            char *tmp = strdup(line);
            temp_field = getfield(tmp,i+1);
            if(i==4)
            {
                if(strcmp(temp_field,"Iris-setosa")==0)
                {
                    label[idx] = 0;
                    continue;
                }
                if(strcmp(temp_field,"Iris-versicolor")==0)
                {
                    label[idx] = 1;
                    continue;
                }
                if(strcmp(temp_field,"Iris-virginica")==0)
                {
                    label[idx] = 2;
                    continue;
                }
            }
            temp_num = atof(temp_field);
            data[idx*4 + i] = temp_num;
        }
        idx++;
        
    }


}


int main(int argc,char *argv[])
{
    int count;
    float *dataset_train,*dataset_test;
    float *labels_train,*labels_test;
    dataset_train = (float *)malloc(FEATURE * TRAIN_NUM*sizeof(float));
    labels_train = (float *)malloc(TRAIN_NUM*sizeof(float));
    dataset_test = (float *)malloc(FEATURE * TEST_NUM*sizeof(float));
    labels_test = (float *)malloc(TEST_NUM*sizeof(float));
    char file_train_set[] = "data/iris_train.data";
    char file_test_set[] = "data/iris_test.data";
    read_csv_iris(dataset_train,labels_train,TRAIN_NUM,file_train_set);
    read_csv_iris(dataset_test,labels_test,TEST_NUM,file_test_set);
    
    return 0;
}
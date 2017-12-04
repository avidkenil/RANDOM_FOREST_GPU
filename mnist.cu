#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "MnistPreProcess.h"

#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define FEATURE 784
#define NUMBER_OF_CLASSES 10

#define FEAT_KEY 0
#define CUT_KEY 1
#define LEFT_KEY 2
#define RIGHT_KEY 3
#define PRED_KEY 4
#define DEPTH_KEY 5

#define NUM_FIELDS 6

#define index(i, j, N)  ((i)*(N)) + (j)
#define index(i, j, N)  ((i)*(N)) + (j)
#define ixt(i, j, t, N, T) ((t)*(N)*(T)) + ((i)*(N)) + (j)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
 


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
}
int next_pow_2(int x){
	int y = 1;
	while(y < x)
		y*=2;
	return y;
}
void debug(int i){
	cudaError_t e=cudaGetLastError();                                 \
	if(e!=cudaSuccess) {                                              \
		printf("%d Cuda failure %s:%d: '%s'\n", i, __FILE__,__LINE__,cudaGetErrorString(e));    
	}
}

/* === Expanding tree memory === */
float* expand(float* d_trees, int num_trees, int tree_arr_length, int new_tree_arr_length){
	float *new_d_trees;
	assert(new_tree_arr_length >= tree_arr_length);

	cudaMalloc((void **) &new_d_trees, num_trees * NUM_FIELDS * new_tree_arr_length *sizeof(float));
	cudaMemcpy(new_d_trees, d_trees, num_trees * NUM_FIELDS * tree_arr_length *sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(d_trees);
	return new_d_trees;
}
__global__ void get_max_tree_length(int* d_tree_lengths, int num_trees, int* d_max_tree_length){
	extern __shared__ int tree_length_buffer[];
	if(threadIdx.x < num_trees){
		tree_length_buffer[threadIdx.x] = d_tree_lengths[threadIdx.x];
	}else{
		tree_length_buffer[threadIdx.x] = -1;
	}
	
	for(int stride=blockDim.x/2; stride > 0; stride >>=1){
		__syncthreads();
		if(threadIdx.x < stride){
			if(tree_length_buffer[threadIdx.x + stride] > tree_length_buffer[threadIdx.x]){
				tree_length_buffer[threadIdx.x] = tree_length_buffer[threadIdx.x + stride];
			}
		}
	}
	if(threadIdx.x == 0){
	   d_max_tree_length[0] = tree_length_buffer[0];
	}
}
float* maybe_expand(float* d_trees, int num_trees, int* tree_arr_length, int* d_tree_lengths,
	                int* max_tree_length, int* d_max_tree_length){
	// I wonder if it's faster just to compute max on CPU.
	int new_tree_arr_length;
	float *new_d_trees;

	get_max_tree_length<<<1, num_trees, next_pow_2(num_trees) * sizeof(int)>>>(
		d_tree_lengths, num_trees, d_max_tree_length
	);
	cudaMemcpy(max_tree_length, d_max_tree_length, sizeof(int), cudaMemcpyDeviceToHost);
	// Buffer of 2 => up to 2 additions at a time
	if(*max_tree_length <= *tree_arr_length-3){
		return d_trees;
	}else{
		new_tree_arr_length = (*tree_arr_length) * 2;
        while(*max_tree_length > new_tree_arr_length-2){
            new_tree_arr_length *= 2;
        }

        printf("Expanding to %d\n", new_tree_arr_length);
        new_d_trees = expand(d_trees, num_trees, *tree_arr_length, new_tree_arr_length);
        *tree_arr_length = new_tree_arr_length;
        return new_d_trees;
	}
}

/* === Tree Initialization === */
__global__ void kernel_initialize_trees(float *d_trees, int* d_tree_lengths, int tree_arr_length){
	d_trees[ixt(0, LEFT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = 0;
	d_trees[ixt(0, RIGHT_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = 0;
	d_trees[ixt(0, DEPTH_KEY, threadIdx.x, NUM_FIELDS, tree_arr_length)] = 0;
	d_tree_lengths[threadIdx.x] = 1;
}
void initialize_trees(float* d_trees, int num_trees, int tree_arr_length, int* d_tree_lengths){
	kernel_initialize_trees<<<1, num_trees>>>(d_trees, d_tree_lengths, tree_arr_length);
}
__global__ void kernel_initialize_batch_pos(int *d_batch_pos, int x_length, int num_trees){
	int i;
	for(i=threadIdx.x; i<x_length; i+=blockDim.x){
		d_batch_pos[index(blockIdx.x, i, x_length)] = 0;
	}
}
void initialize_batch_pos(int *d_batch_pos, int x_length, int num_trees, cudaDeviceProp dev_prop){
	kernel_initialize_batch_pos<<<num_trees, dev_prop.maxThreadsPerBlock>>>(
		d_batch_pos, x_length, num_trees
	);
}

/* === Tree Traversal === */
__global__ void kernel_traverse_trees(float *d_trees, float* d_x, int x_length, int num_trees, int* d_batch_pos){
	// Should optimize this. It's just a bunch of global reads.
	// Also possibly to rewrite this and batch_traverse to support a "next-step" method instead of a full 
	//   traversal while growing
	int pos, new_pos, left_right_key, x_i, tree_id;
	//Overloading x_i as tx
	x_i = threadIdx.x + blockIdx.x * blockDim.x;
	if(x_i >= x_length * num_trees) return;

	// Actually get x_i, tree_id
	tree_id = x_i % num_trees;
	x_i = x_i / num_trees;
	pos = 0;
    while(1){
        if(d_x[index(x_i, (int) d_trees[ixt(pos, FEAT_KEY, tree_id, NUM_FIELDS, TRAIN_NUM)], FEATURE)] < 
        		d_trees[ixt(pos, CUT_KEY, tree_id, NUM_FIELDS, TRAIN_NUM)]){
            left_right_key = LEFT_KEY;
        }else{
            left_right_key = RIGHT_KEY;
        }
        new_pos = (int) d_trees[ixt(pos, left_right_key, tree_id, NUM_FIELDS, TRAIN_NUM)];
        if(new_pos == pos){
            // Leaf nodes are set up to be idempotent
            break;
        }
        pos = new_pos;
    }
    d_batch_pos[x_i] = pos;
}
void batch_traverse_trees(float *d_tree, float *d_x, int x_length, int num_trees, int *d_batch_pos, cudaDeviceProp dev_prop){
	int block_size, num_blocks;
 	block_size = dev_prop.maxThreadsPerBlock;
 	num_blocks = ceil(num_trees * x_length/((float) block_size));
	kernel_traverse_trees<<<num_blocks, block_size>>>(d_tree, d_x, x_length, num_trees, d_batch_pos);
}
__global__ void kernel_advance_trees(float *d_trees, float* d_x, int x_length, int tree_arr_length, int num_trees, int* d_batch_pos){
	int pos, left_right_key, x_i;
	// threadIdx.x = x_i, blockIdx.x = tree_id
	for(x_i=threadIdx.x; x_i < x_length; x_i+=blockDim.x){
		pos = d_batch_pos[x_i];
	    if(d_x[index(x_i, (int) d_trees[ixt(pos, FEAT_KEY, blockIdx.x, NUM_FIELDS, tree_arr_length)], FEATURE)] < 
	    		d_trees[ixt(pos, CUT_KEY, blockIdx.x, NUM_FIELDS, tree_arr_length)]){
	        left_right_key = LEFT_KEY;
	    }else{
	        left_right_key = RIGHT_KEY;
	    }
	    d_batch_pos[x_i] = (int) d_trees[ixt(pos, left_right_key, blockIdx.x, NUM_FIELDS, tree_arr_length)];
	}
}
void batch_advance_trees(float *d_tree, float *d_x, int x_length, int tree_arr_length, int num_trees, int *d_batch_pos, 
						 cudaDeviceProp dev_prop){
	kernel_advance_trees<<<num_trees, dev_prop.maxThreadsPerBlock>>>(
		d_tree, d_x, x_length, tree_arr_length, num_trees, d_batch_pos
	);
}

/* === Valid features === */
__global__ void kernel_collect_min_max(float* d_x, int* d_batch_pos, int desired_pos, int num_trees, 
									   int x_length, float* d_min_max_buffer){
	extern __shared__ float shared_min_max[]; // threadIdx.x * 2
	// Ripe for optimization.
	// threadIdx.x = x_i, blockIdx.x = tree_id, feat = blockIdx.y
	int x_i;
	float minimum, maximum, val;

	minimum = FLT_MAX;
	maximum = -FLT_MAX;
	for(x_i=threadIdx.x; x_i < x_length; x_i+=blockDim.x){
		if(d_batch_pos[index(blockIdx.x, x_i, x_length)] == desired_pos){
			val = d_x[index(x_i, blockIdx.y, FEATURE)];
			if(val < minimum){
				minimum = val;
			}
			if(val > maximum){
				maximum = val;
			}
		}
	}
	shared_min_max[index(threadIdx.x, 0, 2)] = minimum;
	shared_min_max[index(threadIdx.x, 1, 2)] = maximum;

	for(int stride=blockDim.x/2; stride > 0; stride >>=1){
		__syncthreads();
		if(threadIdx.x < stride){
			if(shared_min_max[index(threadIdx.x + stride, 0, 2)] < shared_min_max[index(threadIdx.x, 0, 2)]){
				shared_min_max[index(threadIdx.x, 0, 2)] = shared_min_max[index(threadIdx.x + stride, 0, 2)];
			}
			if(shared_min_max[index(threadIdx.x + stride, 1, 2)] > shared_min_max[index(threadIdx.x, 1, 2)]){
				shared_min_max[index(threadIdx.x, 1, 2)] = shared_min_max[index(threadIdx.x + stride, 1, 2)];
			}
		}
	}
	if(threadIdx.x==0){
		d_min_max_buffer[ixt(blockIdx.y, 0, blockIdx.x, 2, FEATURE)] = shared_min_max[index(0, 0, 2)];
		d_min_max_buffer[ixt(blockIdx.y, 1, blockIdx.x, 2, FEATURE)] = shared_min_max[index(0, 1, 2)];
		
	}
}
void collect_min_max(float* d_x, int* d_batch_pos, int desired_pos, int num_trees, int x_length,
					 float* d_min_max_buffer, cudaDeviceProp dev_prop){
	// Ripe for optimization.
	dim3 grid(num_trees, FEATURE);
	kernel_collect_min_max<<<grid, dev_prop.maxThreadsPerBlock, dev_prop.maxThreadsPerBlock * sizeof(int) * 2>>>(
		d_x, d_batch_pos, desired_pos, num_trees, x_length, d_min_max_buffer
	);	
}
__global__ void kernel_collect_num_valid_feat(int* d_num_valid_feat, float* d_min_max_buffer, int num_trees){
	extern __shared__ int shared_num_valid_feat_buffer[];
	// blockIdx.x = tree_id
	int sub_num_valid_feat, feat_i;
	sub_num_valid_feat = 0;
	for(feat_i=threadIdx.x; feat_i<FEATURE; feat_i+=blockDim.x){
		if(d_min_max_buffer[ixt(feat_i, 0, blockIdx.x, 2, FEATURE)] != 
			d_min_max_buffer[ixt(feat_i, 1, blockIdx.x, 2, FEATURE)]
			){
			sub_num_valid_feat++;
		}
	}
	shared_num_valid_feat_buffer[threadIdx.x] = sub_num_valid_feat;
	for(int stride=blockDim.x/2; stride > 0; stride >>=1){
		__syncthreads();
		if(threadIdx.x < stride){
			shared_num_valid_feat_buffer[threadIdx.x] += shared_num_valid_feat_buffer[threadIdx.x + stride];
		}
	}
	if(threadIdx.x == 0){
	   d_num_valid_feat[blockIdx.x] = shared_num_valid_feat_buffer[0];
	}
}
void collect_num_valid_feat(int* d_num_valid_feat, float* d_min_max_buffer, int num_trees, cudaDeviceProp dev_prop){
	// Ripe for optimization
	int block_size = MIN(dev_prop.maxThreadsPerBlock, next_pow_2(FEATURE)); // Copy this to other places too
	kernel_collect_num_valid_feat<<<num_trees, block_size, block_size * sizeof(int)>>>(
		d_num_valid_feat, d_min_max_buffer, num_trees
	);
}

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
	readData(dataset_test,labels_test,file_test_set,file_test_label);

	float *trees, *d_trees;
	int *tree_arr_length;
	int *tree_lengths, *d_tree_lengths;
	int *max_tree_length, *d_max_tree_length;
	int feat_per_node;
	int *num_valid_feat, *d_num_valid_feat;
	int tree_pos;
	int *batch_pos, *d_batch_pos; // NUM_TRESS * TRAIN_NUM
	float *min_max_buffer, *d_min_max_buffer;
	int *random_feats, *d_random_feats;
	float *random_cuts, *d_random_cuts;
	int *class_counts_a, *class_counts_b;
	int *d_class_counts_a, *d_class_counts_b;
	int prev_depth, max_depth;
	float *d_x, *d_y;

	int num_trees;
	num_trees = 5;
	// Assumption: num_trees < maxNumBlocks, maxThreadsPerBlock
	printf("num_trees %d\n", num_trees);
	srand(2);

	tree_arr_length = (int *)malloc(sizeof(int));
	tree_lengths = (int *)malloc(num_trees * sizeof(int));
	*tree_arr_length = 1024;
	max_tree_length = (int *)malloc(sizeof(int));

	feat_per_node = (int) ceil(sqrt(FEATURE));

	trees = (float *)malloc(num_trees * NUM_FIELDS * (*tree_arr_length) *sizeof(float));
	batch_pos = (int *)malloc(num_trees * TRAIN_NUM *sizeof(float));
	min_max_buffer = (float *)malloc(num_trees * FEATURE * 2 *sizeof(float));
	
	num_valid_feat = (int *)malloc(num_trees * sizeof(int));
	random_feats = (int *)malloc(num_trees * feat_per_node * sizeof(int));
	random_cuts = (float *)malloc(num_trees * feat_per_node * sizeof(float));

	class_counts_a = (int *)malloc(num_trees * feat_per_node * NUMBER_OF_CLASSES *sizeof(int));
	class_counts_b = (int *)malloc(num_trees * feat_per_node * NUMBER_OF_CLASSES *sizeof(int));
	cudaDeviceProp dev_prop;
	cudaGetDeviceProperties(&dev_prop, 0);
	cudaMalloc((void **) &d_trees, num_trees * NUM_FIELDS * (*tree_arr_length) *sizeof(float));
	cudaMalloc((void **) &d_tree_lengths, num_trees * sizeof(int));
	cudaMalloc((void **) &d_max_tree_length, sizeof(int));
	cudaMalloc((void **) &d_batch_pos, num_trees * TRAIN_NUM *sizeof(float));
	cudaMalloc((void **) &d_min_max_buffer, num_trees * FEATURE * 2 *sizeof(float));
	cudaMalloc((void **) &d_num_valid_feat, num_trees *sizeof(int));
	cudaMalloc((void **) &d_random_feats, num_trees * feat_per_node * sizeof(int));
	cudaMalloc((void **) &d_random_cuts, num_trees * feat_per_node * sizeof(float));
	cudaMalloc((void **) &d_class_counts_a, num_trees * feat_per_node * NUMBER_OF_CLASSES *sizeof(int));
	cudaMalloc((void **) &d_class_counts_b, num_trees * feat_per_node * NUMBER_OF_CLASSES *sizeof(int));
	cudaMalloc((void **) &d_x, TRAIN_NUM * FEATURE *sizeof(float));
	cudaMalloc((void **) &d_y, TRAIN_NUM *sizeof(float));
	cudaMemcpy(d_x, dataset_train, TRAIN_NUM * FEATURE *sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, labels_train, TRAIN_NUM *sizeof(float), cudaMemcpyHostToDevice);

	tree_pos = 0;
	initialize_trees(d_trees, num_trees, *tree_arr_length, d_tree_lengths);
	maybe_expand(d_trees, num_trees, tree_arr_length, d_tree_lengths, max_tree_length, d_max_tree_length);

	//batch_traverse_trees(d_trees, d_x, TRAIN_NUM, num_trees, d_batch_pos, dev_prop);
	initialize_batch_pos(d_batch_pos, TRAIN_NUM, num_trees, dev_prop);
	batch_advance_trees(d_trees, d_x, TRAIN_NUM, *tree_arr_length, num_trees, d_batch_pos, dev_prop);
	collect_min_max(d_x, d_batch_pos, tree_pos, num_trees, TRAIN_NUM,
					d_min_max_buffer, dev_prop);
	collect_num_valid_feat(
		d_num_valid_feat, d_min_max_buffer, num_trees, dev_prop
	);
	cudaMemcpy(num_valid_feat, d_num_valid_feat, num_trees * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\n");
	for(int i=0; i<num_trees; i++){
		printf("%d ", num_valid_feat[i]);
	}
	printf("\n");
	debug(0);
}

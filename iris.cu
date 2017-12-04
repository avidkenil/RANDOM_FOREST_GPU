#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <curand_kernel.h>

#define TRAIN_NUM 120
#define TEST_NUM 30
#define FEATURE 4
#define NUMBER_OF_CLASSES 3


#define FEAT_KEY 0
#define CUT_KEY 1
#define LEFT_KEY 2
#define RIGHT_KEY 3
#define PRED_KEY 4
#define DEPTH_KEY 5

#define NUM_FIELDS 6

#define index(i, j, N)  ((i)*(N)) + (j)
#define ixt(i, j, t, N, T) ((t)*(N)*(T)) + ((i)*(N)) + (j)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
 
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

const char* getfield(char* line, int num){
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
void read_csv_iris(float *data, float *label, int row_count, char *filename){
	//data = (float *)malloc(row_count*4*sizeof(float));
	//label = (int *)malloc(row_count*sizeof(int));
	FILE *fp = fopen(filename,"r");
	char line[1024];
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
__global__ void init_random_generator(curandState *state){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(1337, idx, 0, &state[idx]);
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
__global__ void kernel_initialize_trees(float *d_trees, int* d_tree_lengths){
	d_trees[ixt(0, LEFT_KEY, threadIdx.x, NUM_FIELDS, blockIdx.x)] = 0;
	d_trees[ixt(0, RIGHT_KEY, threadIdx.x, NUM_FIELDS, blockIdx.x)] = 0;
	d_trees[ixt(0, DEPTH_KEY, threadIdx.x, NUM_FIELDS, blockIdx.x)] = 0;
	d_tree_lengths[threadIdx.x] = 1;
}
void initialize_trees(float* d_trees, int num_trees, int* d_tree_lengths){
	kernel_initialize_trees<<<1, num_trees>>>(d_trees, d_tree_lengths);
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
        if(d_x[index(x_i, (int) d_trees[ixt(pos, FEAT_KEY, tree_id, NUM_FIELDS, num_trees)], FEATURE)] < 
        		d_trees[ixt(pos, CUT_KEY, tree_id, NUM_FIELDS, num_trees)]){
            left_right_key = LEFT_KEY;
        }else{
            left_right_key = RIGHT_KEY;
        }
        new_pos = (int) d_trees[ixt(pos, left_right_key, tree_id, NUM_FIELDS, num_trees)];
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
__global__ void kernel_advance_trees(float *d_trees, float* d_x, int x_length, int num_trees, int* d_batch_pos){
	int pos, left_right_key, x_i;
	// threadIdx.x = x_i, blockIdx.x = tree_id
	for(x_i=threadIdx.x; x_i < x_length; x_i+=blockDim.x){
		pos = d_batch_pos[x_i];
	    if(d_x[index(x_i, (int) d_trees[ixt(pos, FEAT_KEY, blockIdx.x, NUM_FIELDS, num_trees)], FEATURE)] < 
	    		d_trees[ixt(pos, CUT_KEY, blockIdx.x, NUM_FIELDS, num_trees)]){
	        left_right_key = LEFT_KEY;
	    }else{
	        left_right_key = RIGHT_KEY;
	    }
	    d_batch_pos[x_i] = (int) d_trees[ixt(pos, left_right_key, blockIdx.x, NUM_FIELDS, num_trees)];
	}
}
void batch_advance_trees(float *d_tree, float *d_x, int x_length, int num_trees, int *d_batch_pos, 
						 cudaDeviceProp dev_prop){
	kernel_advance_trees<<<num_trees, dev_prop.maxThreadsPerBlock>>>(
		d_tree, d_x, x_length, num_trees, d_batch_pos
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
		d_min_max_buffer[ixt(blockIdx.y, 0, blockIdx.x, 2, num_trees)] = shared_min_max[index(0, 0, 2)];
		d_min_max_buffer[ixt(blockIdx.y, 1, blockIdx.x, 2, num_trees)] = shared_min_max[index(0, 1, 2)];
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
		if(d_min_max_buffer[ixt(feat_i, 0, blockIdx.x, 2, num_trees)] != 
			d_min_max_buffer[ixt(feat_i, 1, blockIdx.x, 2, num_trees)]
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
	kernel_collect_num_valid_feat<<<num_trees, block_size, block_size>>>(
		d_num_valid_feat, d_min_max_buffer, num_trees
	);
}

int main(int argc,char *argv[])
{
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
	num_trees = 200;
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
	initialize_trees(d_trees, num_trees, d_tree_lengths);
	maybe_expand(d_trees, num_trees, tree_arr_length, d_tree_lengths, max_tree_length, d_max_tree_length);
	//batch_traverse_trees(d_trees, d_x, TRAIN_NUM, num_trees, d_batch_pos, dev_prop);
	initialize_batch_pos(d_batch_pos, TRAIN_NUM, num_trees, dev_prop);
	batch_advance_trees(d_trees, d_x, TRAIN_NUM, num_trees, d_batch_pos, dev_prop);
	cudaMemcpy(batch_pos, d_batch_pos, num_trees * TRAIN_NUM * sizeof(int), cudaMemcpyDeviceToHost);

	collect_min_max(d_x, d_batch_pos, tree_pos, num_trees, TRAIN_NUM,
					d_min_max_buffer, dev_prop);
	collect_num_valid_feat(
		d_num_valid_feat, d_min_max_buffer, num_trees, dev_prop
	);

	cudaMemcpy(min_max_buffer, d_min_max_buffer, num_trees * FEATURE * 2 *sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<FEATURE; i++){
		printf("%f %f\n", min_max_buffer[ixt(i, 0, 0, 2, num_trees)], 
			              min_max_buffer[ixt(i, 1, 0, 2, num_trees)]);
	}
	cudaMemcpy(num_valid_feat, d_num_valid_feat, num_trees * sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<num_trees; i++){
		printf("%d ", num_valid_feat[i]);
	}
	printf("\n");

	debug(0);

	/*
	for(int i=0; i<TRAIN_NUM; i++){
		printf("%d=%f\n", i, dataset_train[index(i, 0, FEATURE)]);
	}
	*/


	max_depth = 4;

	/*
	prev_depth = -1;
	for(tree_pos=0; tree_pos<*tree_length; tree_pos++){
		printf("%d (depth=%f)\n", tree_pos, tree[index(tree_pos, DEPTH_KEY, NUM_FIELDS)]);
		if(tree[index(tree_pos, DEPTH_KEY, NUM_FIELDS)] > max_depth){
			break;
		}
		if(prev_depth!=tree[index(tree_pos, DEPTH_KEY, NUM_FIELDS)]){
			prev_depth = tree[index(tree_pos, DEPTH_KEY, NUM_FIELDS)];
			batch_traverse_tree(tree, dataset_train, TRAIN_NUM, batch_pos);
		}
		grow_tree(
			dataset_train, labels_train,
			tree_pos, tree, tree_length,
			batch_pos, min_max_buffer, 
			feat_per_node, random_feats, random_cuts,
			class_counts_a, class_counts_b,
			max_depth
		);
		tree = maybe_expand(tree, tree_arr_length, *tree_length);
	}


	printf("Train Accuracy %f\n", calc_accuracy(
		tree, dataset_train, labels_train, TRAIN_NUM, batch_pos
	));
	printf("Test Accuracy %f\n", calc_accuracy(
		tree, dataset_test, labels_test, TEST_NUM, batch_pos
	));
	*/
	return 0;
}
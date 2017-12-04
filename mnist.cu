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

float* expand(float* tree, int* tree_arr_length, int new_tree_arr_length){
	float *new_tree;
	int  i;
	assert(new_tree_arr_length >= *tree_arr_length);
	new_tree = (float *)malloc(NUM_FIELDS * new_tree_arr_length *sizeof(float));
	for(i=0; i<NUM_FIELDS * (*tree_arr_length); i++){
		new_tree[i] = tree[i];
	}
	return new_tree;
}

float* maybe_expand(float* tree, int* tree_arr_length, int tree_length){
	int new_tree_arr_length;
	float *new_tree;
	// Buffer of 2 => up to 2 additions at a time
	if(tree_length <= *tree_arr_length-2){
		return tree;
	}else{
		new_tree_arr_length = (*tree_arr_length) * 2;
		while(tree_length > new_tree_arr_length-2){
			new_tree_arr_length *= 2;
		}

		printf("Expanding to %d\n", new_tree_arr_length);
		
		new_tree = expand(tree, tree_arr_length, new_tree_arr_length);
		*tree_arr_length = new_tree_arr_length;
		return new_tree;
	}
}

void batch_traverse_tree(float *tree, float *x, int x_length, int *batch_pos){
	int pos, new_pos, x_i;
	for(x_i=0; x_i < x_length; x_i++){
		pos = 0;
		while(1){
			if(x[index(x_i, (int) tree[index(pos, FEAT_KEY, NUM_FIELDS)], FEATURE)] < tree[index(pos, CUT_KEY, NUM_FIELDS)]){
				new_pos = (int) tree[index(pos, LEFT_KEY, NUM_FIELDS)];
			}else{
				new_pos = (int) tree[index(pos, RIGHT_KEY, NUM_FIELDS)];
			}
			if(new_pos == pos){
				// Leaf nodes are set up to be idempotent
				break;
			}
			pos = new_pos;
		}
		batch_pos[x_i] = pos;
	}
}

void collect_min_max(float* x, int* batch_pos, int desired_pos, float* min_max_buffer){
	float minimum, maximum, value;
	int x_i, feat_i;
	for(feat_i=0; feat_i < FEATURE; feat_i++){
		minimum = FLT_MAX;
		maximum = FLT_MIN;
		for(x_i=0; x_i<TRAIN_NUM; x_i++){
			if(batch_pos[x_i] != desired_pos){
				continue;
			}
			value = x[index(x_i, feat_i, FEATURE)];
			if(value < minimum){
				minimum = value;
			}
			if(value > maximum){
				maximum = value;
			}
		}
		min_max_buffer[index(feat_i, 0, 2)] = minimum;
		min_max_buffer[index(feat_i, 1, 2)] = maximum;
	}
}

int unif(int low, int high){
	return low + ((int) rand()) % (high - low);
}

float unif(float low, float high){
	return (high - low) * ((float)rand() / RAND_MAX) + low;
}

int int_cmp(const void *a, const void *b) 
{ 
	const int *ia = (const int *)a; // casting pointer types 
	const int *ib = (const int *)b;
	return *ia  - *ib; 
	/* integer comparison: returns negative if b > a 
	and positive if a > b */ 
}

int get_num_valid_feats(float* min_max_buffer){
	int num_valid_feats;
	int feat_i;
	num_valid_feats = 0;
	for(feat_i=0; feat_i<FEATURE; feat_i++){
		if(min_max_buffer[index(feat_i, 0, 2)] != min_max_buffer[index(feat_i, 1, 2)]){
			num_valid_feats++;
		}
	}
	return num_valid_feats;
}

void populate_random_feat_cuts(float* min_max_buffer, int num_valid_feats, int feat_per_node, int* random_feats, float* random_cuts){
	int valid_seen, curr_idx;
	int i, feat_i;

	for(i=0; i<feat_per_node; i++){
		// Overloading. First using random_cuts to store indices.
		random_feats[i] = unif(0, num_valid_feats);
	}
	qsort(random_feats, feat_per_node, sizeof(int), int_cmp);
	valid_seen = 0;
	feat_i = 0;
	for(i=0; i<feat_per_node; i++){
		curr_idx = random_feats[i];
		while(valid_seen < curr_idx){
			feat_i++;
			if(min_max_buffer[index(feat_i, 0, 2)] != min_max_buffer[index(feat_i, 1, 2)]){
				valid_seen++;
			}
		}
		random_feats[i] = feat_i;
		random_cuts[i] = unif(
			min_max_buffer[index(feat_i, 0, 2)], min_max_buffer[index(feat_i, 1, 2)]
		);
	}
}

void place_best_feat_cuts(
			float* x, float* y, int* random_feats, float* random_cuts, 
			int* class_counts_a, int* class_counts_b,
			int feat_per_node, int* batch_pos, int tree_pos, float* tree
		){
	int feat_i, i;
	float best_improvement, best_cut, proxy_improvement;
	int best_feat;
	int total_a, total_b;
	float impurity_a, impurity_b;

	best_improvement = FLT_MIN;
	best_feat = -1;
	best_cut = 0;
	for(feat_i=0; feat_i<feat_per_node; feat_i++){
		total_a = 0;
		total_b = 0;
		for(i=0; i<NUMBER_OF_CLASSES; i++){
			class_counts_a[i] = 0;
			class_counts_b[i] = 0;
		}
		for(i=0; i<TRAIN_NUM; i++){
			if(batch_pos[i] != tree_pos){
				continue;
			}
			if(x[index(i, random_feats[feat_i], FEATURE)] < random_cuts[feat_i]){
				class_counts_a[(int) y[i]]++;
				total_a++;
			}else{
				class_counts_b[(int) y[i]]++;
				total_b++;
			}
		}
		impurity_a = 1;
		impurity_b = 1;
		for(i=0; i<NUMBER_OF_CLASSES; i++){
			impurity_a -= pow(((float) class_counts_a[i]) / total_a, 2);
			impurity_b -= pow(((float) class_counts_b[i]) / total_b, 2);
		}
		proxy_improvement = - total_a * impurity_a - total_b * impurity_b;
		if(proxy_improvement > best_improvement){
			best_feat = random_feats[feat_i];
			best_cut = random_cuts[feat_i];
			best_improvement = proxy_improvement;
		}
	}
	tree[index(tree_pos, FEAT_KEY, NUM_FIELDS)] = best_feat;
	tree[index(tree_pos, CUT_KEY, NUM_FIELDS)] = best_cut;
}

float terminate_node(float* y, int* batch_pos, int pos){
	/*
		If all y's are the same return class
		else return -1
	*/
	int i;
	float y_token;
	y_token = -1;
	for(i=0; i<TRAIN_NUM; i++){
		if(batch_pos[i] == pos){
			if(y_token == -1){
				y_token = y[i];
			}else if(y_token != y[i]){
				return -1;
			}
		}
	}
	// Should never return -1 here. We should see at least 1 y.
	return y_token;
}

int grow_tree(
		float* x, float* y, 
		int tree_pos, float* tree, int* tree_length,
		int* batch_pos, float* min_max_buffer, 
		int feat_per_node, int* random_feats, float* random_cuts,
		int* class_counts_a, int* class_counts_b
	){
	float early_termination;
	int num_valid_feats;

	early_termination = terminate_node(y, batch_pos, tree_pos);
	if(early_termination != -1){
		tree[index(tree_pos, PRED_KEY, NUM_FIELDS)] = early_termination;
		return 0;
	}

	collect_min_max(x, batch_pos, tree_pos, min_max_buffer);

	num_valid_feats = get_num_valid_feats(min_max_buffer);
	populate_random_feat_cuts(min_max_buffer, num_valid_feats, feat_per_node, random_feats, random_cuts);

	place_best_feat_cuts(
		x, y, random_feats, random_cuts, 
		class_counts_a, class_counts_b,
		feat_per_node, batch_pos, tree_pos, tree
	);

	int left_tree_pos;
	int right_tree_pos;

	left_tree_pos = *tree_length;
	right_tree_pos = *tree_length + 1;
	*tree_length += 2;

	// Update tree nodes
	tree[index(tree_pos, LEFT_KEY, NUM_FIELDS)] = left_tree_pos;
	tree[index(tree_pos, RIGHT_KEY, NUM_FIELDS)] = right_tree_pos;

	// Prefill child nodes
	tree[index(left_tree_pos, LEFT_KEY, NUM_FIELDS)] = left_tree_pos;
	tree[index(left_tree_pos, RIGHT_KEY, NUM_FIELDS)] = left_tree_pos;
	tree[index(left_tree_pos, DEPTH_KEY, NUM_FIELDS)] = tree[index(tree_pos, DEPTH_KEY, NUM_FIELDS)] + 1;
	tree[index(left_tree_pos, FEAT_KEY, NUM_FIELDS)] = -1;
	tree[index(left_tree_pos, CUT_KEY, NUM_FIELDS)] = -1;

	tree[index(right_tree_pos, LEFT_KEY, NUM_FIELDS)] = right_tree_pos;
	tree[index(right_tree_pos, RIGHT_KEY, NUM_FIELDS)] = right_tree_pos;
	tree[index(right_tree_pos, DEPTH_KEY, NUM_FIELDS)] = tree[index(tree_pos, DEPTH_KEY, NUM_FIELDS)] + 1;
	tree[index(right_tree_pos, FEAT_KEY, NUM_FIELDS)] = -1;
	tree[index(right_tree_pos, CUT_KEY, NUM_FIELDS)] = -1;
	return 2;
}

float calc_accuracy(float* tree, float* x, float* y, int x_length, int* batch_pos){
	int i, pred, correct;
	batch_traverse_tree(tree, x, x_length, batch_pos);
	correct = 0;
	for(i=0; i<x_length; i++){
		pred = (int) tree[index(batch_pos[i], PRED_KEY, NUM_FIELDS)];
		//printf("%d %d\n", i, pred);
		if(pred == (int) y[i]){
			correct++;
		}
	}
	return ((float) correct) / x_length;
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
}

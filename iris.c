#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>

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
        maximum = -FLT_MAX;
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

int int_unif(int low, int high){
    return low + ((int) rand()) % (high - low);
}

float float_unif(float low, float high){
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
        random_feats[i] = int_unif(0, num_valid_feats);
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
        random_cuts[i] = float_unif(
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

    best_improvement = -FLT_MAX;
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
    return y_token;
}

float get_mode(int* batch_pos, int tree_pos, float* y, int* class_counts){
    int i, maximum_count, maximum_class;
    for(i=0; i<NUMBER_OF_CLASSES; i++){
        class_counts[i] = 0;
    }
    for(i=0; i<TRAIN_NUM; i++){
        if(batch_pos[i] == tree_pos){
            class_counts[(int) y[i]]++;
        }
    }
    maximum_count = -1; 
    maximum_class = -1;
    for(i=0; i<NUMBER_OF_CLASSES; i++){
        if(class_counts[i] > maximum_count){
            maximum_count = class_counts[i];
            maximum_class = i;
        }
    }
    return maximum_class;
}

int grow_tree(
        float* x, float* y, 
        int tree_pos, float* tree, int* tree_length,
        int* batch_pos, float* min_max_buffer, 
        int feat_per_node, int* random_feats, float* random_cuts,
        int* class_counts_a, int* class_counts_b,
        int max_depth
    ){
    float early_termination;
    int num_valid_feats;

    if(tree[index(tree_pos, DEPTH_KEY, NUM_FIELDS)] == max_depth){
        tree[index(tree_pos, PRED_KEY, NUM_FIELDS)] = get_mode(
            batch_pos, tree_pos, y, class_counts_a
        );
        return 0;
    }

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
        if(pred == (int) y[i]){
            correct++;
        }
    }
    return ((float) correct) / x_length;
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
    

    float *tree;
    int *tree_arr_length;
    int *tree_length;
    int feat_per_node;
    int tree_pos;
    int *batch_pos;
    float *min_max_buffer;
    int *random_feats;
    float *random_cuts;
    int *class_counts_a, *class_counts_b;
    int prev_depth, max_depth;

    srand(2);

    tree_arr_length = (int *)malloc(sizeof(int));
    tree_length = (int *)malloc(sizeof(int));
    *tree_arr_length = 1024;
    *tree_length = 1;


    feat_per_node = (int) ceil(sqrt(FEATURE));

    tree = (float *)malloc(NUM_FIELDS * (*tree_arr_length) *sizeof(float));
    batch_pos = (int *)malloc(TRAIN_NUM *sizeof(float));
    min_max_buffer = (float *)malloc(FEATURE * 2 *sizeof(float));
    
    random_feats = (int *)malloc(feat_per_node * sizeof(int));
    random_cuts = (float *)malloc(feat_per_node * sizeof(float));

    class_counts_a = (int *)malloc(NUMBER_OF_CLASSES *sizeof(int));
    class_counts_b = (int *)malloc(NUMBER_OF_CLASSES *sizeof(int));

    tree_pos = 0;
    tree[index(0, LEFT_KEY, NUM_FIELDS)] = 0;
    tree[index(0, RIGHT_KEY, NUM_FIELDS)] = 0;
    tree[index(0, DEPTH_KEY, NUM_FIELDS)] = 0;


    max_depth = 1000;

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
    return 0;
}
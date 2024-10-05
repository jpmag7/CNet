// Neural Network C Library
// Never forget to add an extra
// float for biases on input vector

// gcc neuralNet.c -O3 -funroll-loops -fopenmp -fgcse-las

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "neuralNet.h"


// ===> Structs <=== //

typedef struct layer{
	float* weights;
	int size;
} *Layer;


typedef struct neuralNet{
	int depth, in_size;
	float learning_rate;
	Layer* layers;
} *NN;


// ===> Layer <=== //
Layer newLayer(int prev_size, int size, int weight_sign){
	float* ws = (float*) malloc(sizeof(float) * (size * prev_size + size));
	for(int i = 0; i < (size * prev_size + size); i++){
		float r = (float)rand() / RAND_MAX;
		if(weight_sign >= 0)
			ws[i] = r;
		else
			ws[i] = r * pow(-1, rand());
	}
	Layer layer = (Layer) malloc(sizeof(struct layer));
	layer->weights = ws;
	layer->size = size;
	return layer;
}



void deleteLayer(Layer layer){
	free(layer->weights);
	free(layer);
}



float relu(float x){
	if(x < 0)
		return 0.01 * x;
	else return x;
	// Sigmoid
	//return 1 / (1 + exp(-x));
}



float derelu(float x){
	if(x < 0)
		return 0.01;
	else return 1;
	// Desigmoid
	//return relu(x) * (1 - relu(x));
}



void runLayer(Layer layer, float* input, int in_size, float* results){
	int size = layer->size;
	float* ws = layer->weights;
	int l = 0, c = 0;

	// Make matrix multiplication
	for(l = 0; l < size; l++){
		results[l] = 0;
		for(c = 0; c <= in_size; c++){
			results[l] += ws[(l * (in_size + 1)) + c] * input[c];
		}
		float out = relu(results[l]) / ((float)size);
		results[l] = out;
	}
}

float* trainLayer(Layer layer, float* in, int in_size, float* results, float* errors, float learning_rate){
	int out_size = layer->size;
	float* ws = layer->weights;
	// Initialize errors at zero, plus one for bias
	float* new_errors = (float*) calloc((in_size + 1), sizeof(float));

	for(int l = 0; l < out_size; l++){
		// Index of line start
		int line = l * (in_size + 1);
		// Calculate gradient                                  
		float gradient = derelu(results[l] * (float)out_size) * errors[l] * learning_rate * (float)out_size;
		for(int c = 0; c <= in_size; c++){
			// Cache weight position
			int w_pos = line + c;                                      
			float delta = gradient * in[c];
			// Calculate error for previous layer
			new_errors[c] += ws[w_pos] * errors[l] / (float) in_size;           
			// Update the weight 	         
			ws[w_pos] += delta;                                 
		}
	}
	return new_errors;
}




// ===> Neural Net <=== //
NN newNeuralNet(int length, int* ls_sizes, float learning_rate, int weight_sign){
	int depth = length - 1;
	srand(time(0));
	Layer* layers = (Layer *) malloc(sizeof(Layer) * depth);
	for(int i = 0; i < depth; i++)
		layers[i] = newLayer(ls_sizes[i], ls_sizes[i + 1], weight_sign);
	NN nn = (NN) malloc(sizeof(struct neuralNet));
	nn->learning_rate = learning_rate;
	nn->in_size = ls_sizes[0];
	nn->layers = layers;
	nn->depth = depth;
	return nn;
}



void deleteNeuralNet(NN nn){
	for(int i = 0; i < nn->depth; i++)
		deleteLayer(nn->layers[i]);
	free(nn->layers);
	free(nn);
}

int getNeuralNetInputSize(NN nn){return nn->in_size;}

int getNeuralNetOutputSize(NN nn){return nn->layers[nn->depth - 1]->size;}

float* getNeuralNetOutput(NN nn, float** results){return results[nn->depth - 1];}


void runNeuralNet(NN nn, float* in, float** results){
	int in_size = nn->in_size;
	int new_size = in_size;
	float* new_in = in;

	for(int i = 0; i < nn->depth; i++){
		runLayer(nn->layers[i], new_in, new_size, results[i]);
		new_size = nn->layers[i]->size;
		new_in = results[i];
	}
}


float** newResults(NN nn){
	float** results = (float**)malloc(sizeof(float*) * nn->depth);
	for(int i = 0; i < nn->depth; i++){
		results[i] = (float*) malloc(sizeof(float) * (nn->layers[i]->size + 1));
		results[i][nn->layers[i]->size] = 1;
	}
	return results;
}

void deleteResults(NN nn, float** results){
	for(int i = 0; i < nn->depth; i++)
		free(results[i]);
	free(results);
}

float* calculateErrors(float* out, float* targets, int size){
	float* errors = (float*) malloc(sizeof(float) * (size));
	for(int i = 0; i < size; i++)
		errors[i] = targets[i] - out[i];
	return errors;
}


float* trainNeuralNet(NN nn, float* in, float** results, float* errors){
	int in_size = nn->in_size;
	float learning_rate = nn->learning_rate;
	Layer lst_layer = nn->layers[nn->depth - 1];
	int depth = nn->depth;

	for(int l = depth - 1; l > 0; l--){
		float* new_errors = 
			trainLayer(nn->layers[l], results[l - 1], nn->layers[l - 1]->size, results[l], errors, learning_rate);
		free(errors);
		errors = new_errors;
	}
	float* new_errors1 = trainLayer(nn->layers[0], in, in_size, results[0], errors, learning_rate);
	free(errors);
	return new_errors1;
}


void writeNNToFile(NN nn, char* file_name){
	FILE *fp = fopen(file_name, "w");
	if(fp == NULL){
		printf("[WRITE ERROR] opening file %s\n", file_name);
		return;
	}
	int in_size = nn->in_size;
	fprintf(fp, "%ds%ds%fs", nn->depth, nn->in_size, nn->learning_rate);
	for(int i = 0; i < nn->depth; i++){
		Layer layer = nn->layers[i];
		fprintf(fp, "%ds", layer->size);
		for(int l = 0; l < layer->size; l++){
			int line = l * (in_size + 1);
			for(int c = 0; c <= in_size; c++)
				fprintf(fp, "%fs", layer->weights[line + c]);
		}
		in_size = layer->size;
	}
	fclose(fp);
}


NN readNNFromFile(char* file_name){
	FILE *fp = fopen(file_name, "r");
	if(fp == NULL){
		printf("[READ ERROR] opening file %s\n", file_name);
		return NULL;
	}
	NN nn = (NN) malloc(sizeof(struct neuralNet));
	int prev_size;
	fscanf(fp, "%ds%ds%fs", &nn->depth, &prev_size, &nn->learning_rate);
	nn->in_size = prev_size;
	Layer* layers = (Layer *) malloc(sizeof(Layer) * nn->depth);
	for(int i = 0; i < nn->depth; i++){
		Layer layer = (Layer) malloc(sizeof(struct layer));
		fscanf(fp, "%ds", &layer->size);
		layer->weights = (float*) malloc(sizeof(float) * (layer->size * prev_size + layer->size));
		for(int l = 0; l < layer->size; l++){
			int line = l * (prev_size + 1);
			for(int c = 0; c <= prev_size; c++)
				fscanf(fp, "%fs", &(layer->weights[line + c]));
		}
		prev_size = layer->size;
		layers[i] = layer;
	}
	nn->layers = layers;
	fclose(fp);
	return nn;
}


void test(){

	int epocks = 100000;
	int batch_size = 3;
	int layers[] = {2, 1024, 100, 1};
	NN nn = newNeuralNet(4, layers, 0.01, 1);

	float** inputs  = (float**) malloc(sizeof(float*) * 4);
	float** targets = (float**) malloc(sizeof(float*) * 4);

	inputs [0] = (float*) malloc(sizeof(float) * 3);
	targets[0] = (float*) malloc(sizeof(float) * 2);
	inputs [0][0] = 1;
	inputs [0][1] = 0;
	inputs [0][2] = 1;
	targets[0][0] = 1;
	targets[0][1] = 1;
	inputs [1] = (float*) malloc(sizeof(float) * 3);
	targets[1] = (float*) malloc(sizeof(float) * 2);
	inputs [1][0] = 0;
	inputs [1][1] = 1;
	inputs [1][2] = 1;
	targets[1][0] = 1;
	targets[1][1] = 0;
	inputs [2] = (float*) malloc(sizeof(float) * 3);
	targets[2] = (float*) malloc(sizeof(float) * 2);
	inputs [2][0] = 1;
	inputs [2][1] = 1;
	inputs [2][2] = 1;
	targets[2][0] = 0;
	targets[2][1] = 0;
	inputs [3] = (float*) malloc(sizeof(float) * 3);
	targets[3] = (float*) malloc(sizeof(float) * 2);
	inputs [3][0] = 0;
	inputs [3][1] = 0;
	inputs [3][2] = 1;
	targets[3][0] = 0;
	targets[3][1] = 1;

	clock_t start = clock();
	float** results = newResults(nn);

	for(int i = 0; i < epocks; i++){
		int index     = rand() % 4;
		float* input  = inputs[index];
		float* target = targets[index];

		runNeuralNet(nn, input, results);
		float* errors = calculateErrors(results[nn->depth - 1], target, 1);
		float* new_errors = trainNeuralNet(nn, input, results, errors);
    //printf("Error=>%f\n", new_errors[0]);
	}

  
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Trainning of REVERSE in %d epocks, batck_size %d took: %.24lf seconds\n", epocks, batch_size, seconds);
	
	writeNNToFile(nn, "testCNet.cnet");
	printf("Wrote NN to file\n");
	deleteNeuralNet(nn);
	nn = readNNFromFile("testCNet.cnet");
	printf("Done reading NN from file\n");
	remove("testCNet.cnet");
	
	results = newResults(nn);
	runNeuralNet(nn, inputs[0], results);
	printf("[1, 0]: %.24lf   =>  %.24lf\n", results[nn->depth - 1][0], results[nn->depth - 1][1]);
	runNeuralNet(nn, inputs[1], results);
	printf("[0, 1]: %.24lf   =>  %.24lf\n", results[nn->depth - 1][0], results[nn->depth - 1][1]);
	runNeuralNet(nn, inputs[2], results);
	printf("[1, 1]: %.24lf   =>  %.24lf\n", results[nn->depth - 1][0], results[nn->depth - 1][1]);
	runNeuralNet(nn, inputs[3], results);
	printf("[0, 0]: %.24lf   =>  %.24lf\n", results[nn->depth - 1][0], results[nn->depth - 1][1]);

	printf("Threads: %d\n", omp_get_num_procs());

	deleteResults(nn, results);
	deleteNeuralNet(nn);

	for (int i = 0; i < 4; i++){
		free(inputs[i]);
		free(targets[i]);
	}
	free(inputs);
	free(targets);
}

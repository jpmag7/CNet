#include "neuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>

int NUM_FILES = 24;

typedef struct file_struct{
  FILE** files;
  long* sizes;
}* FS;

FS createFileList(){
  FS fs = (FS)malloc(sizeof(struct file_struct));
  fs->files = (FILE**)malloc(sizeof(FILE*) * NUM_FILES);
  fs->sizes = (long*) malloc(sizeof(long)  * NUM_FILES);
  char name[15];
  
  for(int i = 0; i < NUM_FILES; i++){
    sprintf(name, "DataSet/%02d.txt", i + 1);
    fs->files[i] = fopen(name, "r");
    fs->sizes[i] = 0;
    for (char c = getc(fs->files[i]); c != EOF; c = getc(fs->files[i]))
      fs->sizes[i]++;
  }
  return fs;
}


NN createAtentionHead(){
  // {input_size, layers_sizes...}
  int layers[] = {1024, 1024, 682, 341};
	NN atention_head = newNeuralNet(4, layers, 0.01, 1);
  return atention_head;
}


NN createBody(){
  // {input_size, layers_sizes...}
  int layers[] = {682, 682, 341, 682, 341, 682, 341, 256}; 
	NN body = newNeuralNet(8, layers, 0.01, 1);
  return body;
}


float** createInputVectors(FS fs, int* size, float* targets){
  size[0] = 2;
  float** input_vectors = (float**) malloc(sizeof(float*) * 2);
  input_vectors[0] = (float*) malloc(sizeof(float) * (1024 + 1));
  for(int i = 0; i < 1025; i++) input_vectors[0][i] = (float)rand() / RAND_MAX;
  input_vectors[0][1024] = 1;
  input_vectors[1] = (float*) malloc(sizeof(float) * (1024 + 1));
  for(int i = 0; i < 1025; i++) input_vectors[1][i] = (float)rand() / RAND_MAX;
  input_vectors[1][1024] = 1;
  for(int i = 0; i < 256; i++) targets[i] = 1.0f / (i % 3 + 1);
  return input_vectors;
}


void trainCicle(FS fs, NN ah, NN bd){
  // + Create targets
  float* targets = (float*) malloc(sizeof(float) * 256);
  // + Create body results
  float** body_results = newResults(bd);
  // + Create body Input with last Atention head result (+1 for bias)
  float* body_input = (float*) malloc(sizeof(float) * (682 + 1));

  for (int i = 0; i < 1000; i++){
    // + Create Input Vectors
    int input_vectors_size = 0;
    float** input_vectors = createInputVectors(fs, &input_vectors_size, targets);
    // Zero first input
    memset(&input_vectors[0][683], 0, sizeof(float) * 341);
    // + Create result_list
    float*** results_list = (float***) malloc(sizeof(float**) * input_vectors_size);
    for(int v = 0; v < input_vectors_size; v++) results_list[v] = newResults(ah);
    // foreach Vector in Input Vectors
    for(int v = 0; v < input_vectors_size; v++){
      // -> if(Vector not first) Copy result of result_list[Vector-1] to Inputs[Vector]
      if(v > 0) {
        float* ah_output = getNeuralNetOutput(ah, results_list[v - 1]);
        for(int j = 0; j < 341; j++) input_vectors[v][683 + j] = ah_output[j];
      }
      // -> Run Atention Head
      runNeuralNet(ah, input_vectors[v], results_list[v]);
    }
    // Fill body input
    for(int j = 0; j < 341; j++) body_input[j] = input_vectors[0][j];
    float* ah_final_output = getNeuralNetOutput(ah, results_list[input_vectors_size - 1]);
    for(int j = 0; j < 341; j++) body_input[j + 341] = ah_final_output[j];
    body_input[682] = 1;
    // Run Body
    runNeuralNet(bd, body_input, body_results);
    // Calculate body errors
    float* body_errors = calculateErrors(getNeuralNetOutput(bd, body_results), targets, 256);
    // atentionHeadErrors = Train body
    float* ah_errors = trainNeuralNet(bd, body_input, body_results, body_errors);
    // Reversed for loop (foreach Vector in Input Vectors)
    for(int v = input_vectors_size - 1; v >= 0; v--){
      // -> Train atention head (in = Vector, result_list[Vector], atentionHeadErrors)
      ah_errors = trainNeuralNet(ah, input_vectors[v], results_list[v], ah_errors);
      // -> - Delete result_list[Vector]
      free(results_list[v]);
      // -> - Delete Vector(in)
      free(input_vectors[v]);
    }
    // - Delete Atention Heads errors
    free(ah_errors);
    // - Delete Input Vectors pointer
    free(input_vectors);
    // - Delete result_list pointer
    free(results_list);
  }
  // - Delete targets
  free(targets);
  // - Delete body Input
  free(body_input);
  for(int j = 0; j < getNeuralNetOutputSize(bd); j++)
    printf("%f, ", getNeuralNetOutput(bd, body_results)[j]);
  // - Delete body results
  deleteResults(bd, body_results);
  // Save to file
  printf("\nTRAIN CICLE DONE\n");
}

int main(){

  // Open dataset
  FS fs = createFileList();

  // Create AI
  NN ah = createAtentionHead();
  NN bd = createBody();
  
  // Train AI
  trainCicle(fs, ah, bd);
  //test();
  printf("DONE\n");
  
  return 0;
}
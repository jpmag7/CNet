#ifndef NEURALNET_H_
#define NEURALNET_H_

typedef struct neuralNet* NN;
typedef struct layer* Layer;

NN      newNeuralNet(int length, int* ls_sizes, float learning_rate, int weight_sign);
float*  trainNeuralNet(NN nn, float* in, float** results, float* errors);
float*  calculateErrors(float* out, float* targets, int size);
void    runNeuralNet(NN nn, float* in, float** results);
float*  getNeuralNetOutput(NN nn, float** results);
void    writeNNToFile(NN nn, char* file_name);
void    deleteResults(NN nn, float** results);
NN      readNNFromFile(char* file_name);
int     getNeuralNetOutputSize(NN nn);
int     getNeuralNetInputSize(NN nn);
void    deleteNeuralNet(NN nn);
float** newResults(NN nn);
void    test();

#endif
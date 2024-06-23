#ifndef MLP_H
#define MLP_H

#include "./tensor.h"

// Type of a pointer to an activation function
typedef Tensor* (*ActivationFuncPointer)(Tensor*);

typedef struct {
    Tensor *weights;
    Tensor *biases;
    ActivationFuncPointer activation_func;
    int in_features;
    int out_features;
} DenseLayer;

typedef struct {
    DenseLayer** layers;
    int num_layers;
} LayerList;

DenseLayer* create_dense_layer(int in_features, int out_features, char activation[]);
Tensor* forward_dense(Tensor* input, DenseLayer* layer);
LayerList* create_mlp(int in_features, int* layer_sizes, int n_layers);
Tensor* forward_layers(Tensor* input, LayerList* layers);
void free_dense(DenseLayer* layer);
void free_layer_list(LayerList* layers);


#endif // MLP_H
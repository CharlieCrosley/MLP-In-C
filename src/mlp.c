#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utility.h"
#include "./tensor.h"
#include "mlp.h"
#include "tensor_ops.h"

/* Create a dense layer with optional activation function.
   Supported activation functions: relu, sigmoid, NULL */
DenseLayer* create_dense_layer(int in_features, int out_features, char activation[]) {
    DenseLayer* new_layer = (DenseLayer*)malloc(sizeof(DenseLayer));
    ActivationFuncPointer activation_func = get_activation_func_from_str(activation);
    
    // init weights, randomly sample values between -1 and 1 with uniform probability
    float *weight_data = uniform_random_array(in_features * out_features, -1, 1);
    int weight_shape[] = {in_features, out_features};
    Tensor* weights = create_tensor(weight_data, weight_shape, 2, 1);

    float bias_data[out_features];
    // Initialize all bias values to zero
    memset(bias_data, 0, out_features * sizeof(float));
    int bias_shape[] = {out_features};
    Tensor* biases = create_tensor(bias_data, bias_shape, 1, 1);

    new_layer->weights = weights;
    new_layer->biases = biases;
    new_layer->activation_func = activation_func;
    new_layer->in_features = in_features;
    new_layer->out_features = out_features;
    
    return new_layer;
}

Tensor* forward_dense(Tensor* input, DenseLayer* layer) {
    Tensor *matmul_output = matmul(input, layer->weights);
    Tensor *bias_output = add(matmul_output, layer->biases);
    if (layer->activation_func) {
        Tensor *output = layer->activation_func(bias_output);
        return output;
    }
    else {
        return bias_output;
    }
}

LayerList* create_mlp(int in_features, int* layer_sizes, int n_layers) {
    LayerList* mlp = (LayerList*)malloc(n_layers * sizeof(LayerList));
    DenseLayer** layers = (DenseLayer**)malloc(n_layers * sizeof(DenseLayer*));
    mlp->layers = layers;
    mlp->num_layers = n_layers;
    
    if (n_layers == 1) {
        mlp->layers[0] = create_dense_layer(in_features, layer_sizes[0], NULL);
        return mlp;
    }

    // input layer
    mlp->layers[0] = create_dense_layer(in_features, layer_sizes[0], "relu");

    // hidden layers
    for (int i=0; i < n_layers-2; i++) {
        mlp->layers[i+1] = create_dense_layer(layer_sizes[i], layer_sizes[i+1], "relu");
    }
    // output layer
    mlp->layers[n_layers-1] = create_dense_layer(layer_sizes[n_layers-2], layer_sizes[n_layers-1], "sigmoid");
    
    return mlp;
}

Tensor* forward_layers(Tensor* input, LayerList* layers) {
    Tensor* x = input;
    for (int i=0; i < layers->num_layers; i++) {
        x = forward_dense(x, layers->layers[i]);
    }
    return x;
}

void free_dense(DenseLayer* layer) {
    if (layer) {
        free_tensor(layer->weights);
        free_tensor(layer->biases);
        free(layer->activation_func);
        free(layer);
        layer = NULL;
    }
}

void free_layer_list(LayerList* layers) {
    if (layers) {
        for (int i=0; i < layers->num_layers; i++) {
            free_dense(layers->layers[i]);
        }

        free(layers);
        layers = NULL;
    }
}
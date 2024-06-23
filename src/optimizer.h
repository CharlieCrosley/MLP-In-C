#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include "backward.h"

typedef struct SGD {
    float lr;
    // points to a function that takes a pointer to a Tensor struct as its argument.
    void (*update)(struct Topo* topo, float lr); 
} SGD;


SGD* init_sgd(float lr);

#endif // OPTIMIZER_H
#ifndef BACKWARD_H
#define BACKWARD_H

#include "tensor.h"

typedef struct Topo {
    Tensor** ordering;
    int length;
} Topo;

Topo* build_topo(Tensor* t);
Topo* backward(Tensor* t);
void free_graph_from_tensor(Tensor* t);
void free_topo(Topo* topo);
void free_graph_from_topo(Topo* topo);

#endif // BACKWARD_H
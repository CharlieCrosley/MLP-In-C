#ifndef TENSOROPS_H
#define TENSOROPS_H

#include "tensor.h"

Tensor* add(Tensor* a, Tensor* b); 
Tensor* sum(Tensor* t);
Tensor* reduce_sum(Tensor* t);
Tensor* matmul(Tensor* a, Tensor* b);
Tensor* mul(Tensor* a, Tensor* b);
Tensor* relu(Tensor* input);
Tensor* sigmoid(Tensor* input);

#endif // TENSOROPS_H
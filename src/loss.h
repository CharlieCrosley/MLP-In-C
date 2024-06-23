#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

Tensor* binary_cross_entropy(Tensor* y_pred, Tensor* y_true);
void backward_binary_cross_entropy(Tensor* result);

#endif // LOSS_H
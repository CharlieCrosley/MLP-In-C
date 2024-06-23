#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "optimizer.h"

void backward_binary_cross_entropy(Tensor* result) {
    Tensor* y_pred = result->parents[0];
    Tensor* y_true = result->parents[1];

    for (int i = 0; i < y_pred->size; i++) {
        y_pred->grad[i] += (y_pred->data[i] - y_true->data[i]) / 
            ((1 - y_pred->data[i]) * y_pred->data[i]) / 
            y_true->size;
    }
}

/* Binary cross entropy loss with mean reduction */
Tensor* binary_cross_entropy(Tensor* y_pred, Tensor* y_true) {
    if (y_pred->size != y_true->size) {
        printf("Prediction and Truth tensors must have the same size!\n");
        free_graph_from_tensor(y_pred);
        free_tensor(y_true);
        exit(EXIT_FAILURE);
    }

    float loss[1] = {0.0};

    for (int i = 0; i < y_pred->size; i++) {
        // Ensure that y_pred is bounded between a very small value and 1.
        // a very small value to avoid log(0) which is undefined
        float pred = y_pred->data[i];
        if (pred < 1e-5) {
            pred = 1e-5;
        }
        if (pred > 1 - 1e-5) {
            pred = 1 - 1e-5;
        }

        // Compute the BCE loss for the current sample
        loss[0] += -((y_true->data[i] * log(pred)) + ((1 - y_true->data[i]) * log(1 - pred)));
    }
    loss[0] /= y_pred->size;
    int shape[1] = {1};
    Tensor* loss_tensor = create_tensor(loss, shape, 1, y_pred->requires_grad);
    loss_tensor->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
    loss_tensor->parents[0] = y_pred;
    loss_tensor->parents[1] = y_true;
    loss_tensor->num_parents = 2;
    loss_tensor->backward_func = backward_binary_cross_entropy;
    return loss_tensor;
}
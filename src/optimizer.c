#include <stdio.h>
#include <stdlib.h>

#include "optimizer.h"
#include "tensor.h"

void sgd_update(Topo* topo, float lr) {
    // -1 from length to not change loss
    for (int i=0; i < topo->length-1; i++) {
        for (int j=0; j < topo->ordering[i]->size; j++) {
            if (topo->ordering[i]->requires_grad) {
                topo->ordering[i]->data[j] -= topo->ordering[i]->grad[j] * lr;
            }
        }
    }
}

SGD* init_sgd(float lr) {
    SGD* optim = (SGD*)malloc(sizeof(SGD));
    if (!optim) {
        printf("Memory allocation failed when allocating memory for SGD optimizer.\n");
        exit(EXIT_FAILURE);
    }
    optim->lr = lr;
    optim->update = sgd_update;
    return optim;
}
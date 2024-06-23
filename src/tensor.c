#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

/* Create a new tensor from data */
Tensor* create_tensor(float* data, int* shape, int num_dims, int requires_grad) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        fprintf(stderr, "Memory allocation failed when allocating memory for a new tensor.\n");
        exit(EXIT_FAILURE);
    }
    t->num_dims = num_dims;

    t->shape = (int*)malloc(num_dims * sizeof(int));
    if (!t->shape) {
        fprintf(stderr, "Memory allocation failed when allocating memory for a new tensors shape array.\n");
        free_tensor(t);
        exit(EXIT_FAILURE);
    }
    int size = 1;
    for (int i = 0; i < num_dims; i++) {
        t->shape[i] = shape[i];
        size *= shape[i];
    }
    t->size = size;

    t->data = (float*)malloc(size * sizeof(float));
    if (!t->data) {
        fprintf(stderr, "Memory allocation failed when allocating memory for a new tensor data array.\n");
        free_tensor(t);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        t->data[i] = data[i]; // copy data
    }

    t->grad = (float*)calloc(size, sizeof(float)); // initialize all grads to zero with calloc
    if (!t->grad) {
        fprintf(stderr, "Memory allocation failed when allocating memory for a new tensors grad array.\n");
        free_tensor(t);
        exit(EXIT_FAILURE);
    }
    t->backward_func = NULL;
    t->parents = NULL;
    t->num_parents = 0;
    t->requires_grad = requires_grad;
    return t;
}

/* Add a new parent to a tensor */
void add_parent(Tensor* child, Tensor* parent) {
    child->num_parents++;

    if (child->num_parents==1) {
        child->parents = (Tensor**)malloc(child->num_parents * sizeof(Tensor*));
    } else {
        child->parents = (Tensor**)realloc(child->parents, child->num_parents * sizeof(Tensor*));
    }
    child->parents[child->num_parents] = parent;
}


/* Free the memory of a tensor */
void free_tensor(Tensor* t) {
    if (t) {
        if (t->data) {
            free(t->data);
            t->data = NULL;
        }
        if (t->grad) {
            free(t->grad);
            t->grad = NULL;
        }
        if (t->shape) {
            free(t->shape);
            t->shape = NULL;
        }
        if (t->parents) {
            free(t->parents);
            t->parents = NULL;
        }
        free(t);
        t = NULL; // set pointer to NULL to help prevent float freeing
    }
}
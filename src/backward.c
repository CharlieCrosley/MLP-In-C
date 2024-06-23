#include <stdio.h>
#include <stdlib.h>

#include "backward.h"
#include "tensor.h"

/* Linear search to find if the given value is in the array. A hashmap would be more efficient but linear search is very simple */
int has_child_been_visited(Tensor* pointer, Tensor** visited, int visited_length) {
    for (int i=0; i < visited_length; i++) {
        // check if pointer to tensor is in the visited array
        if (pointer == visited[i]) {
            return 1;
        }
    }
    return 0;
}

void build_topo_recursive(Tensor* self, Tensor*** visited_ptr, Tensor*** topo_ptr, int* num_visited_nodes, int* num_topo_nodes, int* topo_capacity) {
    // only add child once
    if (has_child_been_visited(self, *visited_ptr, *num_visited_nodes)) return;
    
    (*visited_ptr)[(*num_visited_nodes)++] = self;
    if (*num_visited_nodes >= *topo_capacity) {
        // Double the size of the allocated memory when the arrays are full
        *topo_capacity *= 2;
        *visited_ptr = (Tensor**)realloc(*visited_ptr, *topo_capacity * sizeof(Tensor*));
        *topo_ptr = (Tensor**)realloc(*topo_ptr, *topo_capacity * sizeof(Tensor*));
        if (*visited_ptr == NULL || *topo_ptr == NULL) {
            perror("Reallocation failed");
            exit(EXIT_FAILURE);
        }
    }
    // Depth-first search so that the required gradients are available when back propagating through the graph
    for (int i = 0; i < self->num_parents; i++) {
        build_topo_recursive(self->parents[i], visited_ptr, topo_ptr, num_visited_nodes, num_topo_nodes, topo_capacity);
    }
    // Add self to topo here
    (*topo_ptr)[(*num_topo_nodes)++] = self;
}

/* Create a topological ordering of the Tensors in the graph. */
Topo* build_topo(Tensor* self) {
    int topo_capacity = 10; // initial size
    Tensor** visited = (Tensor**)calloc(topo_capacity, sizeof(Tensor*));
    int topo_length = 0;

    // Dynamically allocated topo array
    Tensor** topo = (Tensor**)malloc(topo_capacity * sizeof(Tensor*));
    int topo_initial_idx = 0;

    // use &visited and &topo to update the pointer when reallocating memory
    build_topo_recursive(self, &visited, &topo, &topo_length, &topo_initial_idx, &topo_capacity);

    free(visited);

    // resize the topo to free extra allocated memory
    topo = (Tensor**)realloc(topo, topo_length * sizeof(Tensor*));
    Topo* result = (Topo*)malloc(sizeof(Topo));
    result->ordering = topo;
    result->length = topo_length;
    return result;
}

void _compute_gradients(Topo* topo) {
    // reverse list
    for (int i=topo->length-1; i >= 0; i--) {
        // tensor will not have a backward_func assigned if it is a leaf node
        if (topo->ordering[i]->requires_grad && topo->ordering[i]->backward_func) {
            topo->ordering[i]->backward_func(topo->ordering[i]);
        }
    }
}

void _zero_gradients(Topo* topo) {
    for (int i=0; i < topo->length; i++) {
        if (topo->ordering[i]->requires_grad) {
            for (int j=0; j < topo->ordering[i]->size; j++) {
                topo->ordering[i]->grad[j] = 0.0;
            }
        }
    }
}

Topo* backward(Tensor* t) {
    if (t->size != 1) {
        printf("Tensor must be a scaler in order to perform back propagation.\n");
        free_tensor(t);
        exit(EXIT_FAILURE);
    }

    Topo* topo = build_topo(t);
    // Zeroing gradients in backwards function is not ideal when accumulating gradients over multiple batches.
    // But it simplifies and speeds up the code as I dont have to recompute the topo
    _zero_gradients(topo); 
    t->grad[0] = 1.0; // Set the starting tensors gradient to 1
    _compute_gradients(topo);

    return topo;
}

/* Free each tensor in a topological ordering */
void free_graph_from_topo(Topo* topo) {
    for (int i=topo->length-1; i >= 0; i--) {
        // Don't free leaf nodes as they can be inputs or weights/biases
        // Pointer will be NULL if it has already been freed
        if (topo->ordering[i] != NULL && topo->ordering[i]->num_parents > 0) {
            free_tensor(topo->ordering[i]);
        }
    }
    free_topo(topo);
}

/* Build a topological ordering of the tensors graph and free each tensor*/
void free_graph_from_tensor(Tensor* t) {
    Topo* topo = build_topo(t);

    for (int i=topo->length-1; i >= 0; i--) {
        // Don't free leaf nodes as they can be inputs or weights/biases
        if (topo->ordering[i] != NULL && topo->ordering[i]->num_parents > 0) {
            free_tensor(topo->ordering[i]);
        }
    }
    free_topo(topo);
}

void free_topo(Topo* topo) {
    if (topo->ordering) free(topo->ordering);
    if (topo) free(topo);
    topo = NULL;
}
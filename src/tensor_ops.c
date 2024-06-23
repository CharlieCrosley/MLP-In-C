#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "tensor.h"
#include "utility.h"
#include "backward.h"

int ensure_requires_grad(Tensor* t) {
    if (!t->requires_grad) {
        printf("Attempted to compute gradient for tensor with requires_grad=False!\n");
        free_graph_from_tensor(t);
        exit(EXIT_FAILURE);
    }
}

int ensure_one_of_requires_grad(Tensor* a, Tensor* b) {
    if (!a->requires_grad && !b->requires_grad) {
        printf("Attempted to compute gradient for two tensors with requires_grad=False!\n");
        free_graph_from_tensor(a);
        free_graph_from_tensor(b);
        exit(EXIT_FAILURE);
    }
}

void backward_add(Tensor* result) {
    for (int i = 0; i < result->num_parents; i++) {
        ensure_requires_grad(result->parents[i]);
        for (int j = 0; j < result->parents[i]->size; j++) {
            result->parents[i]->grad[j] += result->grad[j];
        }
    }
}

void backward_sum(Tensor* result) {
    Tensor* parent = result->parents[0];
    ensure_requires_grad(parent);
    int last_parent_dim = parent->shape[parent->num_dims-1];
    for (int i = 0; i < result->size; i++) {
        for (int j=0; j < last_parent_dim; j++) {
            parent->grad[i*last_parent_dim + j] += result->grad[i];
        }
    }
}

void backward_reduce_sum(Tensor* result) {
    Tensor* parent = result->parents[0];
    ensure_requires_grad(parent);
    for (int i = 0; i < parent->size; i++) {
        // deposit the grad from the result value into each grad of the parent
        parent->grad[i] += result->grad[0];
    }
}

void backward_matmul(Tensor* result) {
    Tensor* a = result->parents[0];
    Tensor* b = result->parents[1];

    ensure_one_of_requires_grad(a, b);

    // Case 1: One or both of the tensors are 1D
    if (a->num_dims == 1 || b->num_dims == 1) {
        // Get the 1D tensor (both can be 1D)
        Tensor* t_1d = a->num_dims == 1 ? a : b;
        Tensor* t_other = a->num_dims == 1 ? b : a;
        
        int last_dim_size = t_other->shape[t_other->num_dims-1];
        for (int i = 0; i < result->size; i++) {
            for (int j=0; j < last_dim_size; j++) {
                t_1d->grad[j] += result->grad[i] * t_other->data[i*last_dim_size + j];
                t_other->grad[i*last_dim_size + j] += result->grad[i] * t_1d->data[j];
            }
        }
    } 
    // Case 2: Both tensors have arbitrary shapes 2D+
    else { 
        int num_leading_dims = result->num_dims-2;
        int leading_dims_size = 1;
        for (int i = 0; i < num_leading_dims; i++) {
            leading_dims_size *= result->shape[i];
        }
        int M = a->shape[num_leading_dims]; // 2nd last dim in a [.., .., M, ..]
        int K = a->shape[num_leading_dims + 1]; // last dim in a [.., .., .., K]
        int N = b->shape[num_leading_dims + 1]; // last dim in b [.., .., .., N]

        for (int batch = 0; batch < leading_dims_size; batch++) {
            // Calculate offsets since matrix elements are a flattened 1D array
            // take the number of elements in the last two dims and repeat it batch times to offset the calculations
            int offset_a = (batch * M * K) % a->size; 
            int offset_b = (batch * K * N) % b->size;
            int offset_result = batch * M * N;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < K; k++) {
                        a->grad[offset_a + i * K + k] += 
                            result->grad[offset_result + i * N + j] * b->data[offset_b + k * N + j];
                        b->grad[offset_b + k * N + j] += 
                            result->grad[offset_result + i * N + j] * a->data[offset_a + i * K + k];
                    }
                }
            }
        }
    }
}

void backward_mul(Tensor* result) {
    Tensor* a = result->parents[0];
    Tensor* b = result->parents[1];

    ensure_one_of_requires_grad(a, b);

    for (int i = 0; i < result->size; i++) {
        int offset_a = i % a->size;
        int offset_b = i % b->size;
        a->grad[offset_a] += result->grad[i] * b->data[offset_b];
        b->grad[offset_b] += result->grad[i] * a->data[offset_a];    
    }
}

void backward_relu(Tensor* result) {
    Tensor* parent = result->parents[0];
    ensure_requires_grad(parent);
    
    for (int i = 0; i < result->size; ++i) {
        parent->grad[i] += result->grad[i] * (result->data[i] > 0 ? 1 : 0);
    }
}

void backward_sigmoid(Tensor* result) {
    Tensor* parent = result->parents[0];
    ensure_requires_grad(parent);
    
    for (int i = 0; i < result->size; ++i) {
        parent->grad[i] += result->grad[i] * (result->data[i] * (1 - result->data[i]));
    }
}

Tensor* add(Tensor* a, Tensor* b) {
    // Ensure that the tensors have the same shape
    if (!is_broadcastable(a, b)) {
        handle_shape_mismatch(a, b);
    }
    float* result_data;
    // Case 1: Two 1D tensors
    if (a->num_dims == 1 && b->num_dims == 1) {
        int size = a->shape[0];
        result_data = (float*)malloc(size * sizeof(float));
        for (int i = 0; i < size; i++) {
            result_data[i] = a->data[i] + b->data[i];
        }
    } 
    // Case 2: Two arbitrary shaped 2D+ tensors 
    else {
        const Tensor* t1 = a->num_dims > b->num_dims ? a : b;
        const Tensor* t2 = a->num_dims <= b->num_dims ? a : b;

        int num_leading_dims = t1->num_dims - 1;

        // Calculate shape after broadcasting
        int size = 1;
        for (int i=0; i < num_leading_dims; i++) {
            size *= t1->shape[i] > t2->shape[i] ? t1->shape[i] : t2->shape[i];
        }

        int last_dim_size = t1->shape[t1->num_dims - 1];
        result_data = (float*)malloc((size*last_dim_size) * sizeof(float));

        // Essentially flattens the leading dims and adds the last two dims
        for (int batch = 0; batch < size; batch++) {
            for (int i = 0; i < last_dim_size; i++) {
                result_data[batch * last_dim_size + i] = t1->data[(batch * last_dim_size + i) % t1->size] + t2->data[(batch * last_dim_size + i) % t2->size];
            }
        }
    }

    int requires_grad = a->requires_grad || b->requires_grad;
    Tensor* result = create_tensor(result_data, a->shape, a->num_dims, requires_grad);
    free(result_data);

    result->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
    result->parents[0] = a;
    result->parents[1] = b;
    result->num_parents = 2;
    result->backward_func = backward_add;

    return result;
}

/* Sum over the last dim */
Tensor* sum(Tensor* t) {
    int last_dim = t->shape[t->num_dims-1];
    int result_size = t->size / last_dim;
    float result_data[result_size];

    for (int i=0; i < result_size; i++) {
        result_data[i] = 0; // zero before sum
        for (int j=0; j < last_dim; j++) {
            result_data[i] += t->data[i*last_dim + j];
        }
    }

    Tensor* result = create_tensor(result_data, t->shape, t->num_dims-1, t->requires_grad);

    result->parents = (Tensor**)malloc(sizeof(Tensor*));
    if (!result->parents) {
        fprintf(stderr, "Memory allocation failed in sum.\n");
        free_tensor(result);
        exit(EXIT_FAILURE);
    }
    result->parents[0] = t;
    result->num_parents = 1;
    result->backward_func = backward_sum;

    return result;
}

/* Sum all elements across dimensions */
Tensor* reduce_sum(Tensor* t) {
    int result_shape[1] = {1};
    float result_data[1] = {0};
    for (int i=0; i < t->size; i++) {
        result_data[0] += t->data[i];
    }
    
    Tensor* result = create_tensor(result_data, result_shape, 1, t->requires_grad);

    result->parents = (Tensor**)malloc(sizeof(Tensor*));
    if (!result->parents) {
        fprintf(stderr, "Memory allocation failed in reduce_sum.\n");
        free_tensor(result);
        exit(EXIT_FAILURE);
    }
    result->parents[0] = t;
    result->num_parents = 1;
    result->backward_func = backward_reduce_sum;

    return result;
}

Tensor* matmul(Tensor* a, Tensor* b) {
    // Ensure that the tensors are compatible for matmul
    if (!is_broadcastable_matmul(a, b)) {
        handle_shape_mismatch(a, b);
    }

    int result_dims;
    float* result_data;
    int* shape;
    int result_size = 1;

    // Case 1: One or both of the tensors are 1D
    if (a->num_dims == 1 || b->num_dims == 1) {
        // Get the 1D tensor (both can be 1D)
        Tensor* t_1d = a->num_dims == 1 ? a : b;
        Tensor* t_other = a->num_dims == 1 ? b : a;

        // Drop the last dim for the result and ensure it is at least 1
        result_dims = t_other->num_dims-1 < 1 ? 1 : t_other->num_dims-1;
        shape = (int*)malloc(result_dims * sizeof(int));

        if (a->num_dims == 1 && b->num_dims == 1) { // both 1D
            shape[0] = 1;
        } 
        else { // 1D and ND
            for (int i=0; i < t_other->num_dims; i++) {
                shape[i] = t_other->shape[i]; // copy the shape
            }
            result_size = t_other->size;
        }
        
        result_data = (float*)calloc(result_size, sizeof(float));
        int last_dim_size = t_other->shape[t_other->num_dims-1];
        for (int i = 0; i < result_size; i++) {
            for (int j=0; j < last_dim_size; j++) {
                result_data[i] += t_1d->data[j] * t_other->data[i*last_dim_size + j];
            }
        }
    } 
    // Case 2: Both tensors have arbitrary shapes 2D+
    else {
        result_dims = a->num_dims > b->num_dims ? a->num_dims : b->num_dims;
        int num_leading_dims = result_dims - 2; // Number of leading dimensions (batch or arbitrary)
        
        // Get the shape after broadcasting
        shape = (int*)malloc(result_dims * sizeof(int));
        for (int i = 0; i < num_leading_dims; i++) {
            shape[i] = a->shape[i] > b->shape[i] ? a->shape[i] : b->shape[i];;
        }
        shape[num_leading_dims] = a->shape[a->num_dims-2];
        shape[num_leading_dims + 1] = b->shape[b->num_dims-1];

        // Calculate the sizes
        int leading_dims_size = 1;
        for (int i = 0; i < result_dims; i++) {
            result_size *= shape[i];
            if (i < num_leading_dims) leading_dims_size *= shape[i];
        }

        // Initialise the result array to zeros since we are summing not assigning
        result_data = (float*)calloc(result_size, sizeof(float));

        int M = shape[num_leading_dims]; // 2nd last dim in shape [.., .., M, ..]
        int N = shape[num_leading_dims + 1]; // last dim in shape [.., .., .., N]
        int K = a->shape[num_leading_dims + 1]; // last dim in a [.., .., .., K]

        for (int batch = 0; batch < leading_dims_size; batch++) {
            // Calculate offsets since matrix elements are a flattened 1D array
            int offset_a = (batch * M * K) % a->size;
            int offset_b = (batch * K * N) % b->size;
            int offset_result = batch * M * N;
            // loop over last two dims
            for (int i = 0; i < M; i++) { // each row in result
                for (int j = 0; j < N; j++) { // each column in result
                    for (int k = 0; k < K; k++) {  // each column of a
                        // a->data[offset_a + i * K + k] goes through each column of a and increments the row i
                        // b->data[offset_b + k * N + j] goes through each row of b and increments the column with j
                        result_data[offset_result + i * N + j] += a->data[offset_a + i * K + k] * b->data[offset_b + k * N + j];
                    }
                }
            }
        }
    }

    int requires_grad = a->requires_grad || b->requires_grad;
    Tensor* result = create_tensor(result_data, shape, result_dims, requires_grad);

    result->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
    if (!result->parents) {
        fprintf(stderr, "Memory allocation failed in matmul.\n");
        free_tensor(result);
        exit(EXIT_FAILURE);
    }
    result->parents[0] = a;
    result->parents[1] = b;
    result->num_parents = 2;
    result->backward_func = backward_matmul;

    free(result_data);
    free(shape);

    return result;
}

Tensor* mul(Tensor* a, Tensor* b) {
    // Ensure that the tensors are compatible for mul
    if (!is_broadcastable(a, b)) {
        handle_shape_mismatch(a, b);
    }

    // Calculate the shape of the result with broadcasting
    int result_dims = a->num_dims > b->num_dims ? a->num_dims : b->num_dims;
    int shape[result_dims];

    int result_size = 1;
    for (int i = 0; i < result_dims; i++) {
        // Tensors may have differing number of dims
        if (i < a->num_dims && i < b->num_dims) {
            shape[i] = a->shape[i] > b->shape[i] ? a->shape[i] : b->shape[i];
        } 
        else if (i >= a->num_dims) {
            shape[i] = b->shape[i];
        } else {
            shape[i] = a->shape[i];
        }
        result_size *= shape[i];
    }

    float result_data[result_size];
    for (int i = 0; i < result_size; i++) {
        result_data[i] = a->data[i % a->size] * b->data[i % b->size];    
    }

    int requires_grad = a->requires_grad || b->requires_grad;
    Tensor* result = create_tensor(result_data, shape, result_dims, requires_grad);

    result->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
    if (!result->parents) {
        fprintf(stderr, "Memory allocation failed in mul.\n");
        free_tensor(result);
        exit(EXIT_FAILURE);
    }
    result->parents[0] = a;
    result->parents[1] = b;
    result->num_parents = 2;
    result->backward_func = backward_mul;

    return result;
}

Tensor* relu(Tensor* t) {
    float activations[t->size];
    for (int i = 0; i < t->size; ++i) {
        activations[i] = t->data[i] > 0 ? t->data[i] : 0;
    }
    
    Tensor* result = create_tensor(activations, t->shape, t->num_dims, t->requires_grad);

    result->parents = (Tensor**)malloc(sizeof(Tensor*));
    if (!result->parents) {
        fprintf(stderr, "Memory allocation failed in relu.\n");
        free_tensor(result);
        exit(EXIT_FAILURE);
    }
    result->parents[0] = t;
    result->num_parents = 1;
    result->backward_func = backward_relu;
    return result;
}

Tensor* sigmoid(Tensor* t) {
    float activations[t->size];
    for (int i = 0; i < t->size; ++i) {
        activations[i] = 1 / (1 + exp(-t->data[i]));
    }
    Tensor* result = create_tensor(activations, t->shape, t->num_dims, t->requires_grad);

    result->parents = (Tensor**)malloc(sizeof(Tensor*));
    if (!result->parents) {
        fprintf(stderr, "Memory allocation failed in sigmoid.\n");
        free_tensor(result);
        exit(EXIT_FAILURE);
    }
    result->parents[0] = t;
    result->num_parents = 1;
    result->backward_func = backward_sigmoid;
    return result;
}
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utility.h"
#include "tensor.h"
#include "tensor_ops.h"
#include "backward.h"


/* Return N evenly spaced numbers between two values. */
float* linspace(float start, float end, int n) {
    float* result = (float*)malloc(n * sizeof(float));
    if (!result) {
        fprintf(stderr, "Memory allocation failed when allocating memory for linspace.\n");
        exit(EXIT_FAILURE);
    }

    float step = (end - start) / (n - 1);
    for (int i = 0; i < n; i++) {
        result[i] = start + step * i;
    }
    return result;
}

/* Element-wise compare float arrays using relative and absolute tolerances */
int compare_tensor_data(float* data1, float* data2, int size) {
    float rtol = 0.00001;
    float atol = 0.00000001;

    for (int i = 0; i < size; i++) {
        float diff = fabs(data1[i] - data2[i]);
        float tolerance = atol + rtol * fabs(data2[i]);
        
        if (diff > tolerance) {
            printf("Mismatch at index %d: %.8lf != %.8lf\n", i, data1[i], data2[i]);
            return 0;
        }
    }
    return 1;
}

/* Round float array data to dp decimal places */
void round_float_array(float* data, int size, int dp) {
    float coef = pow(10, dp);
    for (int i = 0; i < size; i++) {
        data[i] = ceil(data[i] * coef) / coef;
    }
}

/* Return a string containing the shape of a given tensor */
char* getTensorShapeString(Tensor* tensor) {
    int *sizes = tensor->shape;
    int count = tensor->num_dims;
    // Dont need to start with 2 for '[' and ']' in the buffer size
    // because the last size in the sizes list doesn't need a ", " after it 
    int bufferSize = 0; 

    // Calculate buffer size needed
    for (int i = 0; i < count; i++) {
        bufferSize += snprintf(NULL, 0, "%d", sizes[i]) + 2; // +2 for ", " or terminating null
    }

    // Allocate memory for the sizes string
    char *sizesString = (char *)malloc(bufferSize);
    if (sizesString == NULL) {
        return NULL; // Memory allocation failed
    }

    // Create the sizes string with brackets
    int offset = sprintf(sizesString, "[");
    for (int i = 0; i < count; i++) {
        offset += sprintf(sizesString + offset, "%d", sizes[i]);
        if (i < count - 1) {
            offset += sprintf(sizesString + offset, ", ");
        }
    }
    sprintf(sizesString + offset, "]");

    return sizesString;
}

/* Check if the shape of two tensors is broadcastable */
int is_broadcastable(const Tensor* a, const Tensor* b) {
    // Broadcasting valid conditions:
    // 1. Same size at dim
    // 2. 1 at dim
    // 3. dim does not exist

    // Determine the larger and smaller tensors if they are not the same size
    const Tensor* t1 = a->num_dims > b->num_dims ? a : b;
    const Tensor* t2 = a->num_dims <= b->num_dims ? a : b;
    const int diff = t1->num_dims - t2->num_dims;

    int start_idx = t1->num_dims-1;
    // single dim is treated differently so that it only iterates once
    int end_idx = t2->num_dims == 1 ? t2->num_dims : t1->num_dims - t2->num_dims;
    for (int i=start_idx; i >= end_idx; i--) {
        // negate diff from i so that the last dims are aligned
        if (t1->shape[i] != t2->shape[i-diff] && (t1->shape[i] != 1 && t2->shape[i-diff] != 1)) {
            return 0;
        }
    }
    return 1;
}

/* Check if the shape of two tensors is broadcastable for matrix multiplication */
int is_broadcastable_matmul(const Tensor* a, const Tensor* b) {
    if (a->num_dims == 1) {
        return a->shape[0] == b->shape[b->num_dims-1];
    }
    if (b->num_dims == 1) {
        return b->shape[0] == a->shape[a->num_dims-1];
    }
    // Check that last dimension of a matches second-to-last dimension of b
    if (a->shape[a->num_dims - 1] != b->shape[b->num_dims - 2]) {
        return 0;
    }
    // Check leading dimensions for broadcasting
    int num_leading_dims = a->num_dims - 2;
    if (num_leading_dims > 0) {
        // ignore last two dims
        const Tensor* t1 = a->num_dims > b->num_dims ? a : b;
        const Tensor* t2 = a->num_dims <= b->num_dims ? a : b;
        const int diff = t1->num_dims - t2->num_dims;

        int start_idx = t1->num_dims-1 -2;
        // single dim is treated differently so that it only iterates once
        int end_idx = t2->num_dims == 1 ? t2->num_dims-2 : t2->num_dims-1 -2;
        if (end_idx <= 0) return 1; // no leading dims to compare

        for (int i=start_idx; i >= end_idx; i--) {
            // negate diff from i so that the last dims are aligned
            if (t1->shape[i] != t2->shape[i-diff] && (t1->shape[i] != 1 && t2->shape[i-diff] != 1)) {
                return 0;
            }
        }
    }
    
    return 1;
}

/* Displays the tensor shapes then errors/exits */
void handle_shape_mismatch(Tensor* a, Tensor* b) {
    // Generate shapes only if the check fails
    char *shapeA = getTensorShapeString(a);
    char *shapeB = getTensorShapeString(b);

    free_graph_from_tensor(a);
    free_graph_from_tensor(b);
    if (shapeA && shapeB) {
        printf("Error: Tensor shape mismatch with sizes %s and %s", shapeA, shapeB);

        // Trigger an assertion failure with the error message
        exit(EXIT_FAILURE);
    } else {
        printf("Failed to allocate memory for the sizes of the shape string");
        exit(EXIT_FAILURE);
    }
}

ActivationFuncPointer get_activation_func_from_str(char activation[]) {
    if (activation == NULL) {
        return NULL;
    }
    else if (strcmp(activation, "relu") == 0) 
    {
        return relu;
    } 
    else if (strcmp(activation, "sigmoid") == 0)
    {
        return sigmoid;
    }
    else /* default: */
    {
        printf("Unknown activation function given... defaulting to ReLU.");
        return relu;
    }
}

/* Return a uniformly sampled random float between min and max*/
float generate_uniform_random_float(float min, float max) {
    return min + (float)rand() / RAND_MAX * (max - min);
}

/* Return an array filled with uniformly sampled random floats between min and max */
float* uniform_random_array(int size, float min, float max) {
    float* arr = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i ++) {
        arr[i] = generate_uniform_random_float(min, max);
    }
    return arr;
}

/* Helper function to calculate the stride for a given dimension */
int get_stride(int *shape, int dims, int depth) {
    int stride = 1;
    for (int i = depth; i < dims; i++) {
        stride *= shape[i];
    }
    return stride;
}

void print_tensor_helper(float* values, int* shape, int dims, int depth, int offset) {

    if (depth == dims - 1) {
        // Base case: print the elements in the last dimension
        printf("[");
        for (int i = 0; i < shape[depth]; i++) {
            printf("%.4f", values[offset + i]);
            if (i < shape[depth]-1) printf(" ");
        }
        printf("]");
        
    } else {
        // Recursive case: handle higher dimensions
        printf("[");
        for (int i = 0; i < shape[depth]; i++) {
            // Align the opening parenthesis with blank spaces 
            if (i > 0 && depth > 0) {
                for (int i = 0; i < dims-1; i++) {
                    printf(" ");
                }
            } else if (i > 0 && depth == 0) {
                printf(" ");
            }

            int stride = get_stride(shape, dims, depth + 1);
            print_tensor_helper(values, shape, dims, depth + 1, offset + i * stride);

            if (dims <= 2) {
                if (i+1 != shape[depth]) printf("\n");
            } else {
                if (depth == 0 && i+1 != shape[depth]) printf("]\n\n"); // empty line between batch dimensions
                else if (depth == 0) printf("]");
                else if (depth == dims-2 && i+1 != shape[depth]) printf("\n"); // new line every row except last
            }
        }
        if (depth == 0) printf("]\n"); // highest-level closing parenthesis with new line underneath
    }
}

/* Recursively print a tensors data */
void print_tensor(const Tensor* t, int print_grads) {
    if (print_grads) {
        printf("Tensor Gradients:\n");
        print_tensor_helper(t->grad, t->shape, t->num_dims, 0, 0);
    } else {
        printf("Tensor Data:\n");
        print_tensor_helper(t->data, t->shape, t->num_dims, 0, 0);
    }
    if (t->num_dims == 1) printf("\n");
}
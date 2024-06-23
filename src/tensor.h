#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor {
    float* data;
    float* grad;
    int* shape;
    int size;
    int num_dims;
    void (*backward_func)(struct Tensor*); // points to a function that takes a pointer to a Tensor struct as its argument
    struct Tensor** parents; // pointer to a list of tensor pointers
    int num_parents;
    int requires_grad;
} Tensor;

Tensor* create_tensor(float* data, int* shape, int num_dims, int requires_grad);
void add_parent(Tensor* child, Tensor* parent);
void print_tensor(const Tensor* t, int print_grad);
void free_tensor(Tensor* t);

#endif // TENSOR_H
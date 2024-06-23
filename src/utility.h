#ifndef UTILITY_H
#define UTILITY_H

#include "tensor.h"
#include "mlp.h"

float* linspace(float start, float end, int num);
int compare_tensor_data(float* data1, float* data2, int size);
void round_float_array(float* data, int size, int dp);
char* getTensorShapeString(Tensor* tensor);
void handle_shape_mismatch(Tensor* a, Tensor* b);
void print_tensor_helper(float* values, int* shape, int dims, int depth, int offset);
void print_tensor(const Tensor* t, int print_grads);
int get_stride(int *shape, int dims, int depth);
ActivationFuncPointer get_activation_func_from_str(char activation[]);
float generate_uniform_random_float(float min, float max);
float* uniform_random_array(int size, float min, float max);
int is_broadcastable(const Tensor* a, const Tensor* b);
int is_broadcastable_matmul(const Tensor* a, const Tensor* b);

#endif // UTILITY_H
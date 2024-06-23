#include <stdio.h>

#include "../src/tensor_ops.h"
#include "../src/tensor.h"
#include "../src/utility.h"
#include "../src/mlp.h"
#include "../src/backward.h"

void test_dense_forward() {
    float input_data[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    // batch of 2 with 3 features each
    int input_shape[] = {2,3};
    Tensor *input = create_tensor(input_data, input_shape, 2, 0);

    // Take 3 features, output 2 features with relu activation
    DenseLayer* dense_layer = create_dense_layer(3, 2, "relu");

    // Manually set the weights for predictability
    float weight_data[] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
    float bias_data[] = {0.0, 10.0};
    dense_layer->weights->data = weight_data;
    dense_layer->biases->data = bias_data;

    Tensor *output = forward_dense(input, dense_layer);

    float expected_data[] = {14.0, 24.0, 14.0, 24.0};

    // Check if the result is as expected
    if (compare_tensor_data(output->data, expected_data, 1)) {
        printf("%-30s PASSED\n", "test_dense_forward:");
    } else {
        printf("%-30s FAILED\n", "test_dense_forward:");
    }

    // Free the allocated memory
    free_tensor(input);
    free_dense(dense_layer);
    free_graph_from_tensor(output);
}

void test_dense_backward() {
    float input_data[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    // batch of 2 with 3 features each
    int input_shape[] = {2,3};
    Tensor *input = create_tensor(input_data, input_shape, 2, 1);

    // Take 3 features, output 2 features with relu activation
    DenseLayer* dense_layer = create_dense_layer(3, 2, "relu");

    // Manually set the weights for predictability
    float weight_data[] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
    float bias_data[] = {0.0, 10.0};
    dense_layer->weights->data = weight_data;
    dense_layer->biases->data = bias_data;

    Tensor* output = forward_dense(input, dense_layer); // output from relu
    Tensor* loss = reduce_sum(output);
    float expected_input_grads[] = {2.0, 4.0, 6.0, 2.0, 4.0, 6.0};
    float expected_weight_grads[] = {2.0, 2.0, 4.0, 4.0, 6.0, 6.0};
    Topo* topo = backward(loss);

    // Check if the result is as expected
    if (compare_tensor_data(input->grad, expected_input_grads, 2) && compare_tensor_data(dense_layer->weights->grad, expected_weight_grads, 2)) {
        printf("%-30s PASSED\n", "test_dense_backward:");
    } else {
        printf("%-30s FAILED\n", "test_dense_backward:");
    }

    // Free the allocated memory
    free_tensor(input);
    free_dense(dense_layer);
    free_graph_from_tensor(loss);
    free_topo(topo);
}

// Main function to run tests
int main() {
    test_dense_forward(); 
    test_dense_backward(); 

    return 0;
}

#include <stdio.h>

#include "../src/tensor.h"
#include "../src/utility.h"
#include "../src/mlp.h"
#include "../src/optimizer.h"
#include "../src/tensor_ops.h"

void test_sgd_update() {
    float input_data[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    // batch of 2 with 3 features each
    int input_shape[] = {2,3};
    Tensor *input = create_tensor(input_data, input_shape, 2, 0);

    // Take 3 features, output 2 features with relu activation
    DenseLayer* dense_layer = create_dense_layer(3, 2, "relu");

    // Manually set the weights for predictability
    float weight_data[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float bias_data[] = {0.0, 10.0};
    dense_layer->weights->data = weight_data;
    dense_layer->biases->data = bias_data;

    Tensor* output = forward_dense(input, dense_layer); // output from relu
    Tensor* loss = reduce_sum(output);
    float expected_weights_post_update[] = {0.8, 0.8, 1.6, 1.6, 2.4, 2.4};
    
    Topo* topo = backward(loss);

    SGD* optim = init_sgd(0.1);
    optim->update(topo, optim->lr);

    // Check if the result is as expected
    if (compare_tensor_data(dense_layer->weights->data, expected_weights_post_update, 2)) {
        printf("%-30s PASSED\n", "test_sgd_update:");
    } else {
        printf("%-30s FAILED\n", "test_sgd_update:");
    }

    // Free the allocated memory
    free_tensor(input);
    free_dense(dense_layer);
    free_graph_from_tensor(loss);
    free_topo(topo);
}

int main() {
    test_sgd_update();

    return 0;
}
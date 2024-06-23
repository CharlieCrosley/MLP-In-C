#include <stdio.h>

#include "../src/loss.h"
#include "../src/tensor.h"
#include "../src/utility.h"

const int PADDING_WIDTH = -35;

void test_binary_cross_entropy() {
    int shape[] = {5};
    float y_true_data[] = {1, 0, 1, 0, 1};
    float y_pred_data[] = {0.9, 0.1, 0.8, 0.3, 0.95};
    Tensor* y_true = create_tensor(y_true_data, shape, 1, 0);
    Tensor* y_pred = create_tensor(y_pred_data, shape, 1, 0);

    float expected_loss_data[] = {0.16836658};

    Tensor* loss = binary_cross_entropy(y_pred, y_true);
    
    // Check if the data is as expected
    if (compare_tensor_data(loss->data, expected_loss_data, loss->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_binary_cross_entropy:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_binary_cross_entropy:");
    }

    free_tensor(y_true);
    free_tensor(y_pred);
    free_tensor(loss);
}

void test_binary_cross_entropy_backward() {
    int shape[] = {4};
    float y_true_data[] = {0, 0, 1, 1};
    float y_pred_data[] = {0.1, 0.2, 0.5, 0.9};
    Tensor* y_true = create_tensor(y_true_data, shape, 1, 0);
    Tensor* y_pred = create_tensor(y_pred_data, shape, 1, 1);

    float expected_grads[] = { 0.27777779,  0.31249997, -0.50000000, -0.27777779};

    Tensor* loss = binary_cross_entropy(y_pred, y_true);

    loss->backward_func(loss);
    
    // Check if the data is as expected
    if (compare_tensor_data(y_pred->grad, expected_grads, y_pred->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_binary_cross_entropy_backward:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_binary_cross_entropy_backward:");
    }

    free_tensor(y_true);
    free_tensor(y_pred);
    free_tensor(loss);
}


int main() {
    test_binary_cross_entropy();
    test_binary_cross_entropy_backward();

    return 0;
}
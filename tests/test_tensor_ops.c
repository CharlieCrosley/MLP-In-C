#include <stdio.h>

#include "../src/tensor_ops.h"
#include "../src/tensor.h"
#include "../src/utility.h"
#include "../src/dataset.h"

const int PADDING_WIDTH = -35;

void test_add_1d() {
    int shape[] = {2};
    float data1[] = {1.0, 2.0};
    float data2[] = {3.0, 4.0};

    Tensor* t1 = create_tensor(data1, shape, 1, 0);
    Tensor* t2 = create_tensor(data2, shape, 1, 0);

    Tensor* sum = add(t1, t2);

    float expected_data[] = {4.0, 6.0};

    // Check if the result is as expected
    if (compare_tensor_data(sum->data, expected_data, sum->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_add_1d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_add_1d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(sum);
}

void test_add_3d() {
    int shape1[] = {2,2,2};
    int shape2[] = {1,2,2}; 
    float data1[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float data2[] = {3.0, 4.0, 3.0, 4.0};

    Tensor* t1 = create_tensor(data1, shape1, 3, 0);
    Tensor* t2 = create_tensor(data2, shape2, 3, 0);

    Tensor* sum = add(t1, t2);

    float expected_data[] = {4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0};

    // Check if the result is as expected
    if (compare_tensor_data(sum->data, expected_data, sum->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_add_3d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_add_3d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(sum);
}

void test_add_backward_1d() {
    int shape1[] = {2};
    float data1[] = {1.0, 2.0};
    float data2[] = {3.0, 4.0};

    Tensor* t1 = create_tensor(data1, shape1, 1, 1);
    Tensor* t2 = create_tensor(data2, shape1, 1, 1);

    Tensor* sum = add(t1, t2);

    // Initialize the gradient of the result tensor
    sum->grad[0] = 1.0;
    sum->grad[1] = 1.0;

    // Perform the backward pass
    sum->backward_func(sum);

    float expected_grad[] = {1.0, 1.0};

    // Check if the gradients are as expected
    if (compare_tensor_data(t1->grad, expected_grad, t1->size) && compare_tensor_data(t2->grad, expected_grad, t2->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_add_backward_1d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_add_backward_1d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(sum);
}

void test_sum_3d() {
    int shape[] = {2, 2, 3};
    float data[] = {
        1, 2, 3, 
        4, 5, 6, 

        1, 2, 3, 
        4, 5, 6,
    };
    float expected_result[] = {
        6.0, 15.0,
        6.0, 15.0
    };

    Tensor* t = create_tensor(data, shape, 3, 0);
    Tensor* result = sum(t);

    // Check if the result is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_sum_3d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_sum_3d:");
    }

    // Free the allocated memory
    free_tensor(t);
    free_tensor(result);
}

void test_sum_backward_3d() {
    int shape[] = {2, 2, 3};
    float data[] = {
        1, 2, 3, 
        4, 5, 6, 

        1, 2, 3, 
        4, 5, 6,
    };
    float expected_grad[] = {
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
    
        3.0, 3.0, 3.0,
        4.0, 4.0, 4.0,
    
    };

    Tensor* t = create_tensor(data, shape, 3, 1);
    Tensor* result = sum(t);
    int size = 4;
    for (int i=0; i < size; i++) {
        // init grad with incrementing values 0-3
        result->grad[i] = (float)i+1;
    }

    // Perform the backward pass
    result->backward_func(result);

    // Check if the gradients are as expected
    if (compare_tensor_data(t->grad, expected_grad, t->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_sum_backward_3d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_sum_backward_3d:");
    }

    // Free the allocated memory
    free_tensor(t);
    free_tensor(result);
}

void test_reduce_sum_2d() {
    int shape[] = {2, 3};
    float data[] = {
        1, 2, 3, 
        1, 2, 3,
    };
    float expected_result[] = {
        12.0
    };

    Tensor* t = create_tensor(data, shape, 2, 0);
    Tensor* result = reduce_sum(t);

    // Check if the result is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_reduce_sum_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_reduce_sum_2d:");
    }

    // Free the allocated memory
    free_tensor(t);
    free_tensor(result);
}

void test_reduce_sum_backward_2d() {
    int shape[] = {2, 3};
    float data[] = {
        1, 2, 3, 
        1, 2, 3,
    };
    float expected_grads[] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0
    };

    Tensor* t = create_tensor(data, shape, 2, 1);
    Tensor* result = reduce_sum(t);

    // Set final grad to 1.0 and perform backward pass
    result->grad[0] = 1.0;
    result->backward_func(result);

    // Check if the gradients are as expected
    if (compare_tensor_data(t->grad, expected_grads, t->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_reduce_sum_backward_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_reduce_sum_backward_2d:");
    }

    // Free the allocated memory
    free_tensor(t);
    free_tensor(result);
}

void test_matmul_3d() {
    int shape1[] = {2, 3, 4};
    int shape2[] = {2, 4, 5};
    float data1[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,

        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    };
    float data2[] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,

        21, 22, 23, 24, 25,
        26, 27, 28, 29, 30,
        31, 32, 33, 34, 35,
        36, 37, 38, 39, 40
    };
    float expected_result[] = {
        110,  120,  130,  140,  150,
        246,  272,  298,  324,  350,
        382,  424,  466,  508,  550,

        1678, 1736, 1794, 1852, 1910,
        2134, 2208, 2282, 2356, 2430,
        2590, 2680, 2770, 2860, 2950
    };

    Tensor* t1 = create_tensor(data1, shape1, 3, 0);
    Tensor* t2 = create_tensor(data2, shape2, 3, 0);

    // Perform batched matrix multiplication
    Tensor* result = matmul(t1, t2);

    // Check if the result is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_matmul_3d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_matmul_3d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_matmul_2d() {
    int data_shape1[] = {2, 3}; 
    int data_shape2[] = {3, 4}; 
    float data1[] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
    };
    float data2[] = {
        1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 
    };
    float expected_result[] = {
        6., 12., 18., 24., 15., 30., 45., 60.
    };

    Tensor* t1 = create_tensor(data1, data_shape1, 2, 0);
    Tensor* t2 = create_tensor(data2, data_shape2, 2, 0);

    // Perform batched matrix multiplication
    Tensor* result = matmul(t1, t2);

    // Check if the result is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_matmul_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_matmul_2d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_matmul_1d() {
    int shape[] = {4};
    float data1[] = {
        1, 2, 3, 4
    };
    float data2[] = {
        2.0, 2.0, 2.0, 2.0
    };
    float expected_result[] = {
        20.0
    };

    Tensor* t1 = create_tensor(data1, shape, 1, 0);
    Tensor* t2 = create_tensor(data2, shape, 1, 0);

    // Perform batched matrix multiplication
    Tensor* result = matmul(t1, t2);

    // Check if the result is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_matmul_1d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_matmul_1d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_matmul_1d_and_3d() {
    int shape1[] = {2, 3, 4};
    int shape2[] = {4};
    float data1[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,

        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    };
    float data2[] = {
        2.0, 2.0, 2.0, 2.0
    };
    float expected_result[] = {
        20.0,  52.0,  84.0,
        116.0, 148.0, 180.0
    };

    Tensor* t1 = create_tensor(data1, shape1, 3, 0);
    Tensor* t2 = create_tensor(data2, shape2, 1, 0);

    // Perform batched matrix multiplication
    Tensor* result = matmul(t1, t2);

    // Check if the result is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_matmul_1d_and_3d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_matmul_1d_and_3d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_matmul_backward_1d_and_3d() {
    int shape1[] = {2, 3, 4};
    int shape2[] = {4};
    float data1[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,

        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    };
    float data2[] = {
        2.0, 2.0, 2.0, 2.0
    };
    float expected_grad1[] = {
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,

        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
    };
    float expected_grad2[] = {
        66.0, 72.0, 78.0, 84.0
    };

    Tensor* t1 = create_tensor(data1, shape1, 3, 1);
    Tensor* t2 = create_tensor(data2, shape2, 1, 1);

    // Perform batched matrix multiplication
    Tensor* result = matmul(t1, t2);

    // Initialize gradients as 1.0 for the output tensor
    int size = 6; // shape [2, 3]
    for (int i=0; i < size; i++) {
        result->grad[i] = 1.0;
    }
    
    result->backward_func(result);

    // Check if the result is as expected
    if (compare_tensor_data(t1->grad, expected_grad1, t1->size) && compare_tensor_data(t2->grad, expected_grad2, t2->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_matmul_backward_1d_and_3d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_matmul_backward_1d_and_3d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_matmul_backward_1d() {
    int shape[] = {4};
    float data1[] = {
        1, 2, 3, 4,
    };
    float data2[] = {
        2.0, 2.0, 2.0, 2.0
    };
    float expected_grad1[] = {
        2.0, 2.0, 2.0, 2.0,
    };
    float expected_grad2[] = {
        1, 2, 3, 4,
    };

    Tensor* t1 = create_tensor(data1, shape, 1, 1);
    Tensor* t2 = create_tensor(data2, shape, 1, 1);

    // Perform batched matrix multiplication
    Tensor* result = matmul(t1, t2);

    // Initialize gradients as 1.0 for the output tensor
    result->grad[0] = 1.0;

    result->backward_func(result);

    // Check if the result is as expected
    if (compare_tensor_data(t1->grad, expected_grad1, t1->size) && compare_tensor_data(t2->grad, expected_grad2, t2->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_matmul_backward_1d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_matmul_backward_1d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_matmul_backward_3d() {
    int shape1[] = {2, 2, 3};
    int shape2[] = {2, 3, 2};
    float data1[] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0
    };
    float data2[] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0
    };

    Tensor* t1 = create_tensor(data1, shape1, 3, 1);
    Tensor* t2 = create_tensor(data2, shape2, 3, 1);

    Tensor* result = matmul(t1, t2);

    // Initialize gradients as 1.0 for the output tensor
    int size = 8; // shape [2, 2, 2]
    for (int i=0; i < size; i++) {
        result->grad[i] = 1.0;
    }

    // Perform the backward pass
    result->backward_func(result);

    // Expected gradients (computed with pytorch)
    float expected_grad1[] = {
        3.0, 7.0, 11.0, 3.0, 7.0, 11.0,
        15.0, 19.0, 23.0, 15.0, 19.0, 23.0
    };
    float expected_grad2[] = {
        5.0, 5.0, 7.0, 7.0, 9.0, 9.0,
        17.0, 17.0, 19.0, 19.0, 21.0, 21.0
    };

    // Check if the gradients are as expected
    if (compare_tensor_data(t1->grad, expected_grad1, t1->size) && compare_tensor_data(t2->grad, expected_grad2, t2->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_matmul_backward_3d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_matmul_backward_3d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_matmul_backward_2d() {
    int shape1[] = {2, 3};
    int shape2[] = {3, 2};
    float data1[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    float data2[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};

    Tensor* t1 = create_tensor(data1, shape1, 2, 1);
    Tensor* t2 = create_tensor(data2, shape2, 2, 1);

    Tensor* result = matmul(t1, t2);

    int size = 4;
    for (int i=0; i < size; i++) {
        result->grad[i] = 1.0;
    }

    // Perform the backward pass
    result->backward_func(result);

    float expected_grad1[] = {3.0, 4.0, 5.0, 3.0, 4.0, 5.0};
    float expected_grad2[] = {2.0, 2.0, 4.0, 4.0, 6.0, 6.0};

    // Check if the gradients are as expected
    if (compare_tensor_data(t1->grad, expected_grad1, t1->size) && compare_tensor_data(t2->grad, expected_grad2, t2->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_matmul_backward_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_matmul_backward_2d:");
    }
    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_mul_scalar() {
    int data_shape1[] = {2, 3}; 
    int data_shape2[] = {1}; 
    float data1[] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
    };
    float data2[] = {
        2.0
    };
    float expected_result[] = {
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0
    };

    Tensor* t1 = create_tensor(data1, data_shape1, 2, 0);
    Tensor* t2 = create_tensor(data2, data_shape2, 1, 0);

    // Perform batched matrix multiplication
    Tensor* result = mul(t1, t2);

    // Check if the result is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_mul_scalar:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_mul_scalar:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_mul_2d() {
    int data_shape1[] = {2, 3}; 
    int data_shape2[] = {1, 3}; 
    float data1[] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
    };
    float data2[] = {
        2.0, 2.0, 2.0
    };
    float expected_result[] = {
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0
    };

    Tensor* t1 = create_tensor(data1, data_shape1, 2, 0);
    Tensor* t2 = create_tensor(data2, data_shape2, 2, 0);

    // Perform batched matrix multiplication
    Tensor* result = mul(t1, t2);

    // Check if the result is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_mul_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_mul_2d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}


void test_mul_3d() {
    int shape[] = {2, 3, 4};
    float data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,

        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    };
    float expected_result[] = {
        1,   4,   9,  16,
        25,  36,  49,  64,
        81, 100, 121, 144,

        169, 196, 225, 256,
        289, 324, 361, 400,
        441, 484, 529, 576
    };

    Tensor* t1 = create_tensor(data, shape, 3, 0);
    Tensor* t2 = create_tensor(data, shape, 3, 0);

    // Perform batched matrix multiplication
    Tensor* result = mul(t1, t2);

    // Check if the result is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_mul_3d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_mul_3d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_mul_backward_2d() {
    int shape[] = {2, 3}; 
    float data[] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
    };
    float expected_grad[] = {
        1.0, 2.0, 3.0,  
        4.0, 5.0, 6.0,
    };

    Tensor* t1 = create_tensor(data, shape, 2, 1);
    Tensor* t2 = create_tensor(data, shape, 2, 1);

    // Perform element wise multiplication
    Tensor* result = mul(t1, t2);

    int size = 6; // shape [2, 3]
    for (int i=0; i < size; i++) {
        result->grad[i] = 1.0;
    }
    result->backward_func(result);

    // Check if the result is as expected
    if (compare_tensor_data(t1->grad, expected_grad, t1->size) && compare_tensor_data(t2->grad, expected_grad, t2->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_mul_backward_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_mul_backward_2d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(result);
}

void test_relu_2d() {
    int shape1[] = {2, 3};
    float data1[] = {1.0, -2.0, -3.0, 1.0, 2.0, -3.0};

    Tensor* t1 = create_tensor(data1, shape1, 2, 0);
    Tensor* result = relu(t1);

    float expected_result[] = {1.0, 0.0, 0.0, 1.0, 2.0, 0.0};

    // Check if the gradients are as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_relu_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_relu_2d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(result);
}

void test_relu_backward_2d() {
    int shape1[] = {2, 3};
    float data1[] = {1.0, -2.0, -3.0, 1.0, 2.0, -3.0};

    Tensor* t1 = create_tensor(data1, shape1, 2, 1);
    Tensor* result = relu(t1);
    
    int size = 6;
    for (int i=0; i < size; i++) {
        result->grad[i] = 1.0;
    }
    
    // Perform the backward pass
    result->backward_func(result);
    
    float expected_grad[] = {1.0, 0.0, 0.0, 1.0, 1.0, 0.0};

    // Check if the gradients are as expected
    if (compare_tensor_data(t1->grad, expected_grad, t1->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_relu_backward_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_relu_backward_2d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(result);
}

void test_sigmoid_2d() {
    int shape[] = {2, 3};
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    Tensor* t1 = create_tensor(data, shape, 2, 0);
    Tensor* result = sigmoid(t1);

    float expected_result[] = {0.731059, 0.880797, 0.952574, 0.982014, 0.993307, 0.997527};

    // Check if the data is as expected
    if (compare_tensor_data(result->data, expected_result, result->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_sigmoid_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_sigmoid_2d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(result);
}

void test_sigmoid_backward_2d() {
    int shape[] = {2, 3}; 
    float data[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};

    Tensor* t1 = create_tensor(data, shape, 2, 1);
    Tensor* result = sigmoid(t1);

    int size = 6;
    for (int i=0; i < size; i++) {
        result->grad[i] = 1.0;
    }

    // Perform the backward pass
    result->backward_func(result);

    float expected_grad[] = {0.196612, 0.104994, 0.045177, 0.196612, 0.104994, 0.045177};

    // Check if the gradients are as expected
    if (compare_tensor_data(t1->grad, expected_grad, t1->size)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_sigmoid_backward_2d:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_sigmoid_backward_2d:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(result);
}

void test_broadcasting_valid_diff_dims() {
    int shape1[] = {2,3,2};
    int shape2[] = {1,2};
    float data1[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    float data2[] = {3.0, 4.0};

    Tensor* t1 = create_tensor(data1, shape1, 3, 0);
    Tensor* t2 = create_tensor(data2, shape2, 2, 0);

    // Check if the result is as expected
    if (is_broadcastable(t1, t2)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_broadcasting_valid_diff_dims:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_broadcasting_valid_diff_dims:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
}

void test_broadcasting_valid_same_dims() {
    int shape1[] = {2,3,2};
    int shape2[] = {2,1,1};
    float data1[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    float data2[] = {3.0, 4.0};

    Tensor* t1 = create_tensor(data1, shape1, 3, 0);
    Tensor* t2 = create_tensor(data2, shape2, 3, 0);

    // Check if the result is as expected
    if (is_broadcastable(t1, t2)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_broadcasting_valid_same_dims:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_broadcasting_valid_same_dims:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
}

void test_broadcasting_invalid_same_dims() {
    int shape1[] = {2,3,2};
    int shape2[] = {1,2,2};
    float data1[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    float data2[] = {3.0, 4.0};

    Tensor* t1 = create_tensor(data1, shape1, 3, 0);
    Tensor* t2 = create_tensor(data2, shape2, 3, 0);

    // Check if the result is as expected
    if (!is_broadcastable(t1, t2)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_broadcasting_valid_same_dims:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_broadcasting_valid_same_dims:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
}

void test_broadcasting_invalid_diff_dims() {
    int shape1[] = {2,3,2};
    int shape2[] = {2,2};
    float data1[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    float data2[] = {1.0, 1.0, 1.0, 1.0};

    Tensor* t1 = create_tensor(data1, shape1, 3, 0);
    Tensor* t2 = create_tensor(data2, shape2, 2, 0);

    // Check if the result is as expected
    if (!is_broadcastable(t1, t2)) {
        printf("%-*s PASSED\n", PADDING_WIDTH, "test_broadcasting_invalid:");
    } else {
        printf("%-*s FAILED\n", PADDING_WIDTH, "test_broadcasting_invalid:");
    }

    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
}


int main() {
    test_add_1d();
    test_add_3d();
    test_add_backward_1d();

    test_sum_3d();
    test_sum_backward_3d();

    test_reduce_sum_2d();
    test_reduce_sum_backward_2d();

    test_matmul_3d();
    test_matmul_2d();
    test_matmul_1d();
    test_matmul_1d_and_3d();
    test_matmul_backward_3d();
    test_matmul_backward_2d();
    test_matmul_backward_1d();
    test_matmul_backward_1d_and_3d();

    test_mul_scalar();
    test_mul_2d();
    test_mul_3d();
    test_mul_backward_2d();

    test_relu_2d();
    test_relu_backward_2d();

    test_sigmoid_2d();
    test_sigmoid_backward_2d();

    test_broadcasting_valid_diff_dims();
    test_broadcasting_valid_same_dims();
    test_broadcasting_invalid_same_dims();
    test_broadcasting_invalid_diff_dims();

    return 0;
}

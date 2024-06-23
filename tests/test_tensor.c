#include <stdio.h>

#include "../src/tensor_ops.h"
#include "../src/tensor.h"

/* Test for shape mismatch */
void test_add_shape_mismatch_2d() {
    int shape1[] = {2, 2}; // Shape for 2D tensor with 2x3 elements
    int shape2[] = {2, 3}; // Shape for 2D tensor with 2x2 elements
    float data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float data2[] = {7.0, 8.0, 9.0, 10.0};

    // Create tensors
    Tensor* t1 = create_tensor(data1, shape1, 2, 0);
    Tensor* t2 = create_tensor(data2, shape2, 2, 0);

    // Perform addition (should error and exit)
    Tensor* sum = add(t1, t2);

    // If it didn't exit, it failed 
    printf("test_add_shape_mismatch: FAILED\n");
    
    // Free the allocated memory
    free_tensor(t1);
    free_tensor(t2);
    free_tensor(sum);
}


int main() {
    test_add_shape_mismatch_2d(); // this test will error and exit if correct, so test it individually

    return 0;
}

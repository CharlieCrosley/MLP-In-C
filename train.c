#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "src/dataset.h"
#include "src/loss.h"
#include "src/mlp.h"
#include "src/optimizer.h"
#include "src/tensor.h"
#include "src/utility.h"
#include "src/tensor_ops.h"


/* Export the 2D points from the dataset and classify then export a grid of 2D points to 
   form a decision. This can be visualised in python or some other software*/
void export_points_for_decision_boundary(LayerList* mlp, float* dataset_points, int n_dataset_points) {
    int min_x = dataset_points[0];
    int min_y = dataset_points[1];
    int max_x = dataset_points[0];
    int max_y = dataset_points[1];

    for (int i=0; i < n_dataset_points; i++) {
        if (dataset_points[i*2] < min_x) {
            min_x = dataset_points[i*2];
        }
        if (dataset_points[i*2] > max_x) {
            max_x = dataset_points[i*2];
        }
        if (dataset_points[i*2 + 1] < min_y) {
            min_y = dataset_points[i*2 + 1];
        }
        if (dataset_points[i*2 + 1] > max_y) {
            max_y = dataset_points[i*2 + 1];
        }
    }

    int n_x_steps = 21;
    int n_y_steps = 15;
    float* x_coords = linspace(min_x-1, max_x+1, n_x_steps);
    float* y_coords = linspace(min_y-1.5, max_y+1, n_y_steps);
    int n_points = n_x_steps * n_y_steps;
    float points[n_points*2];
    int point = 0;
    for (int y=0; y < n_y_steps; y++) {
        for (int x=0; x < n_x_steps; x++) {
            points[point++] = x_coords[x];
            points[point++] = y_coords[y];
        }
    }
    
    int input_shape[2] = {n_points, 2};
    Tensor* input = create_tensor(points, input_shape, 2, 0);
    Tensor* output = forward_layers(input, mlp);

    export_2d_points_to_txt("linspace_points.txt", points, n_points);
    export_2d_points_to_txt("dataset_points.txt", dataset_points, n_dataset_points);
    export_1d_array_to_txt("linspace_labels.txt", output->data, n_points);
}

int main() {
    int random_seed = 1;
    srand(random_seed);

    int n_samples = 100;
    int input_shape[] = {n_samples, 2};
    int label_shape[] = {n_samples, 1};
    Dataset* moons = create_moons(n_samples / 2, n_samples / 2, 0.1);

    int layer_sizes[] = {16, 16, 1};
    LayerList* mlp = create_mlp(2, layer_sizes, 3);

    int n_steps = 100;
    // init lr doesnt matter here as im manually changing it in the training loop
    float init_lr = 1.0;
    SGD* optim = init_sgd(init_lr); 

    Tensor* input = create_tensor(moons->x, input_shape, 2, 0);
    Tensor* y_true = create_tensor(moons->y, label_shape, 2, 0);

    float alpha_data[1] = {1e-4};
    int alpha_shape[1] = {1};
    Tensor* alpha = create_tensor(alpha_data, alpha_shape, 1, 0);
    
    // TRAINING LOOP
    for (int i=0; i < n_steps; i++) {
        Tensor* output = forward_layers(input, mlp);
        Tensor* loss = binary_cross_entropy(output, y_true);
        
        Tensor* reg_loss; // L2 Regularization
        for (int layer=0; layer < mlp->num_layers; layer++) {
            Tensor* weights = mlp->layers[layer]->weights;
            Tensor* biases = mlp->layers[layer]->biases;
            Tensor* w_sqr = mul(weights, weights);
            Tensor* b_sqr = mul(biases, biases);
            Tensor* w_sum = reduce_sum(w_sqr);
            Tensor* b_sum = reduce_sum(b_sqr);
            reg_loss = add(w_sum, b_sum);
        }

        reg_loss = mul(alpha, reg_loss);
        loss = add(loss, reg_loss);

        Topo* topo = backward(loss);

        float accuracy = 0;
        for (int j = 0; j < y_true->size; j++) {
            accuracy += (output->data[j] >= 0.5) == (y_true->data[j] == 1);
        }
        accuracy /= y_true->size;

        // very small decay, this example works well with high lr
        float lr = init_lr - (init_lr-0.6)*i/n_steps;
        optim->update(topo, lr);
        printf("Step: %d;   Loss: %.8f   Accuracy: %.3f%%   LR: %f\n", i+1, loss->data[0], accuracy*100, lr);

        free_graph_from_topo(topo); // free intermediate tensors
    }
    
    export_points_for_decision_boundary(mlp, moons->x, moons->length);

    free_dataset(moons);
    free_layer_list(mlp);
    free_tensor(input);
    free_tensor(y_true);
    return 0;
}
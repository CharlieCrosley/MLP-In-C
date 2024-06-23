#ifndef DATASET_H
#define DATASET_H

#include "tensor.h"

#define PI 3.14159265358979323846

typedef struct Dataset {
    float* x; // data
    float* y; // labels
    int length;
} Dataset;

Dataset* create_moons(int n_samples_outer_circle, int n_samples_inner_circle, float noise);
void export_2d_points_to_txt(char* file_name, float* points, int n_points);
void export_1d_array_to_txt(char* file_name, float* array, int length);
void free_dataset(Dataset* dataset);

#endif // DATASET_H
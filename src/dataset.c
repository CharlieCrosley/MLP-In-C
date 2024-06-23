#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dataset.h"
#include "utility.h"

/* Export array of points [x1,y1,x2,y2,...] to a text file */
void export_2d_points_to_txt(char* file_name, float* points, int n_points) {
    FILE *f = fopen(file_name, "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int i=0; i < n_points*2; i+=2) {
        fprintf(f, "%f,%f\n", points[i], points[i+1]);
    }
    fclose(f);
}

void export_1d_array_to_txt(char* file_name, float* array, int length) {
    FILE *f = fopen(file_name, "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int i=0; i < length-1; i++) {
        fprintf(f, "%f\n", array[i]);
    }
    fprintf(f, "%f", array[length-1]);
    fclose(f);
}

/* Make two interleaving half circles. Implementation from sklearns make_moons.
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html */
Dataset* create_moons(int n_samples_outer_circle, int n_samples_inner_circle, float noise) {
    float* data = (float*)malloc((n_samples_outer_circle + n_samples_inner_circle) * 2 * sizeof(float)); // [N, 2]
    float* labels = (float*)malloc((n_samples_outer_circle + n_samples_inner_circle) * sizeof(float));
    if (!data || !labels) {
        fprintf(stderr, "Memory allocation failed when allocating memory for data or labels.\n");
        exit(EXIT_FAILURE);
    }

    float* linspace_values_outer = linspace(0, PI, n_samples_outer_circle);
    for (int i = 0; i < n_samples_outer_circle; i++) {
        data[i * 2] = cos(linspace_values_outer[i]) + generate_uniform_random_float(-noise, noise);
        data[i * 2 + 1] = sin(linspace_values_outer[i]) + generate_uniform_random_float(-noise, noise);
        labels[i] = 0;
    }

    float* linspace_values_inner = linspace(0, PI, n_samples_inner_circle);
    for (int i = 0; i < n_samples_inner_circle; i++) {
        data[(n_samples_outer_circle + i) * 2] = 1 - cos(linspace_values_inner[i]) + generate_uniform_random_float(-noise, noise);
        data[(n_samples_outer_circle + i) * 2 + 1] = 1 - sin(linspace_values_inner[i]) - 0.5  + generate_uniform_random_float(-noise, noise);
        labels[n_samples_outer_circle + i] = 1;
    }

    Dataset* moons = (Dataset*)malloc(sizeof(Dataset));
    if (!moons) {
        fprintf(stderr, "Memory allocation failed when allocating memory for Dataset.\n");
        exit(EXIT_FAILURE);
    }

    moons->x = data;
    moons->y = labels;
    moons->length = n_samples_outer_circle + n_samples_inner_circle;

    free(linspace_values_outer);
    free(linspace_values_inner);
    return moons;
}

void free_dataset(Dataset* dataset) {

    if (dataset) {
        if (dataset->x) {
            free(dataset->x);
            dataset->x = NULL;
        }
        if (dataset->y) {
            free(dataset->y);
            dataset->y = NULL;
        }
        free(dataset);
        dataset = NULL; // set pointer to NULL to help prevent float freeing
    }
}
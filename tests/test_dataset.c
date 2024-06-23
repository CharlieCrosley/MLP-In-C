#include <stdlib.h>

#include "../src/dataset.h"

/* Create moons dataset and export the points to be visualised by 3rd party software e.g. python*/
void test_moons_dataset() {
    Dataset* moons = create_moons(50, 50, 0.1);

    export_2d_points_to_txt("moons.txt", moons->x, moons->length);
    free(moons->x);
    free(moons->y);
    free(moons);
}

int main() {
    test_moons_dataset();

    return 0;
}
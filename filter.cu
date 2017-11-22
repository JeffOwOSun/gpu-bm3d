#include "filter.h"
#include <stdio.h>


__global__ void kernel() {
    printf("Here in kernel%s\n");
    printf("Image width: %d, height: %d\n", cu_const_params.image_width, cu_const_params.image_height);
}


void run_kernel() {
    kernel<<<1,1>>>();
}

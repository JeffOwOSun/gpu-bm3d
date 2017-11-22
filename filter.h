#ifndef __FILTER_H__
#define __FILTER_H__
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef uchar
#define uchar unsigned char
#endif
/*
 * Read-only variables for all cuda kernels. These variables
 * will be stored in the "constant" memory on GPU for fast read.
 */
struct GlobalConstants {

    unsigned int searching_window_size;
    unsigned int patch_size;
    unsigned int max_group_size;
    unsigned int distance_threshold_1;
    unsigned int distance_threshold_2;
    unsigned int stripe;
    float sigma;
    float lambda_3d;
    float beta;

    int image_channels;
    int image_width;
    int image_height;
    uchar* image_data;
};

extern __constant__ GlobalConstants cu_const_params;

void run_kernel();

#endif

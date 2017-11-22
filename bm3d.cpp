#include "bm3d.h"
#include "params.h"
#include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

/*
 * Initialize params struct
 */
Bm3d::Bm3d() {

}

Bm3d::~Bm3d() {

}

/*
 * Set first step params
 */
void Bm3d::set_fst_step_param() {

}

/*
 * Set second step params
 */
void Bm3d::set_2nd_step_param() {

}

/*
 * Set device params and allocate device memories
 */
void Bm3d::set_device_param() {

}

/*
 * Initialize image stats and allocate memory
 */
void Bm3d::copy_image_to_device(uchar *src_image,
                                int width,
                                int height,
                                int channels) {
    // set width and height
}

void Bm3d::free_device_params() {

}

/*
 * Take an image and run the algorithm to denoise.
 */
void Bm3d::denoise(uchar *src_image,
                   uchar *dst_image,
                   int width,
                   int height,
                   int channels,
                   int step,
                   int verbose = 1) {
    copy_image_to_device(src_image, width, height, channels);

    set_device_param();
    printf("Params: patch size: %zu\n", h_fst_step_params.patch_size);
    printf("Params: pdistance_threshold_1: %zu\n", h_fst_step_params.distance_threshold_1);
    // first step

    // second step

    // copy image from device to host
}

/*
 * Perform the first step denoise
 */
void Bm3d::denoise_fst_step() {

}

/*
 * Perform the second step denoise
 */
void Bm3d::denoise_2nd_step() {

}

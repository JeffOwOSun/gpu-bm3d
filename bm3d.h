#ifndef __BM3D_H__
#define __BM3D_H__

#include "params.h"
#include <string>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>


#ifndef uchar
#define uchar unsigned char
#endif


class Bm3d
{
private:
    // image
    int h_width;
    int h_height;
    int h_channels;
    uchar* d_noisy_image;               // noisy image
    uchar* d_denoised_image;            // save denoised image

    //Auxiliary arrays
    uint* d_stacks;                     //Addresses of similar patches to each reference patch of a batch
    std::vector<float*> d_numerator;    //Numerator used for aggregation
    std::vector<float*> d_denominator;  //Denminator used for aggregation
    uint* d_num_patches_in_stack;       //Number of similar patches for each referenca patch of a batch that are stored in d_stacks
    // cuComplex* d_transformed_stacks;    //3D groups of a batch
    float* d_weight;                   //Weights for aggregation
    float* d_wien_coef;             //Only for two step denoising, contains wiener coefficients
    float* d_kaiser_window;         //Kaiser window used for aggregation



    // model parameter
    Params h_fst_step_params;
    Params h_2nd_step_params;

    // device parameter

public:
    Bm3d();
    ~Bm3d();

    void set_fst_step_param();

    void set_2nd_step_param();

    void set_device_param(uchar* src_image);

    void copy_image_to_device(uchar *src_image,
                              int width,
                              int height,
                              int channels);

    void free_device_params();

    void denoise(uchar *src_image,
                 uchar *dst_image,
                 int width,
                 int height,
                 int channels,
                 int step,
                 int verbose);

    void denoise_fst_step();

    void denoise_2nd_step();

    void run_kernel();

    void test_cufft(uchar*, uchar*);

    /* data */
};

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

#endif

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

#include "stopwatch.hpp"


#ifndef uchar
#define uchar unsigned char
#endif

#define idx2(x,y,dim_x) ( (x) + ((y)*(dim_x)) )
#define idx3(x,y,z,dim_x,dim_y) ( (x) + ((y)*(dim_x)) + ((z)*(dim_x)*(dim_y)) )
#define BATCH_2D 512
#define BATCH_1D 512

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
    cufftComplex* precompute_patches;       // 3D array of precomputed 2D transformation of all the patches
    cufftComplex* d_transformed_stacks;        // 4D array to store the intermediate result, iterate patch first then width then height (Group, patch, width, height)
    cufftComplex* d_rearrange_stacks;
    Q* d_stacks;                               // 3D array of patch addresses, size is [num_ref * max_num_patches_in_stack]
    uint* d_num_patches_in_stack;       //Number of similar patches for each referenca patch that are stored in d_stacks
    std::vector<float*> d_numerator;    //Numerator used for aggregation
    std::vector<float*> d_denominator;  //Denminator used for aggregation
    // cuComplex* d_transformed_stacks;    //3D groups of a batch
    float* d_weight;                   //Weights for aggregation
    float* d_wien_coef;             //Only for two step denoising, contains wiener coefficients
    float* d_kaiser_window;         //Kaiser window used for aggregation
    int total_ref_patches;
    int total_patches;


    // model parameter
    Params h_fst_step_params;
    Params h_2nd_step_params;

    // device parameter
    cufftHandle plan;
    cufftHandle plan1D;

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

    void denoise_fst_step(uchar*);

    void denoise_2nd_step();

    void precompute_2d_transform();

    void test_fill_precompute_data(uchar*);

    void inspect_patch(uchar*, float2* h_data, int width, int height, int i, int j);

    void run_kernel();

    void test_cufft(uchar*, uchar*);

    void test_block_matching(uchar *input_image, int width = 40, int height = 40);

    void arrange_block(uchar*);

    void test_arrange_block(uchar*);

    void DFT1D();

    void do_block_matching();

    void fetch_data();
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
    int total_ref_patches;

    int image_channels;
    int image_width;
    int image_height;
    uchar* image_data;
};

#endif

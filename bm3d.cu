#include "bm3d.h"

/*
 * Read-only variables for all cuda kernels. These variables
 * will be stored in the "constant" memory on GPU for fast read.
 */
__constant__ GlobalConstants cu_const_params;


////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel() {
    printf("Here in kernel\n");
    printf("Image width: %d, height: %d\n", cu_const_params.image_width, cu_const_params.image_height);
}


////////////////////////////////////////////////////////////////////////////////////////
// Class member functions
///////////////////////////////////////////////////////////////////////////////////////

/*
 * Initialize params struct
 */
Bm3d::Bm3d() {
    h_width = 0;
    h_height = 0;
    h_channels = 0;
    d_noisy_image = NULL;
    d_denoised_image = NULL;

    d_stacks = NULL;
    d_num_patches_in_stack = NULL;
    d_weight = NULL;
    d_wien_coef = NULL;
    d_kaiser_window = NULL;
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
void Bm3d::set_device_param(uchar* src_image) {
    int deviceCount = 0;
    cudaError_t err;
    err = cudaGetDeviceCount(&deviceCount);
    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);
    std::string name;
    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // copy original image to cuda
    int size = h_width * h_height;
    cudaMalloc(&d_noisy_image, sizeof(uchar) * h_channels * size);
    cudaMemcpy(d_noisy_image, src_image, sizeof(uchar) * h_channels * size, cudaMemcpyHostToDevice);

    // Only use the generic params for now
    GlobalConstants params;
    params.image_width = h_width;
    params.image_height = h_height;
    params.image_data = d_noisy_image;
    params.image_channels = h_channels;

    params.patch_size = h_fst_step_params.patch_size;
    params.searching_window_size = h_fst_step_params.searching_window_size;
    params.stripe = h_fst_step_params.stripe;
    params.max_group_size = h_fst_step_params.max_group_size;
    params.distance_threshold_1 = h_fst_step_params.distance_threshold_1;
    params.distance_threshold_2 = h_fst_step_params.distance_threshold_2;
    params.sigma = h_fst_step_params.sigma;
    params.lambda_3d = h_fst_step_params.lambda_3d;
    params.beta = h_fst_step_params.beta;
    printf("params: %d, %d\n", params.image_width, params.image_height);

    err = cudaMemcpyToSymbol(cu_const_params, &params, sizeof(GlobalConstants));

    printf("%s\n", cudaGetErrorString(err));
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
    if (d_noisy_image) {
        cudaFree(d_noisy_image);
    }
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
    h_width = width;
    h_height = height;
    h_channels = channels;
    set_device_param(src_image);
    // first step
    test_cufft((float*)src_image);
    // second step

    // copy image from device to host
    free_device_params();
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

void Bm3d::run_kernel() {
    kernel<<<1,1>>>();
}

void Bm3d::test_cufft(float* h_data) {
    int size = h_width * h_height;
    cufftHandle plan;
    cufftReal *d_in_data;
    cufftComplex *hostOutputData = (cufftComplex*)malloc( (size / 2 + 1) * sizeof(cufftComplex));

    cudaMalloc((void**)&d_in_data, sizeof(cufftReal) * size);
    cudaMemcpy(d_in_data, (cufftReal*)h_data, sizeof(cufftReal) * size, cudaMemcpyHostToDevice);

    cufftComplex *data;
    cudaMalloc((void**)&data, sizeof(cufftComplex) * (size/2 + 1));

    if(cufftPlan2d(&plan, h_width, h_height, CUFFT_R2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT Plan error: Plan failed");
        return;
    }

    if (cufftExecR2C(plan, d_in_data, data) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }
    cudaMemcpy(hostOutputData, data, (size / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed results copy\n");
        return;
    }
    for (int i=0;i<size/2+1;i++) {
        printf("%d: (%.3f, %.3f)\n", i, hostOutputData[i].x, hostOutputData[i].y);
    }
}

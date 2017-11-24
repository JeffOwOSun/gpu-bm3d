#include "bm3d.h"

/*
 * Read-only variables for all cuda kernels. These variables
 * will be stored in the "constant" memory on GPU for fast read.
 */
__constant__ GlobalConstants cu_const_params;

float abspow2(cuComplex & a)
{
    return (a.x * a.x) + (a.y * a.y);
}

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////


__global__ void kernel() {
    printf("Here in kernel\n");
    printf("Image width: %d, height: %d\n", cu_const_params.image_width, cu_const_params.image_height);
}

__global__ void real2complex(uchar* h_data, cufftComplex *output) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int index = j*cu_const_params.image_width + i;

    if (i<cu_const_params.image_width && j<cu_const_params.image_height) {
        output[index].x = h_data[index];
        output[index].y = 0.0f;
    }
}

__global__ void complex2real(cufftComplex *data, uchar* output, int size) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int index = j*cu_const_params.image_width + i;

    if (i<cu_const_params.image_width && j<cu_const_params.image_height) {
        output[index] = data[index].x / (float)(size);
    }
}

__global__ void fill_data(uint2 *d_stacks, cufftComplex *data_stack, int size, int patch_size, int group_size) {
    for (int i=0;i<group_size;i++) {
        int b_idx = blockIdx.x * group_size + i;
        int ref_x = d_stacks[b_idx].x;
        int ref_y = d_stacks[b_idx].y;

        int start_idx = b_idx * patch_size * patch_size;
        data_stack += start_idx;
        data_stack[idx2(threadIdx.x, threadIdx.y, patch_size)].x = (float)(cu_const_params.image_data[idx2(ref_x+threadIdx.x, ref_y+threadIdx.y, cu_const_params.image_width)]);
        data_stack[idx2(threadIdx.x, threadIdx.y, patch_size)].y = 0.0f;
        printf("idx: %d, %f\n", idx2(threadIdx.x, threadIdx.y, patch_size) + start_idx, data_stack[idx2(threadIdx.x, threadIdx.y, patch_size)].x);
    }

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
    test_cufft(src_image, dst_image);
    // arrange_block(src_image);
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

void Bm3d::test_cufft(uchar* src_image, uchar* dst_image) {
    Stopwatch init_time;
    Stopwatch exec_time;
    init_time.start();
    int size = h_width * h_height;
    int patch_size = 16;
    int group_size = 4;
    int batch = size / (patch_size*patch_size*group_size);

    cufftHandle plan;
    cufftHandle plan1D;
    uchar *h_data;
    uchar *d_data;
    cudaMalloc(&d_data, sizeof(uchar) * size);

    cudaMalloc(&h_data, sizeof(uchar) * size);
    cudaMemcpy(h_data, src_image, sizeof(uchar) * size, cudaMemcpyHostToDevice);

    cufftComplex *data;
    cudaMalloc(&data, sizeof(cufftComplex) * size);
    int n[2] = {16,16};

    if(cufftPlanMany(&plan, 2, n,
                     NULL, 1, 0,
                     NULL, 1, 0,
                     CUFFT_C2C, size/256) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT Plan error: Plan failed");
        return;
    }
    if(cufftPlan1d(&plan1D, patch_size*patch_size*group_size,
                     CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT Plan error: Plan failed");
        return;
    }
    init_time.stop();
    exec_time.start();
    // get input in shape
    dim3 dimBlock(16,16);
    dim3 dimGrid(h_width/16, h_height/16);
    real2complex<<<dimGrid, dimBlock>>>(h_data, data);

    if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }

    if (cufftExecC2C(plan1D, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }

    if (cufftExecC2C(plan1D, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }

    if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }
    complex2real<<<dimGrid, dimBlock>>>(data, d_data, n[0]*n[1]);
    cudaMemcpy(dst_image, d_data, size * sizeof(uchar), cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed results copy\n");
        return;
    }
    exec_time.stop();
    printf("Init: %f\n", init_time.getSeconds());
    printf("Exec: %f\n", exec_time.getSeconds());
    for (int i=0;i<size;i++) {
        printf("%d: (%zu, %zu)\n", i, src_image[i], dst_image[i]);
    }
}

/*
 *  arrange_block - according to the stacked patch indices, fill in the transformed
 *                  data array for 2D DCT. Input is an array of uint2, every N uint2
 *                  is a group. This kernel will put each group into an continuous array
 *                  of cufftComplex num with x component to be the value, y component to be 0.f
 */
void Bm3d::arrange_block(uchar* src_image) {
    // initialize stacked patch indices which is a uint2 indices, each entry is the top
    // left indices of the patch
    int size = 8;
    int group_size = 2;
    int patch_size = 4;
    uint2 *h_stacks;
    uint2 *d_stacks;
    cufftComplex *data_stack;

    h_stacks = (uint2*)malloc(sizeof(uint2) * size);
    for (int i=0;i<size;i++) {
        h_stacks[i].x = i*size;
        h_stacks[i].y = 0;
        for (int j=0;j<patch_size;j++) {
            for (int k=0;k<patch_size;k++) {
                printf("Image id: %d, %d\n", j*h_width + i*patch_size + k, src_image[idx2(i*size + k, j, h_width)]);
            }
        }
    }
    cudaMalloc(&d_stacks, sizeof(uint2) * size);
    cudaMemcpy(d_stacks, h_stacks, sizeof(uint2) * size, cudaMemcpyHostToDevice);

    cudaMalloc(&data_stack, sizeof(cufftComplex) * size * patch_size * patch_size);

    // group per block, each pixel maps to one thread
    dim3 dimBlock(patch_size, patch_size);
    dim3 dimGrid(size/group_size);
    fill_data<<<dimGrid, dimBlock>>>(d_stacks, data_stack, size, patch_size, group_size);

}

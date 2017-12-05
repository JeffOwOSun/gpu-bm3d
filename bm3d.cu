#include "bm3d.h"

/*
 * Read-only variables for all cuda kernels. These variables
 * will be stored in the "constant" memory on GPU for fast read.
 */
__constant__ GlobalConstants cu_const_params;

#include "block_matching.cu_inl"


float norm2(cuComplex & a) {
    return (a.x * a.x) + (a.y * a.y);
}

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////


__global__ void kernel() {
    printf("Here in kernel\n");
    printf("Image width: %d, height: %d\n", cu_const_params.image_width, cu_const_params.image_height);
}

__global__ void fill_precompute_data(cufftComplex* precompute_patches) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int width = (cu_const_params.image_width - cu_const_params.patch_size + 1);
    int height = (cu_const_params.image_height - cu_const_params.patch_size + 1);
    if (i >= width || j >= height) {
        return;
    }
    // (i,j) is the top left corner of the patch
    for (int q=j;q<j+cu_const_params.patch_size;q++) {
        for (int p=i;p<i+cu_const_params.patch_size;p++) {
            // (p,q) is the image pixel
            int z = idx2(p-i,q-j,cu_const_params.patch_size);
            int index = idx3(z, i, j, cu_const_params.patch_size*cu_const_params.patch_size, width);
            precompute_patches[index].x = (float)(cu_const_params.image_data[idx2(p, q, cu_const_params.image_width)]);
            precompute_patches[index].y = 0.0f;
        }
    }
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

__global__ void complex2real(cufftComplex *data, float* output, int total_size, int trans_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= total_size) {
        return;
    }
    output[index] = data[index].x / (float)(trans_size);
}

/*
 *  normalize cufft inverse result by dividing number of elements per batch
 */
__global__ void normalize(cufftComplex *data, int size) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int index = idx2(i, j, cu_const_params.image_width);
    data[index].x = data[index].x / (float)(size);
    data[index].y = data[index].y / (float)(size);
}

/*
 * taking d_rearrange_stacks and perform thresholding. Count number of non zeros
 * Also will normalize the 1D transform result.
 */
__global__ void hard_filter(cufftComplex *d_rearrange_stacks, float *d_weight) {
    int group_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (group_id >= cu_const_params.total_ref_patches) {
        return;
    }
    int non_zero = 0;
    float threshold = cu_const_params.lambda_3d * cu_const_params.lambda_3d *
                      cu_const_params.sigma * cu_const_params.sigma *
                      blockIdx.x * blockIdx.y;
    int patch_size = cu_const_params.patch_size;
    int offset = group_id*cu_const_params.max_group_size * patch_size * patch_size;
    int norm_factor = cu_const_params.max_group_size;
    float x, y, val;
    for (int i=0; i<patch_size*patch_size*cu_const_params.max_group_size;i++) {
        x = d_rearrange_stacks[offset + i].x;
        y = d_rearrange_stacks[offset + i].y;
        x = x / norm_factor;
        y = y / norm_factor;
        val = x*x + y*y;
        if (val < threshold) {
            x = 0.0f;
            y = 0.0f;
        } else {
            ++non_zero;
        }
        d_rearrange_stacks[offset + i].x = x;
        d_rearrange_stacks[offset + i].y = y;
    }
    d_weight[group_id] = 1.0f / (float)non_zero;
}

/*
 *  Each thread maps to a group, d_transformed_stacks is organized as (w, h, patch in group)
 */
__global__ void fill_patch_major_from_source(Q* d_stacks, uint* d_num_patches_in_stack, uchar* input_data, cufftComplex* d_transformed_stacks) {
    int group_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (group_id >= cu_const_params.total_ref_patches) {
        return;
    }
    int width = cu_const_params.image_width;
    int patch_size = cu_const_params.patch_size;

    // start patch num
    int start = group_id*cu_const_params.max_group_size;
    int offset = start * patch_size * patch_size;

    for (int z=0;z<d_num_patches_in_stack[group_id];z++) {
        // fill in the actual data
        uint patch_x = d_stacks[z+start].position.x;
        uint patch_y = d_stacks[z+start].position.y;
        for (int k=0;k<patch_size*patch_size;k++) {
            int index = idx2(patch_x + (k%patch_size), patch_y + (k/patch_size), width);
            int output_index = idx2(k, z, patch_size*patch_size);
            d_transformed_stacks[output_index+offset].x = (float)(input_data[index]);
        }
    }
}

/*
 *  Each thread maps to a group
 */
__global__ void fill_stack_major_data(cufftComplex* d_transformed_stacks, cufftComplex* d_rearrange_stacks) {
    int group_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (group_id >= cu_const_params.total_ref_patches) {
        return;
    }
    int patch_size = cu_const_params.patch_size;

    // start patch num
    int start = group_id*cu_const_params.max_group_size;
    int offset = start * patch_size * patch_size;

    for (int z=0;z<cu_const_params.max_group_size;z++) {
        for (int k=0;k<patch_size*patch_size;k++) {
            int w = k % patch_size;
            int h = k / patch_size;
            int output_index = idx3(z, w, h, cu_const_params.max_group_size, patch_size);
            int index = idx2(k, z, patch_size*patch_size);
            d_rearrange_stacks[output_index + offset].x = d_transformed_stacks[index + offset].x;
            d_rearrange_stacks[output_index + offset].y = d_transformed_stacks[index + offset].y;
        }
    }
}

__global__ void fill_patch_major_from_1D_layout(cufftComplex* d_rearrange_stacks, cufftComplex* d_transformed_stacks) {
    int group_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (group_id >= cu_const_params.total_ref_patches) {
        return;
    }
    int patch_size = cu_const_params.patch_size;

    // start patch num
    int start = group_id*cu_const_params.max_group_size;
    int offset = start * patch_size * patch_size;

    for (int i=0;i<patch_size*patch_size*cu_const_params.max_group_size;i++) {
        int h = (i / (cu_const_params.max_group_size * patch_size));
        int xz = i - h*cu_const_params.max_group_size * patch_size;
        int w = xz / cu_const_params.max_group_size;
        int z = xz % cu_const_params.max_group_size;
        int index = idx3(w, h, z, patch_size, patch_size);
        d_transformed_stacks[index+offset].x = d_rearrange_stacks[i+offset].x;
        d_transformed_stacks[index+offset].y = d_rearrange_stacks[i+offset].y;
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
    total_patches = (h_width - h_fst_step_params.patch_size + 1) * (h_height - h_fst_step_params.patch_size + 1);
    total_ref_patches = ((h_width - h_fst_step_params.patch_size) / h_fst_step_params.stripe + 1) * ((h_height - h_fst_step_params.patch_size) / h_fst_step_params.stripe + 1);

    cudaGetDeviceCount(&deviceCount);
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

    cudaMalloc(&precompute_patches, sizeof(cufftComplex) * total_patches * h_fst_step_params.patch_size * h_fst_step_params.patch_size);
    cudaMalloc(&d_stacks, sizeof(Q) * total_ref_patches * h_fst_step_params.max_group_size);
    cudaMalloc(&d_num_patches_in_stack, sizeof(uint) * total_ref_patches);
    cudaMalloc(&d_transformed_stacks, sizeof(cufftComplex) * h_fst_step_params.patch_size * h_fst_step_params.patch_size * h_fst_step_params.max_group_size * total_ref_patches);
    cudaMalloc(&d_rearrange_stacks, sizeof(cufftComplex) * h_fst_step_params.patch_size * h_fst_step_params.patch_size * h_fst_step_params.max_group_size * total_ref_patches);
    cudaMalloc(&d_weight, sizeof(float) * total_ref_patches);

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
    params.total_ref_patches = total_ref_patches;

    cudaMemcpyToSymbol(cu_const_params, &params, sizeof(GlobalConstants));
    int dim2D[2] = {h_fst_step_params.patch_size, h_fst_step_params.patch_size};
    // create cufft transform plan
    if(cufftPlanMany(&plan, 2, dim2D,
                     NULL, 1, 0,
                     NULL, 1, 0,
                     CUFFT_C2C, total_ref_patches*h_fst_step_params.max_group_size) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT Plan error: Plan failed");
        return;
    }
    int batch_size = total_ref_patches * h_fst_step_params.patch_size * h_fst_step_params.patch_size;
    if(cufftPlan1d(&plan1D, h_fst_step_params.max_group_size, CUFFT_C2C, batch_size) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT Plan error: Plan failed");
        return;
    }
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
    // precompute_2d_transform();
    denoise_fst_step();
    // fetch_data();
    // test_fill_precompute_data(src_image);
    // first step
    // test_cufft(src_image, dst_image);
    // DFT1D();
    // second step

    // copy image from device to host
    free_device_params();
}

/*
 * Perform the first step denoise
 */
void Bm3d::denoise_fst_step() {
    //Block matching, each thread maps to a ref patch
    do_block_matching();
    //gather patches, convert addresses to actual data

    arrange_block(d_noisy_image);

    //perform 2d dct transform
    // Stopwatch trans;
    // trans.start();
    if (cufftExecC2C(plan, d_transformed_stacks, d_transformed_stacks, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }
    // trans.stop();
    // printf("Transform takes %.5f\n", trans.getSeconds());

    // transpose d_transformed_stacks to d_rearrange_stacks
    rearrange_to_1D_layout();
    // perform 1d transform
    if (cufftExecC2C(plan1D, d_rearrange_stacks, d_rearrange_stacks, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }
    // hard thresholding and normalize
    hard_threshold();
    // inverse 1d transform
    if (cufftExecC2C(plan1D, d_rearrange_stacks, d_rearrange_stacks, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }
    // transpose d_rearrange_stacks back to d_transformed_stacks
    rearrange_to_2D_layout();
    // inverse 2d transform
    // if (cufftExecC2C(plan, d_transformed_stacks, d_transformed_stacks, CUFFT_INVERSE) != CUFFT_SUCCESS) {
    //     fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
    //     return;
    // }
    // aggregate to single image by writing into buffer
}

/*
 * Perform the second step denoise
 */
void Bm3d::denoise_2nd_step() {
    //Block matching estimate image, each thread maps to a ref patch

    //gather patches, convert addresses to actual data

    //gather noisy image patches, convert addresses to actual data

    // perform 2d dct transform on estimate

    // perform 1d transform on estimate

    // calculate Wiener coefficient for each group

    // apply wiener coefficient to each group of transformed noisy data

    // inverse 1d transform on noisy data

    // inverse 2d transform on noisy data

    // aggregate to single image by writing into buffer
}

void Bm3d::run_kernel() {
    kernel<<<1,1>>>();
}

/*
 * precompute the 2D transform on all the patches, the data is organized as follows:
 * for patch at (i,j) with patch size = 2, then in precompute_patches, the data is
 * stored as (i,j) (i+1,j) (i,j+1) (i+1,j+1), so the dimension is height*width*4
 * we first iterate z dim, then x dim then y dim.
 */
void Bm3d::precompute_2d_transform() {
    // prepare data
    Stopwatch fill_time;
    Stopwatch tran_time;
    int patch_size = h_fst_step_params.patch_size;
    int width = (h_width - patch_size + 1);
    int height = (h_height - patch_size + 1);
    int size = width*height*patch_size*patch_size;

    float* h_data = (float*)malloc(size*sizeof(float));
    dim3 dimBlock(16,16);
    dim3 dimGrid((width+15)/16, (height+15)/16);
    fill_time.start();
    fill_precompute_data<<<dimGrid, dimBlock>>>(precompute_patches);
    fill_time.stop();
    // 2D transformation
    tran_time.start();
    for(int i=0;i<width*height*patch_size*patch_size;i+=patch_size*patch_size*BATCH_2D) {
        if (cufftExecC2C(plan, precompute_patches+i, precompute_patches+i, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
            return;
        }
    }
    tran_time.stop();
    printf("Data filling using %f\n", fill_time.getSeconds());
    printf("Exec using %f\n", tran_time.getSeconds());
}

void Bm3d::test_fill_precompute_data(uchar* src_image) {
    int patch_size = h_fst_step_params.patch_size;
    int width = (h_width - patch_size + 1);
    int height = (h_height - patch_size + 1);
    int size = width*height*patch_size*patch_size;
    float2* d_data;
    float2* h_data = (float2*)malloc(size*sizeof(float2));
    cudaMalloc(&d_data, sizeof(float2) * size);

    dim3 dimBlock(16,16);
    dim3 dimGrid((width+15)/16, (height+15)/16);
    fill_precompute_data<<<dimGrid, dimBlock>>>((cufftComplex*)d_data);
    cudaMemcpy(h_data, d_data, size * sizeof(float2), cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed results copy\n");
        return;
    }
    inspect_patch(src_image, h_data, width, height, 0,0);
}

void Bm3d::inspect_patch(uchar* src_image, float2* h_data, int width, int height, int i, int j) {
    int p2 = h_fst_step_params.patch_size*h_fst_step_params.patch_size;
    h_data = h_data + j*width*p2 + i*p2;
    for (int q=j;q<j+h_fst_step_params.patch_size;q++) {
        for (int p=i;p<i+h_fst_step_params.patch_size;p++) {
            // (p,q) is the image pixel
            printf("Image Data: %zu, test data: %0.f\n", src_image[idx2(p,q,h_width)], (*h_data).x);
            h_data++;
        }
    }
}

void Bm3d::test_cufft(uchar* src_image, uchar* dst_image) {

}

void Bm3d::test_block_matching(uchar *input_image, int width, int height) {
    // generate a dummy image
    printf("testing block_matching\n");
    if (!input_image) {
        const int img_width = 40; // a 40 by 40 checkerboard of 8x8 patch
        const int patch_width = 8;
        uchar *dummy_image = (uchar *)malloc(img_width * img_width * sizeof(uchar));
        bool isWhite = false;
        for (int y = 0; y < img_width; y += patch_width) {
            for (int x = 0; x < img_width; x += patch_width) {
                // (x, y) is the top-left corner coordinate
                for (int j = 0; j < patch_width; ++j) {
                    for (int i = 0; i < patch_width; ++i) {
                        // (x + i, y + j) is the pixel coordinate
                        int idx = idx2(x+i, y+j, img_width);
                        input_image[idx] = isWhite ? 255 : 0;
                    }
                }
                isWhite = !isWhite;
            }
        }

        // set up the parameters and consts
        input_image = dummy_image;
    }
    h_width = width;
    h_height = height;
    h_channels = 1;
    set_device_param(input_image);

    printf("width, height: %d %d\n", width, height);

    // determine how many threads we need to spawn
    const int num_ref_patches_x = (h_width - h_fst_step_params.patch_size) / h_fst_step_params.stripe + 1;
    const int total_ref_patches = ((h_width - h_fst_step_params.patch_size) / h_fst_step_params.stripe + 1) * ((h_height - h_fst_step_params.patch_size) / h_fst_step_params.stripe + 1);
    printf("total_ref_patches %d\n", total_ref_patches);
    const int total_num_threads = total_ref_patches;
    const int threads_per_block = 256;
    const int num_blocks = (total_num_threads + threads_per_block - 1) / threads_per_block;
    printf("total_num_threads %d num_block %d\n", total_ref_patches, num_blocks);

    // cudaError_t code = cudaGetLastError();
    // if (code != cudaSuccess) {
    //     fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(code));
    //     return;
    // }
    // call our block matching magic
    block_matching<<<num_blocks, threads_per_block>>>(d_stacks, d_num_patches_in_stack);
    Q *h_stacks = (Q *)malloc(sizeof(Q) * total_ref_patches * h_fst_step_params.max_group_size);
    cudaMemcpy(h_stacks, d_stacks, sizeof(Q) * total_ref_patches * h_fst_step_params.max_group_size, cudaMemcpyDeviceToHost);
    uint *h_num_patches_in_stack = (uint *)malloc(sizeof(uint) * total_ref_patches);
    cudaMemcpy(h_num_patches_in_stack, d_num_patches_in_stack, sizeof(uint) * total_ref_patches, cudaMemcpyDeviceToHost);

    // print the first stack
    const int which_stack = 13970;
    const int stack_x = which_stack % num_ref_patches_x;
    const int stack_y = which_stack / num_ref_patches_x;

    h_stacks = &h_stacks[which_stack * h_fst_step_params.max_group_size];



    printf("number of patches in stack %d: %d\n", which_stack, h_num_patches_in_stack[which_stack]);
    for (int i = 0; i < h_num_patches_in_stack[which_stack]; ++i) {
        const uint start_x = h_stacks[i].position.x;
        const uint start_y = h_stacks[i].position.y;
        printf("distance %d, x %d y %d\n", h_stacks[i].distance, start_x, start_y);
        for (int y = 0; y < h_fst_step_params.patch_size; ++y) {
            for (int x = 0; x < h_fst_step_params.patch_size; ++x) {
                const int idx = idx2( start_x + x, start_y + y, width);
                input_image[idx] = 255;
            }
        }
    }

    // set the original ref patch to 0
    for (int y = 0; y < h_fst_step_params.patch_size; ++y) {
        for (int x = 0; x < h_fst_step_params.patch_size; ++x) {
            const int idx = idx2(
                stack_x * h_fst_step_params.stripe + x,
                stack_y * h_fst_step_params.stripe + y,
                width);
            input_image[idx] = 0;
        }
    }

    // for (int y = 0; y < img_width; y += 1) {
    //     for (int x = 0; x < img_width; x += 1) {
    //         int idx = idx2(x, y, img_width);
    //         switch(input_image[idx]) {
    //             case 255:
    //                 printf("x");
    //                 break;
    //             case 127:
    //                 printf("o");
    //                 break;
    //             case 110:
    //                 printf("*");
    //                 break;
    //             default:
    //                 printf(" ");
    //         }
    //     }
    //     printf("\n");
    // }

    free_device_params();
}

/*
 *  arrange_block - according to the stacked patch indices, fetching data from the transformed
 *                  data array of 2D DCT. Input is an array of uint2, every N uint2
 *                  is a group. This kernel will put each group into an continuous array
 *                  of cufftComplex num with x component to be the value, y component to be 0.f
 */
void Bm3d::arrange_block(uchar* input_data) {
    // input: Q* each struct is a patch with top left index
    // output: d_transformed_stacks, each patch got patch*patch size continuous chunk
    // each group will be assigned a thread
    Stopwatch arrange;
    arrange.start();
    int thread_per_block = 512;
    int num_blocks = (total_ref_patches + thread_per_block - 1) / thread_per_block;
    fill_patch_major_from_source<<<num_blocks, thread_per_block>>>(d_stacks, d_num_patches_in_stack, input_data, d_transformed_stacks);
    cudaDeviceSynchronize();
    arrange.stop();
    printf("Arrange block takes %f\n", arrange.getSeconds());
}

void Bm3d::test_arrange_block(uchar *input_data) {
    int size = h_fst_step_params.patch_size * h_fst_step_params.patch_size * h_fst_step_params.max_group_size * total_ref_patches;

    Q* test_q = (Q*)malloc(sizeof(Q)*total_ref_patches * h_fst_step_params.max_group_size);
    for (int i=0;i<2*h_fst_step_params.max_group_size; i++) {
        test_q[i].position.x = i;
        test_q[i].position.y = 0;
    }
    float* h_data = (float*)malloc(sizeof(float) * size);
    float* d_data;
    cudaMalloc(&d_data, sizeof(float) * size);
    cudaMemcpy(d_stacks, test_q, sizeof(Q) * total_ref_patches * h_fst_step_params.max_group_size, cudaMemcpyHostToDevice);
    uint* h_num_patches = (uint*)calloc(total_ref_patches, sizeof(uint));
    h_num_patches[0] = h_fst_step_params.max_group_size;
    h_num_patches[1] = h_fst_step_params.max_group_size - 2;
    cudaMemcpy(d_num_patches_in_stack, h_num_patches, sizeof(uint)*total_ref_patches, cudaMemcpyHostToDevice);
    arrange_block(d_noisy_image);

    if (cufftExecC2C(plan, d_transformed_stacks, d_transformed_stacks, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }

    if (cufftExecC2C(plan, d_transformed_stacks, d_transformed_stacks, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return;
    }
    int threads_per_block = 512;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    complex2real<<<num_blocks, threads_per_block>>>(d_transformed_stacks, d_data, size, h_fst_step_params.patch_size*h_fst_step_params.patch_size);

    cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed results copy\n");
        return;
    }
    for (int i=0;i<2*h_fst_step_params.patch_size*h_fst_step_params.patch_size*h_fst_step_params.max_group_size;i++) {
        int x = i/(h_fst_step_params.patch_size*h_fst_step_params.patch_size);
        int y = 0;
        if (i % (h_fst_step_params.patch_size*h_fst_step_params.patch_size) == 0) {
            printf("Patch (%d, %d)\n", x, 0);
        }
        int z = i - x*(h_fst_step_params.patch_size*h_fst_step_params.patch_size);
        int index = idx2(x+(z%h_fst_step_params.patch_size), y+(z/h_fst_step_params.patch_size), h_width);
        printf("Transform: %.3f vs Original: %zu\n",
            h_data[i],
            input_data[index]
            );
    }
}

/*
 * fetch_data - according to the stacked patch indices, fetching data from the transformed
 *              data array of 2D DCT. Input is an array of uint2, every N uint2
 *              is a group. For each group of dim (width, height, num_patches), we will go
 *              through num_pathches first, then width then height.
 */
void Bm3d::rearrange_to_1D_layout() {
    Stopwatch fetch;
    fetch.start();
    int thread_per_block = 512;
    int num_blocks = (total_ref_patches + thread_per_block - 1) / thread_per_block;
    fill_stack_major_data<<<num_blocks, thread_per_block>>>(d_transformed_stacks, d_rearrange_stacks);
    cudaDeviceSynchronize();
    fetch.stop();
    printf("rearrange_to_1D_layout takes %.5f\n", fetch.getSeconds());
}

void Bm3d::rearrange_to_2D_layout() {
    Stopwatch fetch;
    fetch.start();
    int thread_per_block = 512;
    int num_blocks = (total_ref_patches + thread_per_block - 1) / thread_per_block;
    fill_patch_major_from_1D_layout<<<num_blocks, thread_per_block>>>(d_rearrange_stacks, d_transformed_stacks);
    cudaDeviceSynchronize();
    fetch.stop();
    printf("rearrange_to_2D_layout takes %.5f\n", fetch.getSeconds());
}

/*
 * do_block_matching - launch kernel to run block matching
 */
void Bm3d::do_block_matching() {
    // determine how many threads we need to spawn
    Stopwatch bm_time;
    bm_time.start();
    printf("total_ref_patches %d\n", total_ref_patches);
    const int total_num_threads = total_ref_patches;
    const int threads_per_block = 256;
    const int num_blocks = (total_num_threads + threads_per_block - 1) / threads_per_block;
    printf("total_num_threads %d num_block %d\n", total_ref_patches, num_blocks);
    block_matching<<<num_blocks, threads_per_block>>>(d_stacks, d_num_patches_in_stack);
    cudaDeviceSynchronize();
    bm_time.stop();
    printf("Block Matching: %f\n", bm_time.getSeconds());
}

void Bm3d::hard_threshold() {
    Stopwatch hard_threshold;
    hard_threshold.start();
    int thread_per_block = 512;
    int num_blocks = (total_ref_patches + thread_per_block - 1) / thread_per_block;
    hard_filter<<<num_blocks, thread_per_block>>>(d_rearrange_stacks, d_weight);
    cudaDeviceSynchronize();
    hard_threshold.stop();
    printf("Hard threshold takes %.5f\n", hard_threshold.getSeconds());
}

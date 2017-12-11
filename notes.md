# General Pipeline

## Step1: Get basic estimate
1. Divide the image into blocks with fix size.
2. For each reference block, search its neighbor region for similar blocks according to a distance metric.
3. For each group of blocks, apply 3D transform (2D + 1D) to get coefficients. Hard thresholding these coefficients (shrinking). Apply the inverse transform and get an estimate for each block in the group.
4. Compute the weight for each block in the group.
5. Compute the basic estimate for the image block by weighted average of the corresponding blocks in each group.

## Step2: Get final estimate
1. Use the basic estimate to get basic estimate groups by performing simmilarity test. According to the indices we also group the original image.
2. compute the Wiener shrinkage coefficients for each group and its weight.
3. Apply the coefficients to the transformed image. Then apply the inverse transform to get the estimate for each block in the group.
4. Compute the final estimate for the image block by weighted average of the corresponding blocks in each group.


## Implementation details
0. Is the two steps completely dependent on each other? And how to map the computation to threads and blocks.
1. Divide the image into blocks and precompute the transformation of the blocks
2. Which transformation (DCT/HAAR)
3. How to search similar blocks, over all the images or in a restricted neighbors
4. Each group only contains the block indices, we can allocate additional buffers for precomputed 2D tranformations, basic estimates.
5. Deal with border effects.
6. Color image.
7. use a heap to maintain the N most similar patches wrt current ref patch
8. Aggregate: to prevent potential contention due to per pixel atomic add, we could allocate an array for each patch c, each ref patch can write to c's array after atomic adding the cur index.
9. Using convolution for block matching

## Problems:
1. The initialization takes fair amount of time, cufft plan takes lots of time
2. how to do DCT or haar



## Tricks
1. .cu and .cpp are different even with the same complier (nvcc or gcc), .cu file are separate and will not link global variable togather, it does not resolve unlinked variable so `extern` will not work.
2. we could write device funtions in different file and include them in the main .cu file. This will essentially expand the include file.
3. we uses 4d array to store our results, patch_x -> patch_y -> patch -> group, but performing 1D transform needs 0.21946s, problem is that we call the kernel 28561 times. The overhead is large.
4. Try rearrange the data into

## Todo
1. 2D transform and hard thresholding. (cufft), input is a n*(# ref) uint array, with first as ref. Each entry (4 bytes) will store x,y values, each takes 2 bytes. Also we have an array to store how many patchs for each ref.




## Summary

We implemented the state-of-art image de-noising algorithm, block matching and 3D filtering (BM3D) in CUDA on NVIDIA GPU. We compared the performance of our implementation with OpenCV implementation and also referenced open source implementation in CUDA. We also show that our implementation can be used in real-time video denoising.

## Background

### Algorithm description

The block matching and 3D filtering (BM3D) algorithm is a novel method for image denoising based on collaborative filtering in transform domain. Since first proposed in 2007, BM3D has been the state-of-the-art until today. The algorithm consists of three stages:

 - Block Matching A set of similar patches of the 2D image is grouped into a 3D data array which we call group.
 - Collaborative Filtering A 3D transform is applied to the group to produce a sparse representation in transformed frequency domain, and filtered. After that, an inverse transformation is carried out to cast the filtered data back into image domain, which is a 3D array again, but noise-free
 - Reconstruct the Image The groups are redistributed into their original positions.

The algorithm runs the aforementioned 3-step procedure twice. In the first run, the noisy image is processed with hard thresholding in the sparse transform to remove low frequency part. We then apply the inverse transform of the array back to group of patches to construct the estimated image.

In the second run, with this estimated image as input, we apply the block matching and 3D transform to get the wiener filter coefficient for each group. Next, we apply the 3D transform on the original noisy image patch according to the block matching result of the estimated image. We do the elementwise multiplication of the wiener filter coefficent with the transformed data. We then apply the inverse transform of the array back to group of pathces to construct the final output image.

### Parallel option

In the above mention steps, we can easily identify a natural parallel option for block matching. We can assign each thread a reference patch and search all the patches in the search region to find similar patches and store them in the global data structure. This approach we do not need any thread or block synchronization since each thread will write to different part of the memory. The communication cost is minimized.

For the fourier transform, CUDA already has well optimized library, so we will use their APIs to apply the FFT transform.

As for reconstruction of the image from group of patches, since each patch of the original image may come from multiple groups, when we put these patches back, we need synchronization on updating the values. We assign each group a thread to perform the aggregation and we need to use `atomicAdd` to ensure no data races.

Both block matching and aggregation steps are the most time consuming steps. In our approach we will try to optimize them as much as possible. (need justify)

## Approach

### FFT

CUDA has very fast FFT library for 1D, 2D and 3D transformation. To use the CUDA FFT transform, we need to create a transformation plan first which involves allocating buffers in the GPU memory and all the initialization. After creating the plan, we can apply the plan on the data and the actual computation is very fast. (initialization time vs computation time). CUDA FFT also supports batch mode which allows us to perform a batch of transformation by calling API once and CUDA will handle all the kernel lauches optimization behind.

Fast fourier transform is crucial to the BM3D algorithm and we tried different approaches for the transformation. In the original paper and reference implementation, they both separate the 3D transformation into 2D transformation on the patches and 1D transformation across the patches. (may need picture) So we tried this approach first to compare the performance. After the block matching stage, we will have an array of patch indices and also an array to store how many patches in each of the group. We then gather the data from the images according to these indices to form a 3D matrix. The first dimension is the patch width, the second dimension is the patch height and the third dimension is the patch number. 2D FFT will then transform this array into frequency domain. To perform the 1D transform across patches for each patch pixel location (figure), we need to reorganize the array to form a 4D matrix. The first dimension is the pixel value across patches in the same group, the second dimension is the patch width, the third dimension is the patch height and the fouth dimension is the group. In this way, the values to be transformed will be continuous in memory. We then apply the 1D transform and also the hard thresholding to filter the data. During the hard thresholding, we need to record number of non-zero values for each group as the coefficient for aggregation stage. So we map each thread to a group fo patches so that each thread will work independently.

After thresholding, we apply the inverse 1D transform on the 4D matrix, which will then be reorganized back into 3D matrix for 2D inverse transform. Finally the aggregation step will put the patches back to the original image.

In this approach, we will initialize two CUFFT transform plans, 2D and 1D. 2D transform will be applied in both steps on the original images. To save the computation time, we decide to precompute the 2D transformation for each patch in the original images. After the precomputation, we just fill up the 3D data matrix based on the block matching result. This actually saves the 2D forward transformation computation time. We still need to perform the inverse transformation. In our experiment the time saving is not significant. One batch of transformation on all the image patches only cost $$. Since CUFFT library is highly optimized. Once the plan is created, the actual computation time is very short and not all pathces are needed to be transformed. Thus it isn't worthwhile to precompute the all the transformation in the initialization. In fact, during the experiment, the number of transformation for precomputing all the patches and the inverse transformation of the group patches are different. In a single API call on one plan is not going to cater two needs. So we have two options. One is to create two plans with two batch size parameter for the two cases, but we found creating a CUFFT plan will need 0.2 seconds which is 20% of the total time. We need to create as few plan as possible. Another option is to fix the batch size, so that we have only one plan and call the API several times to perform all the transformations. However, calling the API multiple times will introduce large overhead by lauching kernels. The overhead will scale linearly by number of patches. Also, if we use 2D and 1D transformation, we need to reorganize the data layout between two transformation which will introduce extra computation time.

To optimize the above problems, we turn to 3D transformation in the CUFFT library. In this approach, we only need one CUFFT transformation plan since the plan configuration is the same. We can set the batch size to the number of groups so that we only need to call the API once to perform all the transformation. Also, we do not need to change the data layout for the transformation. It turns out the execution of 3D transformation is very fast which taks only 0.006 seconds, 1.5% of the total time.




















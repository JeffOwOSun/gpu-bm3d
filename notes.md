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

## Todo
1. 2D transform and hard thresholding. (cufft), input is a n*(# ref) uint array, with first as ref. Each entry (4 bytes) will store x,y values, each takes 2 bytes. Also we have an array to store how many patchs for each ref.



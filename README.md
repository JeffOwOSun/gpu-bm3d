GPU-BM3D

*by Tianxiong Wang and Yushi Sun*

URL: [io.jowos.moe/gpu-bm3d](io.jowos.moe/gpu-bm3d)

# Final Report

## Summary

## Background

## Approach

### Block matching
We first divide the entire noisy image into a set of overlapping *reference patches*. This is done in a sliding-window manner, each patch has size of 8x8, with a stride of 3 by default. The last column and row are guaranteed to generate references, even if the dimensions of image may not be divisible by 3. 

For an example 512x512 input, a total of [(512 - 8 + 1)/3]^2 = 28561 *reference patches* are generated.

Every CUDA thread will be given one *reference patch*. Within each thread, a local window of 64 by 64 around the reference patch is searched for *q patches* that are close match to the *reference patch*. The distance metric we use in matching is the L2-distance in pixel space. This is an approximation to the original paper, where a 2D transformation and a hard-thresholding is applied before applying L2-distance in frequency space. It's much simpler in computation and easier for implementation.

![block matching image here](https://github.com/JeffOwOSun/gpu-bm3d/raw/master/assets/lena_eyebrow.png)

*An image showing three q_patches in a group. The black frame denotes the reference patch*

A maximum of 8 *q patches* are kept for each *reference patch*. After the matching, a *stack* of *num_ref_patch x max_num_patch_in_group* patches is produced, each row containing the *max_num_patch_in_group* closest *q patches* to the respective *reference patch*.

### FFT

CUDA has very fast FFT library for 1D, 2D and 3D transformation. To use the CUDA FFT transform, we need to create a transformation plan first which involves allocating buffers in the GPU memory and all the initialization. After creating the plan, we can apply the plan on the data and the actual computation is very fast (refer to the running time breakdown graph below). CUDA FFT also supports batch mode which allows us to perform a batch of transformations by calling API once and CUDA will handle the optimization of the kernel lauches behind.

Fast fourier transform is crucial to the BM3D algorithm and we tried different approaches for the transformation. In the original paper and reference implementation, they both separate the 3D transformation into 2D transformation on the patches and 1D transformation across the patches. So we tried this approach first to compare the performance. After the block matching stage, we will have an array of patch indices and also an array to store how many patches in each of the group. We then gather the data from the images according to these indices to form a 3D matrix. 

<div style="text-align:center; width=100%;">
  <img alt="3D matrix data layout" src="https://github.com/JeffOwOSun/gpu-bm3d/raw/master/assets/width_major.jpg"/>
  <img alt="4D matrix data layout" src="https://github.com/JeffOwOSun/gpu-bm3d/raw/master/assets/channel_major.jpg"/>
  <p><em>
    Left: The data layout for 3D matrix. Right: The data layout for 4D matrix
  </em></p>
</div> 

The first dimension is the patch width, the second dimension is the patch height and the third dimension is the patch number. 2D FFT will then transform this array into frequency domain. To perform the 1D transform across patches for each patch pixel location, we need to reorganize the array to form a 4D matrix. 


The first dimension is the pixel value across patches in the same group, the second dimension is the patch width, the third dimension is the patch height and the fouth dimension is the group. In this way, the values to be transformed will be continuous in memory. We then apply the 1D transform and also the hard thresholding to filter the data. During the hard thresholding, we need to record number of non-zero values for each group as the coefficient for aggregation stage. So we map each thread to a group fo patches so that each thread will work independently.

After thresholding, we apply the inverse 1D transform on the 4D matrix, which will then be reorganized back into 3D matrix for 2D inverse transform. Finally the aggregation step will put the patches back to the original image.

In this approach, we will initialize two CUFFT transform plans, 2D and 1D. 2D transform will be applied in both steps on the original images. To save the computation time, we decide to precompute the 2D transformation for each patch in the original images. After the precomputation, we just fill up the 3D data matrix based on the block matching result. This actually saves the 2D forward transformation computation time. We still need to perform the inverse transformation. In our experiment the time saving is not significant. One batch of transformation on all the image patches only cost 0.1s. Since CUFFT library is highly optimized. Once the plan is created, the actual computation time is very short and not all pathces are needed to be transformed. Thus it isn't worthwhile to precompute the all the transformation in the initialization. In fact, during the experiment, the number of transformation for precomputing all the patches and the inverse transformation of the group patches are different. In a single API call on one plan is not going to cater two needs. So we have two options. One is to create two plans with two batch size parameter for the two cases, but we found creating a CUFFT plan will need 0.2 seconds which is 20% of the total time. We need to create as few plan as possible. Another option is to fix the batch size, so that we have only one plan and call the API several times to perform all the transformations. However, calling the API multiple times will introduce large overhead by lauching kernels. The overhead will scale linearly by number of patches. Also, if we use 2D and 1D transformation, we need to reorganize the data layout between two transformation which will introduce extra computation time.

To optimize the above problems, we turn to 3D transformation in the CUFFT library. In this approach, we only need one CUFFT transformation plan since the plan configuration is the same. We can set the batch size to the number of groups so that we only need to call the API once to perform all the transformation. Also, we do not need to change the data layout for the transformation. It turns out the execution of 3D transformation is very fast which taks only 0.006 seconds, 1.5% of the total time.

### Aggregate
After the inverse transformation, the aggregation step returns the filtered patches in the stacks back to their original positions. Because there are overlaps in the patch generation, we keep a weight for each pixel and normalize it.

Each CUDA thread is assigned one stack of patches. From the previous step, a weight of *1/num_nonzero_coeff* is calculated for every stack. 
We use two image-sized global buffers, *numerator* and *denominator* for storage of pixel value and normalization factor.
For every pixel *p* in the stack, we use *atomic_add* statement to increment the corresponding *numerator* entry by *weight x p* and *denominator* by *weight*.
After the accumulation is done, a reduction step, consisting of dividing every *numerator* entry by *denominator* entry is applied to normalize the pixel values.

## Results Showcase
![showcase of lenas results here image here](https://github.com/JeffOwOSun/gpu-bm3d/raw/master/assets/lenas.jpg)
*Left: Original Lena. Mid: With noise variance=20. Right: Denoised Lena*

We show the PSNR ranking of the paper result and our implementation as below:

Noisy | Reference | Our Implementation
:---: | :---: | :---:
22.11 | 33.05 | 32.17

### A different parallelization assignment
We set out to explore a different work assignment approach from the reference open-source implementation.
The *cuda_bm3d* implementation assigns every *reference patch* to a thread block in GPU, where as we assign a single thread a block of patches in both block matching and aggregation.

![block matching time versus grid search dimension image here](https://github.com/JeffOwOSun/gpu-bm3d/raw/master/assets/block_matching_scaling.png)

*block matching time of different job assignment scheme versus grid search dimension. Here the total number of reference patches is 28561*

![block matching time versus number of total reference patches image here](https://github.com/JeffOwOSun/gpu-bm3d/raw/master/assets/block_matching_num_ref_patch.png)

*block matching time of different job assignment scheme versus number of total reference patches (adjusted by setting the reference patch stride). The matching scaling here is set to be 32 by 32*

The apparent advantage of our parallelization scheme is its simplicity of implementation. However, it's obvious that the reference implementation's per-block assignment is much faster and more robust to increase in the problem size.

One perceived bane of per-thread allocation is that memory access on warp level benefit from blocked access. A thread-level parallelization will suffer from contention and waste the shared-memory caches which are *much faster than global memory* per-documentatin.

### Running time breakdown
![running time breakdown image here](https://github.com/JeffOwOSun/gpu-bm3d/raw/master/assets/running_time_breakdown.png)
*The bar chart of running time breakdown. Our implementation strips away unnecessary computation time in transformation. Note the difference in parallelization scheme results in difference in block matching and aggregation time*

### Video denoising

# Proposal

## Summary
We are going to use CUDA to parallelize OpenCV's implementation of bm3d algorithm, and analyze our implementation against an open source incarnation available on [GitHub](https://github.com/DawyD/bm3d-gpu). We may further extend our effort to video denoising and realtime application.

## Background

The block matching and 3D filtering (BM3D) algorithm is a novel method for image denoising based on collaborative filtering in transform domain. Since first proposed in 2007, BM3D has been the state-of-the-art until today. The algorithm consists of three stages:

1. **Block Matching** A set of similar patches of the 2D image is grouped into a 3D data array which we call group.
2. **Collaborative Filtering** A 3D transform is applied to the group to produce a sparse representation in transform domain, and filtered. After that, an inverse transformation is carried out to cast the filtered data back into image domain, which is a 3D array again, but noise-free
3. **Reconstruct the Image** The groups are redistributed into their original positions.

The algorithm runs the aforementioned 3-step procedure twice. In the first run, the noisy image is processed with hard thresholding in the sparse transform to produce an original noise-free image. Then with this image as input, the same procedure is carried out with wiener filter instead of hard thresholding. The latter makes the assumption that energy spectrum of the first output is correct, and is more efficient than hard-thresholding.

![BM3D scheme](https://github.com/JeffOwOSun/gpu-bm3d/raw/master/BM3D-pipeline.png "Scheme of the BM3D algorithm")

## The Challenge
Since the BM3D algorithm is split into 3 steps. Each step depends on the result from the last step so identifying the parallel options can be difficult. To achieve good computation performance, we also may consider relax the algorithm a little bit to test the quality. There will be a quality and computation performance trade off by choosing different hyper parameters. Also copying image data back and forth between cpu and gpu is very expensive, so a clean and efficient implementation to hide memory latency is needed to achieve realtime performance. We hope to apply what we learnt from 15-618 to these state-of-art algorithms to improve our abilities to break down problems and parallel computations to achieve good performance. This will also horn our skills on writing efficient GPU code

## Resources
We also apply our algorithm onto example videos. 
We used OpenCV for decoding video and displaying denoised result.

<div style="text-align:center; width=100%;">
  <img alt="3D matrix data layout" src="https://github.com/JeffOwOSun/gpu-bm3d/raw/master/assets/original.gif"/>
  <img alt="4D matrix data layout" src="https://github.com/JeffOwOSun/gpu-bm3d/raw/master/assets/denoised.gif"/>
  <p><em>
    Left: Noisy Video. Right: Denoised Video
  </em></p>
</div> 

### Hardware

i7-8700K with 32GB of RAM and GTX1080 Ti

### Software

The serial code to optimize: OpenCV

The parallel baseline: [GitHub](https://github.com/DawyD/bm3d-gpu)

## Goals and Deliverables

### What we plan to achieve

#### The Code

Our primary goal is to come up with an efficient bm3d implementation that rivals and even outperforms the exisiting open source implemtation.

##### First version

In order to achieve this we will profile the pre-existing OpenCV implementation to find out the most computationally expensive part. We'll then devise a parallel work distribution scheme by analyzing the dependencies. If necessary, approximations of the original algorithm may be taken without affecting the overall performance. The first version of GPU implementation will mainly focus on correctness.

##### Second version

After we verify our implementation, we will analyze the bottlenecks and further optimize the performance. We'll try to improve over first version by removing duplicate calculations, reducing cache misses and lowering artifactual communications. We aim to apply our algorithm to real-time scenario if possible.

##### Demostration
We will show pre-processed outputs and compare with inputs at poster session. If time allowed, a live demo will also be implemented.

#### The Analysis
We'll also produce a detailed performance analysis with timing instructions and perf to pin-point the bottlenecks and compare our algorithm with the open source implementation. Charts and graphs of speedups, scalability, latency, effectiveness and bottlenecks of each system will be included, as well as our thoughts and reasoning.

We will answer the following key questions in our analysis

1. How well does each algorithm scale?
2. What is the reason behind the difference of performance in terms design and work distribution?
3. Can we borrow in our implementation some good ideas in the open source implementation?

### What we hope to achieve

#### V-BM3D

V-BM3D is BM3D's little brother that can handle videos. If time allowed, we will further extend our gpu acceleration to [v-bm3d](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D) algorithm that can handle temporal-spatial inputs (i.e. Videos). Our ultimate goal is come up with an efficient algorithm suitable for real-time application.

#### Realtime Application
If we were able to achieve real-time application, a live demo is also possible. We'll build a lightweight backend that receives the streams of videos taken by our phones and returns a stream of processed video. We'll also need a simple webpage to handle the showcase.

### The backup
Had things gone wrong, we would still be able to produce a detailed analysis comparing OpenCV's serial implementation with the open source GPU implementation. We will cover charts and graphs of speedups, scalability, latency, effectiveness and bottlenecks of each system, and provide our thoughts and reasoning as compensation.

## Platform Choice
We choose to implement our effort in C++/Linux. For one thing, both OpenCV and our baseline readily compile in Linux. For another, GHC machines come with pre-installed environment so we can simply kick start it. Also, since we have already gained experience with CUDA in C++, we should have a better time implementing our code.

The algorithm can be parallelized because for each 3D block it aggregates for the image, the calculation is independent beyond blocks. We choose GPU as our platform because the task is computationally intense and thus the copy/paste of the resource can be compensated.

# Checkpoint

## Summary
In the past two weeks, we were mainly reading the reference paper and understanding the BM3D algorithm. The algorithm has two dependent steps. Each step will go through a series of sub-steps: block matching, 2D transform, 1D transform with thresholding, 2D inverse transfrom then get back the estimate for original image. We profiled the openCV serial implementation on our local machine and find out the first step takes up 65% total computation time. The most time consuming part of the steps are block matching and thresholding. We also profiled our reference GPU implementation, similar observations were found. After analyzing the implementation, we think we can improve the existing one by storing precomputation and better block-thread mapping. We have not finished our own implementation of the algorithm using CUDA, but we are clear about the path.

## Goals and Deliverables
We think our initial primary goals can be achieved by producing a faster GPU implementation of BM3D algorithm. We will present the speed up between our approach vs reference implementation and also Opencv implementation on CPU. Although we are a bit behind of our initial schedule, we are clear about the path towards our goal. The realtime image processing is possible after our initial investigation. However, our time for doing V-BM3D is very limited.

## Concerns
One of the most intensive operation for this algorithm is 2D transform such as fourier transform, Bior transform or Haar wavelet tranform. The reference GPU implementation uses Cuda native support for fast fourier transform which is extensively optimized for hardware. The margin of improvement of our own design against reference implementation is small. We are not sure if our design will run faster than the reference. Also CUDA seems doesn't have support for other 2D transformation operations.

Since GHC machine doesn't have opencv package and the version of opencv on latedays is very old which can not satisfy our needs, we will have to run opencv serial implementation on our local machine. The execution time may not be comparable with the one running on GHC machine.

## Schedule

- Week 0: Run and understand the implementation of OpenCV BM3D algorithm. Profile the OpenCV implementation to find out possible optimizable portions **DONE**
- Week 1 & 2: Analyze the algorithm to find out suitable work distribution schemes. **DONE** Complete a workable GPU accelerated implementation. **UNDONE**
- ~~Week 3 & 4: Profile the implementation to find out bottlenecks. Optimize memory access and reduce extraneous communication.~~
- Week 3: Complete a workable GPU accelerated implementation.
- Week 4: Profile the implementation to find out bottlenecks. Optimize memory access and reduce extraneous communication.~~
- Week 5: Analyze performance with open source implementation and write reports.

GPU-BM3D

*by Tianxiong Wang and Yushi Sun*

URL: [io.jowos.moe/gpu-bm3d](io.jowos.moe/gpu-bm3d)

# Summary
We are going to use CUDA to parallelize OpenCV's implementation of bm3d algorithm, and analyze our implementation against an open source incarnation available on [GitHub](https://github.com/DawyD/bm3d-gpu). We may further extend our effort to video denoising and realtime application.

# Background

The block matching and 3D filtering (BM3D) algorithm is a novel method for image denoising based on collaborative filtering in transform domain. Since first proposed in 2007, BM3D has been the state-of-the-art until today. The algorithm consists of three stages:

1. **Block Matching** A set of similar patches of the 2D image is grouped into a 3D data array which we call group.
2. **Collaborative Filtering** A 3D transform is applied to the group to produce a sparse representation in transform domain, and filtered. After that, an inverse transformation is carried out to cast the filtered data back into image domain, which is a 3D array again, but noise-free
3. **Reconstruct the Image** The groups are redistributed into their original positions.

The algorithm runs the aforementioned 3-step procedure twice. In the first run, the noisy image is processed with hard thresholding in the sparse transform to produce an original noise-free image. Then with this image as input, the same procedure is carried out with wiener filter instead of hard thresholding. The latter makes the assumption that energy spectrum of the first output is correct, and is more efficient than hard-thresholding.

![BM3D scheme](https://github.com/JeffOwOSun/gpu-bm3d/raw/master/BM3D-pipeline.png "Scheme of the BM3D algorithm")

# The Challenge
Since the BM3D algorithm is split into 3 steps. Each step depends on the result from the last step so identifying the parallel options can be difficult. To achieve good computation performance, we also may consider relax the algorithm a little bit to test the quality. There will be a quality and computation performance trade off by choosing different hyper parameters. Also copying image data back and forth between cpu and gpu is very expensive, so a clean and efficient implementation to hide memory latency is needed to achieve realtime performance. We hope to apply what we learnt from 15-618 to these state-of-art algorithms to improve our abilities to break down problems and parallel computations to achieve good performance. This will also horn our skills on writing efficient GPU code

# Resources

## Hardware

GTX1080 in the GHC machines.

## Software

The serial code to optimize: OpenCV

The parallel baseline: [GitHub](https://github.com/DawyD/bm3d-gpu)

# Goals and Deliverables

## What we plan to achieve

### The Code

Our primary goal is to come up with an efficient bm3d implementation that rivals and even outperforms the exisiting open source implemtation.

#### First version

In order to achieve this we will profile the pre-existing OpenCV implementation to find out the most computationally expensive part. We'll then devise a parallel work distribution scheme by analyzing the dependencies. If necessary, approximations of the original algorithm may be taken without affecting the overall performance. The first version of GPU implementation will mainly focus on correctness.

#### Second version

After we verify our implementation, we will analyze the bottlenecks and further optimize the performance. We'll try to improve over first version by removing duplicate calculations, reducing cache misses and lowering artifactual communications. We aim to apply our algorithm to real-time scenario if possible.

#### Demostration
We will show pre-processed outputs and compare with inputs at poster session. If time allowed, a live demo will also be implemented.

### The Analysis
We'll also produce a detailed performance analysis with timing instructions and perf to pin-point the bottlenecks and compare our algorithm with the open source implementation. Charts and graphs of speedups, scalability, latency, effectiveness and bottlenecks of each system will be included, as well as our thoughts and reasoning.

We will answer the following key questions in our analysis

1. How well does each algorithm scale?
2. What is the reason behind the difference of performance in terms design and work distribution?
3. Can we borrow in our implementation some good ideas in the open source implementation?

## What we hope to achieve

### V-BM3D

V-BM3D is BM3D's little brother that can handle videos. If time allowed, we will further extend our gpu acceleration to [v-bm3d](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D) algorithm that can handle temporal-spatial inputs (i.e. Videos). Our ultimate goal is come up with an efficient algorithm suitable for real-time application.

### Realtime Application
If we were able to achieve real-time application, a live demo is also possible. We'll build a lightweight backend that receives the streams of videos taken by our phones and returns a stream of processed video. We'll also need a simple webpage to handle the showcase.

## The backup
Had things gone wrong, we would still be able to produce a detailed analysis comparing OpenCV's serial implementation with the open source GPU implementation. We will cover charts and graphs of speedups, scalability, latency, effectiveness and bottlenecks of each system, and provide our thoughts and reasoning as compensation.

# Platform Choice
We choose to implement our effort in C++/Linux. For one thing, both OpenCV and our baseline readily compile in Linux. For another, GHC machines come with pre-installed environment so we can simply kick start it. Also, since we have already gained experience with CUDA in C++, we should have a better time implementing our code.

The algorithm can be parallelized because for each 3D block it aggregates for the image, the calculation is independent beyond blocks. We choose GPU as our platform because the task is computationally intense and thus the copy/paste of the resource can be compensated.

# Schedule

- Week 0: Run and understand the implementation of OpenCV BM3D algorithm. Profile the OpenCV implementation to find out possible optimizable portions **DONE**
- Week 1 & 2: Analyze the algorithm to find out suitable work distribution schemes. **DONE** Complete a workable GPU accelerated implementation. **UNDONE**
- ~~Week 3 & 4: Profile the implementation to find out bottlenecks. Optimize memory access and reduce extraneous communication.~~
- Week 3: Complete a workable GPU accelerated implementation.
- Week 4: Profile the implementation to find out bottlenecks. Optimize memory access and reduce extraneous communication.~~
- Week 5: Analyze performance with open source implementation and write reports.


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

## Final Report Outline
1. Results to present:
    1. PSNR
    2. compare with opencv and bm3d-gpu
    3. problem size!
    4. break down chart (initialization, steps, blabla)
    5. Block matching demonstration
2. Experiments
    1. 2d+1d -> 3d
    2. pre computation
    3. thresholding.
    4. per-thread -> per-block allocation








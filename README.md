GPU-BM3D
*by Tianxiong Wang and Yushi Sun*

[io.jowos.moe/gpu-bm3d](io.jowos.moe/gpu-bm3d)

# Summary
We are going to use CUDA to parallelize OpenCV's implementation of bm3d algorithm, and analyze our implementation against an open source incarnation available on [GitHub](https://github.com/DawyD/bm3d-gpu). We may further extend our effort to video denoising and realtime application.

# Background
Dabov et.al [1] proposed in 2007 a novel method for image denoising based on collaborative filtering in transform domain. This algorithm is called block matching and 3D filtering (BM3D) and comprises three major steps. First, a set of similar 2D image fragments (i.e. blocks) is grouped into 3D data arrays that are referred to as groups. This step is referred to as block matching. Second, a 3D transform is applied to the groups, resulting in a sparse representation, that is filtered in the transform domain, and after inversion of the transform, produces the noise-free predicted blocks. This step is referred to as collaborative filtering. Finally, the predicted noise-free blocks are returned to their original positions to form the recovered image. BM3D relies on the effectiveness of the block matching and collaborative filtering to produce good denoising results. 

The BM3D algorithm is comprised of two "runs" of the aforementioned steps. First, the noisy image is processed using the block matching, collaborative filtering and aggregation, using hard thresholding in the shrinkage of the transform coefficients. This produces a basic estimate for the original noise free image. Then, using this basic estimate as input, block matching is applied, being more accurate because the noise is already significantly attenuated. The same groups formed in this basic estimate are formed in the original image. Then, the collaborative filtering and aggregation is applied, but wiener filtering is used instead of hard thresholding for the shrinkage. The wiener filter assumes that the basic estimate energy spectrum is the true energy spectrum of the image, and allows for a more efficient filtering than hard thresholding, improving the final image quality.

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
**The Code** Our primary goal is to come up with an efficient bm3d implementation that rivals and even outperforms the exisiting open source implemtation. We aim to do real-time denoising with the algorithm. We will show pre-processed outputs and compare with inputs at poster session. 
**The Analysis** We'll also produce a detailed performance analysis with timing instructions and perf to pin-point the bottlenecks and compare our algorithm with the open source implementation.  Charts and graphs of speedups, scalability, latency, effectiveness and bottlenecks of each system will be included, as well as our thoughts and reasoning. 

We will answer the following key questions in our analysis

1. How well does each algorithm scale?
2. What is the reason behind the difference of performance in terms design and work distribution?
3. Can we borrow in our implementation some good ideas in the open source implementation?

## What we hope to achieve
**V-BM3D** V-BM3D is BM3D's little brother that can handle videos. If time allowed, we will further extend our gpu acceleration to [v-bm3d](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D) algorithm that can handle temporal-spatial inputs (i.e. Videos). Our ultimate goal is come up with an efficient algorithm suitable for real-time application.
**Realtime Application** If we were able to achieve real-time application, a live demo is also possible. We'll build a lightweight backend that receives the streams of videos taken by our phones and returns a stream of processed video. We'll also need a simple webpage to handle the showcase.

## The backup
Had things gone wrong, we would still be able to produce a detailed analysis comparing OpenCV's serial implementation with the open source GPU implementation. We will cover charts and graphs of speedups, scalability, latency, effectiveness and bottlenecks of each system, and provide our thoughts and reasoning as compensation. 

# Platform Choice
We choose to implement our effort in C++/Linux. For one thing, both OpenCV and our baseline readily compile in Linux. For another, GHC machines come with pre-installed environment so we can simply kick start it. Also, since we have already gained experience with CUDA in C++, we should have a better time implementing our code.

The algorithm can be parallelized because for each 3D block it aggregates for the image, the calculation is independent beyond blocks. We choose GPU as our platform because the task is computationally intense.

# Schedule
TODO

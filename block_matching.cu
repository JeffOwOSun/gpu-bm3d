/**
 does block matching with cuda
**/
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include "params.h"

// Index conversion
#define idx2(x,y,dim_x) ( (x) + ((y)*(dim_x)) )


// compute the distance between two image patches
__device__ int distance(
    const uchar* __restrict image,
    const uint2 p,
    const uint2 q,
    const uint dim
)
{
    if (p == q) return 0;
    uint l2norm = 0;
    for (int y = 0; y < dim; ++y) {
        for (int x = 0; x < dim; ++x) {
            int diff = image[idx2(p.x+x,p.y+y)] - image[idx2(q.x+x,q.y+y)];
            l2norm += diff * diff;
        }
    }
    return l2norm;
}


__global__ void block_matching(
	Q* g_stacks, 				//OUT: Size [num_ref * max_num_patches_in_stack]
	uint* g_num_patches_in_stack,	//OUT: For each reference patch contains number of similar patches. Size [num_ref]
)
{
    uchar* image = cu_const_params.image_data;
    uint2 image_dim;
    image_dim.x = cu_const_params.image_width;
    image_dim.y = cu_const_params.image_height;
    uint patch_dim = cu_const_params.patch_size;
    uint match_radius = cu_const_params.searching_window_size;
    uint ref_stride = cu_const_params.stripe;
    uint match_stride = 1; // make it one for now
    uint max_num_patches_in_stack = cu_const_params.max_group_size;


    assert(image_dim.x >= patch_dim && image_dim.y >= patch_dim);
    // step 1: do pointer arithmetic to find out the position of this reference
    // tid is the id of reference patch
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint2 num_ref_patch = make_uint2(
        image_dim.x > patch_dim.x ?
        (image_dim.x - patch_dim.x) / ref_stride.x + 1 :
        1,
        image_dim.y > patch_dim.y ?
        (image_dim.y - patch_dim.y) / ref_stride.y + 1 :
        1
    );

    // find the x, y index of the reference patch
    uint ty = tid / num_ref_patch.x;
    if (ty >= num_ref_patch.y) return;
    uint tx = tid % num_ref_patch.x;

    // find the start position of the reference patch
    uint2 start_ref = make_uint2(tx * ref_stride, ty * ref_stride);

    // adjust edge case
    if (start_ref.x > image_dim.x - patch_dim)
        start_ref.x = image_dim.x - patch_dim;
    if (start_ref.y > image_dim.y - patch_dim)
        start_ref.y = image_dim.y - patch_dim;

    // step 2: match within the range
    uint2 start_q;
    start_q.x = start_ref.x > match_radius ? start_ref.x - match_radius : 0;
    start_q.y = start_ref.y > match_radius ? start_ref.y - match_radius : 0;

    Q* my_stack = &g_stacks[tid * max_num_patches_in_stack];
    uint &stack_count = g_num_patches_in_stack[tid];
    for (uint qy = start_q.y;
         qy <= image_dim.y - patch_dim && qy <= start_ref.y + match_radius;
         qy += match_stride) {
        for (uint qx = start_q.x;
            qx <= image_dim.x - patch_dim && qx <= start_ref.x + match_radius;
            qx += match_stride) {
            // calculate distance between q patch and ref patch
            uint2 q = make_uint2(qx,qy);
            int dist = distance(start_ref, q);
            if (stack_count < max_num_patches_in_stack) {
                // still space in the stack, just add the patch in
                my_stack[stack_count].distance = dist;
                my_stack[stack_count].position = q;
                ++stack_count;
            } else {
                // find the largest distance in stack and compare.
                int largest_dist = 0;
                uint largest_dist_idx = 0;
                for (uint i = 0; i < max_num_patches_in_stack; ++i) {
                    if (my_stack[i].distance > largest_dist) {
                        largest_dist = my_stack[i].distance;
                        largest_dist_idx = i;
                    }
                }
                // compare largest distance with new patch distance
                if (dist < largest_dist) {
                    // replace the worst patch in stack with this new patch
                    my_stack[largest_dist_idx].distance = dist;
                    my_stack[largest_dist_idx].position = q;
                }
            }
        }
    }
}

extern "C" void do_block_matching(
    Q* g_stacks,                //OUT: Size [num_ref * max_num_patches_in_stack]
    uint* g_num_patches_in_stack,   //OUT: For each reference patch contains number of similar patches. Size [num_ref]
    ) {
    block_matching<<<gridDim, blockDim>>>(
        g_stacks,
        g_num_patches_in_stack);
}

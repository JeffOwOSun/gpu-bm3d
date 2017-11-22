#ifndef __FILTER_H__
#define __FILTER_H__
#include <cuda.h>
#include <cuda_runtime.h>


__constant__ GlobalConstants cu_const_params;

void run_kernel();

#endif

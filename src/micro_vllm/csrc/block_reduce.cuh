#pragma once
#include <cuda_runtime.h>

namespace micro_vllm {
// Call from every thread of a power-of-two block. Scratch holds blockDim.x floats.
__device__ __forceinline__ float block_reduce_sum(float local_sum, float* partial_sums) {
    const int thread = threadIdx.x;
    partial_sums[thread] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            partial_sums[thread] += partial_sums[thread + stride];
        }
        __syncthreads();
    }
    return partial_sums[0];
}
}  // namespace micro_vllm

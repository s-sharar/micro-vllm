// Multiply matching features, then use the same block reduction as reduce_sum.
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "block_reduce.cuh"

namespace {
constexpr int kDotThreads = 256;

__global__ void dot_product_kernel(
    const float* a, const float* b, float* output, int64_t size
) {
    __shared__ float partial_sums[kDotThreads];
    const int thread = threadIdx.x;
    float local_sum = 0.0f;
    for (int64_t index = thread; index < size; index += blockDim.x) {
        local_sum += a[index] * b[index];
    }
    const float sum = micro_vllm::block_reduce_sum(local_sum, partial_sums);
    if (thread == 0) {
        output[0] = sum;
    }
}
}  // namespace

torch::Tensor dot_product_cuda(const torch::Tensor& a, const torch::Tensor& b) {
    const c10::cuda::CUDAGuard device_guard(a.device());
    auto output = torch::empty({}, a.options());
    const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    dot_product_kernel<<<1, kDotThreads, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), a.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

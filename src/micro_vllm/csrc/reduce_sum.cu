// One-block teaching reduction: thread-local sums, then a shared-memory tree.
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "block_reduce.cuh"

namespace {
constexpr int kReductionThreads = 256;

__global__ void reduce_sum_kernel(const float* values, float* output, int64_t size) {
    __shared__ float partial_sums[kReductionThreads];
    const int thread = threadIdx.x;
    float local_sum = 0.0f;
    for (int64_t index = thread; index < size; index += blockDim.x) {
        local_sum += values[index];
    }
    const float sum = micro_vllm::block_reduce_sum(local_sum, partial_sums);
    if (thread == 0) {
        output[0] = sum;
    }
}
}  // namespace

torch::Tensor reduce_sum_cuda(const torch::Tensor& values) {
    const c10::cuda::CUDAGuard device_guard(values.device());
    auto output = torch::empty({}, values.options());
    const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device());
    // Even an empty input launches: all local sums are zero, so the result is zero.
    reduce_sum_kernel<<<1, kReductionThreads, 0, stream>>>(
        values.data_ptr<float>(), output.data_ptr<float>(), values.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// Each GPU thread adds one pair of elements and writes one output element.
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* a, const float* b, float* output, int64_t size
) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = a[index] + b[index];
    }
}

torch::Tensor vector_add_cuda(const torch::Tensor& a, const torch::Tensor& b) {
    const c10::cuda::CUDAGuard device_guard(a.device());
    auto output = torch::empty_like(a);
    const int64_t size = a.numel();
    if (size == 0) {
        return output;  // CUDA cannot launch a grid containing zero blocks.
    }

    constexpr int threads_per_block = 256;
    const int64_t blocks = (size + threads_per_block - 1) / threads_per_block;
    const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    vector_add_kernel<<<blocks, threads_per_block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;  // Work is enqueued; the CPU does not wait for completion.
}

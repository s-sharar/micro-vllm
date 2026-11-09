// Exact scalar copies to fixed physical pages; no attention or allocation.
#include <torch/types.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace {
constexpr int kThreads = 256;

struct SourceStrides {
    int64_t head;
    int64_t token;
};

template <typename scalar_t>
__global__ void write_paged_kv_kernel(
    const scalar_t* key, const scalar_t* value, scalar_t* key_pages, scalar_t* value_pages,
    const int64_t* table, int64_t elements, int64_t tokens, int64_t heads, int64_t features,
    int64_t page_size, int64_t physical_pages, int64_t start,
    SourceStrides key_stride, SourceStrides value_stride) {
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += step) {
        const int64_t feature = index % features;
        const int64_t token = (index / features) % tokens;
        const int64_t head = index / (features * tokens);
        const int64_t position = start + token;
        const int64_t physical = table[position / page_size];
        // Table contents are an owning-cache precondition; still prevent OOB writes.
        if (physical < 0 || physical >= physical_pages) {
            continue;
        }
        const int64_t destination = ((physical * heads + head) * page_size
                                     + position % page_size) * features + feature;
        key_pages[destination] = key[head * key_stride.head + token * key_stride.token + feature];
        value_pages[destination] = value[head * value_stride.head + token * value_stride.token + feature];
    }
}
}  // namespace

void write_paged_kv_cuda(const torch::Tensor& key, const torch::Tensor& value,
                        const torch::Tensor& key_pages, const torch::Tensor& value_pages,
                        const torch::Tensor& page_table, int64_t start) {
    const c10::cuda::CUDAGuard device_guard(key.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(key.get_device());
    const int64_t elements = key.numel();
    const int blocks = static_cast<int>(std::min<int64_t>((elements - 1) / kThreads + 1, 4096));
    const SourceStrides key_stride{key.stride(1), key.stride(2)};
    const SourceStrides value_stride{value.stride(1), value.stride(2)};
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, key.scalar_type(),
                                  "write_paged_kv_cuda", [&] {
        write_paged_kv_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
            key_pages.data_ptr<scalar_t>(), value_pages.data_ptr<scalar_t>(),
            page_table.data_ptr<int64_t>(), elements, key.size(2), key.size(1), key.size(3),
            key_pages.size(2), key_pages.size(0), start, key_stride, value_stride);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

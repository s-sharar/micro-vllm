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
    int64_t row;
    int64_t head;
    int64_t token;
};

struct RowSpans {
    const int64_t* starts;  // [rows]; nullptr means one row starting at fixed_start
    const int64_t* counts;  // [rows]; nullptr means every source token is valid
    int64_t fixed_start;
    int64_t table_width;
};

template <typename scalar_t>
__global__ void write_paged_kv_kernel(
    const scalar_t* key, const scalar_t* value, scalar_t* key_pages, scalar_t* value_pages,
    const int64_t* table, RowSpans spans, int64_t elements, int64_t tokens, int64_t heads,
    int64_t features, int64_t page_size, int64_t physical_pages,
    SourceStrides key_stride, SourceStrides value_stride) {
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += step) {
        const int64_t feature = index % features;
        const int64_t token = (index / features) % tokens;
        const int64_t head = (index / (features * tokens)) % heads;
        const int64_t row = index / (features * tokens * heads);
        if (spans.counts != nullptr && token >= spans.counts[row]) {
            continue;
        }
        const int64_t position = (spans.starts == nullptr ? spans.fixed_start : spans.starts[row])
                                 + token;
        const int64_t logical = position / page_size;
        if (position < 0 || logical >= spans.table_width) {
            continue;
        }
        const int64_t physical = table[row * spans.table_width + logical];
        // Table contents are an owning-cache precondition; still prevent OOB writes.
        if (physical < 0 || physical >= physical_pages) {
            continue;
        }
        const int64_t destination = ((physical * heads + head) * page_size
                                     + position % page_size) * features + feature;
        key_pages[destination] = key[row * key_stride.row + head * key_stride.head
                                     + token * key_stride.token + feature];
        value_pages[destination] = value[row * value_stride.row + head * value_stride.head
                                         + token * value_stride.token + feature];
    }
}

void launch_write(const torch::Tensor& key, const torch::Tensor& value,
                  const torch::Tensor& key_pages, const torch::Tensor& value_pages,
                  const torch::Tensor& page_table, RowSpans spans) {
    const c10::cuda::CUDAGuard device_guard(key.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(key.get_device());
    const int64_t elements = key.numel();
    const int blocks = static_cast<int>(std::min<int64_t>((elements - 1) / kThreads + 1, 4096));
    const SourceStrides key_stride{key.stride(0), key.stride(1), key.stride(2)};
    const SourceStrides value_stride{value.stride(0), value.stride(1), value.stride(2)};
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, key.scalar_type(),
                                  "write_paged_kv_cuda", [&] {
        write_paged_kv_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
            key_pages.data_ptr<scalar_t>(), value_pages.data_ptr<scalar_t>(),
            page_table.data_ptr<int64_t>(), spans, elements, key.size(2), key.size(1),
            key.size(3), key_pages.size(2), key_pages.size(0), key_stride, value_stride);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void write_paged_kv_cuda(const torch::Tensor& key, const torch::Tensor& value,
                        const torch::Tensor& key_pages, const torch::Tensor& value_pages,
                        const torch::Tensor& page_table, int64_t start) {
    launch_write(key, value, key_pages, value_pages, page_table,
                 RowSpans{nullptr, nullptr, start, page_table.numel()});
}

void write_paged_kv_batch_cuda(const torch::Tensor& key, const torch::Tensor& value,
                              const torch::Tensor& key_pages, const torch::Tensor& value_pages,
                              const torch::Tensor& page_table, const torch::Tensor& starts,
                              const torch::Tensor& counts) {
    launch_write(key, value, key_pages, value_pages, page_table,
                 RowSpans{starts.data_ptr<int64_t>(), counts.data_ptr<int64_t>(), 0,
                          page_table.size(1)});
}

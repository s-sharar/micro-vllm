// Metadata-only validation: the owning cache validates immutable page IDs once.
#include <torch/extension.h>
#include <limits>
#include <cstdint>

void write_paged_kv_cuda(const torch::Tensor& key, const torch::Tensor& value,
                        const torch::Tensor& key_pages, const torch::Tensor& value_pages,
                        const torch::Tensor& page_table, int64_t start);

namespace {
void validate_storage(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "paged KV writes require CUDA tensors");
    TORCH_CHECK(tensor.layout() == torch::kStrided && !tensor.is_nested(),
                "paged KV writes require strided tensors");
    TORCH_CHECK(!tensor.requires_grad(), "paged KV writes do not support gradients");
    TORCH_CHECK(!tensor.is_neg(), "paged KV writes do not support lazy negative views");
}

void validate_data(const torch::Tensor& tensor, const torch::Tensor& key) {
    validate_storage(tensor);
    const auto dtype = tensor.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "paged KV writes require float16, bfloat16 or float32 dtype");
    TORCH_CHECK(dtype == key.scalar_type(), "K/V and pages must have the same dtype");
    TORCH_CHECK(tensor.device() == key.device(), "all tensors must use the same device");
}

void validate_sources(const torch::Tensor& key, const torch::Tensor& value) {
    TORCH_CHECK(key.dim() == 4 && value.dim() == 4,
                "K/V must have shape [1,H,N,D]");
    TORCH_CHECK(key.sizes() == value.sizes(), "K/V must have the same shape");
    TORCH_CHECK(key.size(0) == 1, "K/V shape requires batch size 1");
    TORCH_CHECK(key.numel() > 0, "K/V head, token and feature dimensions must be positive");
    for (const auto& source : {key, value}) {
        TORCH_CHECK(source.stride(3) == 1, "K/V require feature stride 1");
        for (int dimension = 0; dimension < 3; ++dimension) {
            TORCH_CHECK(source.stride(dimension) >= 0, "K/V require nonnegative outer strides");
        }
    }
}

void validate_pages(const torch::Tensor& key, const torch::Tensor& key_pages,
                    const torch::Tensor& value_pages) {
    TORCH_CHECK(key_pages.dim() == 4 && value_pages.dim() == 4,
                "pages must have shape [P,H,B,D]");
    TORCH_CHECK(key_pages.sizes() == value_pages.sizes(), "K/V pages must have the same shape");
    TORCH_CHECK(key_pages.numel() > 0, "page dimensions must be positive");
    TORCH_CHECK(key_pages.size(1) == key.size(1) && key_pages.size(3) == key.size(3),
                "page shape must match K/V head count and feature dimension");
    TORCH_CHECK(key_pages.is_contiguous() && value_pages.is_contiguous(),
                "K/V pages must be contiguous");
}

void validate_table(const torch::Tensor& table, const torch::Tensor& pages, int64_t start,
                    int64_t tokens) {
    validate_storage(table);
    TORCH_CHECK(table.device() == pages.device(), "all tensors must use the same device");
    TORCH_CHECK(table.scalar_type() == torch::kInt64, "page table must use int64 dtype");
    TORCH_CHECK(table.dim() == 1, "page table must be one-dimensional (1D)");
    TORCH_CHECK(table.numel() > 0, "page table must be nonempty");
    TORCH_CHECK(table.is_contiguous(), "page table must be contiguous");
    TORCH_CHECK(table.numel() <= pages.size(0), "page table cannot exceed physical page capacity");
    TORCH_CHECK(table.numel() <= std::numeric_limits<int64_t>::max() / pages.size(2),
                "page table capacity is too large");
    const int64_t capacity = table.numel() * pages.size(2);
    TORCH_CHECK(start >= 0 && start <= capacity, "start must lie within cache capacity");
    TORCH_CHECK(tokens <= capacity - start, "append exceeds cache capacity");
}

bool storage_overlaps(const torch::Tensor& left, const torch::Tensor& right) {
    const auto left_storage = left.storage();
    const auto right_storage = right.storage();
    const auto left_address = reinterpret_cast<std::uintptr_t>(left_storage.data());
    const auto right_address = reinterpret_cast<std::uintptr_t>(right_storage.data());
    if (left_storage.nbytes() == 0 || right_storage.nbytes() == 0) {
        return false;
    }
    // DLPack can create independent StorageImpl objects covering the same memory.
    // Compare byte intervals without adding their lengths (avoids address overflow).
    return left_address <= right_address
        ? right_address - left_address < left_storage.nbytes()
        : left_address - right_address < right_storage.nbytes();
}

void validate_aliases(const torch::Tensor& key, const torch::Tensor& value,
                      const torch::Tensor& key_pages, const torch::Tensor& value_pages,
                      const torch::Tensor& page_table) {
    TORCH_CHECK(!storage_overlaps(key_pages, value_pages), "K/V pages must not alias storage");
    for (const auto& source : {key, value, page_table}) {
        TORCH_CHECK(!storage_overlaps(source, key_pages) && !storage_overlaps(source, value_pages),
                    "sources and page table must not alias destination storage");
    }
}
}  // namespace

void write_paged_kv(const torch::Tensor& key, const torch::Tensor& value,
                    const torch::Tensor& key_pages, const torch::Tensor& value_pages,
                    const torch::Tensor& page_table, int64_t start) {
    for (const auto& tensor : {key, value, key_pages, value_pages}) {
        validate_data(tensor, key);
    }
    validate_sources(key, value);
    validate_pages(key, key_pages, value_pages);
    validate_table(page_table, key_pages, start, key.size(2));
    validate_aliases(key, value, key_pages, value_pages, page_table);
    write_paged_kv_cuda(key, value, key_pages, value_pages, page_table, start);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("write_paged_kv", &write_paged_kv, "Private paged K/V CUDA writes");
}

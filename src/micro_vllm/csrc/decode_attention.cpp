// Input contracts for FP32 references and FP16/FP32 contiguous/paged decode.
#include <torch/extension.h>
#include <limits>

torch::Tensor decode_attention_cuda(const torch::Tensor& query,
                                   const torch::Tensor& key, const torch::Tensor& value);
torch::Tensor decode_attention_online_cuda(const torch::Tensor& query,
                                          const torch::Tensor& key, const torch::Tensor& value);
torch::Tensor decode_attention_gqa_cuda(const torch::Tensor& query,
                                      const torch::Tensor& key, const torch::Tensor& value);

torch::Tensor decode_attention_paged_cuda(const torch::Tensor& query,
    const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& table, int64_t length);

namespace {
void validate_storage(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "decode_attention requires CUDA tensors");
    TORCH_CHECK(tensor.layout() == torch::kStrided, "decode_attention requires strided tensors");
    TORCH_CHECK(!tensor.requires_grad(), "decode_attention does not support gradients");
    TORCH_CHECK(!tensor.is_neg(), "decode_attention does not support lazy negative views");
}

void validate_tensor(const torch::Tensor& tensor) {
    validate_storage(tensor);
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "decode_attention requires float32 tensors");
    TORCH_CHECK(tensor.is_contiguous(), "decode_attention requires contiguous tensors");
}

void validate_gqa_tensor(const torch::Tensor& tensor) {
    validate_storage(tensor);
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat16,
                "GQA requires float32 or float16 tensors");

}

void validate_gqa_strides(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.stride(3) == 1, "GQA requires feature stride 1");
    for (int dimension = 0; dimension < 3; ++dimension) {
        TORCH_CHECK(tensor.stride(dimension) >= 0, "GQA requires nonnegative outer strides");
    }
}

void validate_attention_inputs(const torch::Tensor& query,
                              const torch::Tensor& key, const torch::Tensor& value) {
    validate_tensor(query);
    validate_tensor(key);
    validate_tensor(value);
    TORCH_CHECK(query.device() == key.device() && key.device() == value.device(),
                "Q, K, V must use the same device");
    TORCH_CHECK(query.dim() == 1 && key.dim() == 2 && value.dim() == 2,
                "Q must have shape [D] and K/V must have shape [L,D]");
    TORCH_CHECK(query.numel() > 0 && key.size(0) > 0 && key.size(1) > 0
                && value.size(0) > 0 && value.size(1) > 0,
                "context length and head dimension must be positive");
    TORCH_CHECK(key.sizes() == value.sizes(), "K and V must have the same shape");
    TORCH_CHECK(query.size(0) == key.size(1), "Q and K/V must share the head dimension");
}

void validate_gqa_inputs(const torch::Tensor& query,
                         const torch::Tensor& key, const torch::Tensor& value) {
    validate_gqa_tensor(query);
    validate_gqa_tensor(key);
    validate_gqa_tensor(value);
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
                "Q, K, V must use the same dtype");
    TORCH_CHECK(query.device() == key.device() && key.device() == value.device(),
                "Q, K, V must use the same device");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
                "Q must have shape [1,Hq,1,D] and K/V must have shape [1,Hkv,L,D]");
    validate_gqa_strides(query);
    validate_gqa_strides(key);
    validate_gqa_strides(value);
    TORCH_CHECK(query.size(0) == 1 && key.size(0) == 1 && value.size(0) == 1,
                "GQA currently requires batch size 1");
    TORCH_CHECK(query.numel() > 0 && key.numel() > 0 && value.numel() > 0,
                "head counts, context length, and head dimension must be positive");
    TORCH_CHECK(query.size(2) == 1, "GQA decode requires exactly one query token");
    TORCH_CHECK(key.sizes() == value.sizes(), "K and V must have the same shape");
    TORCH_CHECK(query.size(3) == key.size(3), "Q and K/V must share the head dimension");
    TORCH_CHECK(query.size(1) % key.size(1) == 0,
                "query head count must be divisible by KV head count");
}

// Page IDs are owned and validated at PagedKVCache allocation. Inspect only
// metadata here: no device-to-host synchronization or GPU table mutation.
void validate_paged_table(const torch::Tensor& table, const torch::Tensor& key,
                          const torch::Tensor& query, int64_t length) {
    validate_storage(table);
    TORCH_CHECK(table.scalar_type() == torch::kInt64, "page table requires int64");
    TORCH_CHECK(table.dim() == 1, "page table must be one-dimensional");
    TORCH_CHECK(table.is_contiguous(), "page table must be contiguous");
    TORCH_CHECK(table.device() == query.device(), "Q, K/V pages and table must use the same device");
    TORCH_CHECK(table.size(0) > 0, "page table length must be positive");
    TORCH_CHECK(table.size(0) <= key.size(0), "page table length exceeds physical pages");
    TORCH_CHECK(table.size(0) <= std::numeric_limits<int64_t>::max() / key.size(2),
                "page table capacity exceeds int64 bounds");
    TORCH_CHECK(length > 0, "context length must be positive");
    TORCH_CHECK(length <= table.size(0) * key.size(2), "context length exceeds page table capacity");
}

void validate_paged_inputs(const torch::Tensor& query, const torch::Tensor& key,
                          const torch::Tensor& value, const torch::Tensor& table, int64_t length) {
    validate_gqa_tensor(query);
    validate_gqa_tensor(key);
    validate_gqa_tensor(value);
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
                "Q and K/V pages must use the same dtype");
    TORCH_CHECK(query.device() == key.device() && key.device() == value.device(),
                "Q and K/V pages must use the same device");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
                "Q must have shape [1,Hq,1,D] and K/V pages shape [P,Hkv,B,D]");
    validate_gqa_strides(query);
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), "K/V pages must be contiguous");
    TORCH_CHECK(query.size(0) == 1, "paged attention requires batch size 1");
    TORCH_CHECK(query.numel() > 0 && key.numel() > 0 && value.numel() > 0,
                "page dimensions, head counts, and head dimension must be positive");
    TORCH_CHECK(query.size(2) == 1, "paged decode requires exactly one query token");
    TORCH_CHECK(key.sizes() == value.sizes(), "K and V pages must have the same shape");
    TORCH_CHECK(query.size(3) == key.size(3), "Q and K/V pages must share the head dimension");
    TORCH_CHECK(query.size(1) % key.size(1) == 0,
                "query head count must be divisible by KV head count");
    validate_paged_table(table, key, query, length);
}
}  // namespace

torch::Tensor decode_attention(const torch::Tensor& query,
                              const torch::Tensor& key, const torch::Tensor& value) {
    validate_attention_inputs(query, key, value);
    return decode_attention_cuda(query, key, value);
}

torch::Tensor decode_attention_online(const torch::Tensor& query,
                                     const torch::Tensor& key, const torch::Tensor& value) {
    validate_attention_inputs(query, key, value);
    return decode_attention_online_cuda(query, key, value);
}

torch::Tensor decode_attention_gqa(const torch::Tensor& query,
                                 const torch::Tensor& key, const torch::Tensor& value) {
    validate_gqa_inputs(query, key, value);
    return decode_attention_gqa_cuda(query, key, value);
}

// Internal extension entry point. The supported Python API accepts a cache,
// keeping its immutable device page table and page storage private.
torch::Tensor decode_attention_paged(const torch::Tensor& query,
    const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& table, int64_t length) {
    validate_paged_inputs(query, key, value, table, length);
    return decode_attention_paged_cuda(query, key, value, table, length);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("_decode_attention_paged", &decode_attention_paged, "Internal paged CUDA decode attention");
    module.def("decode_attention", &decode_attention, "Single-head CUDA decode attention");
    module.def("decode_attention_online", &decode_attention_online, "Online single-head CUDA decode attention");
    module.def("decode_attention_gqa", &decode_attention_gqa, "FP16/FP32 grouped-query CUDA decode attention");
}

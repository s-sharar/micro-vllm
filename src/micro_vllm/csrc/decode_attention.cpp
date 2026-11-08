// Input contracts and bindings for the FP32 decode-attention kernels.
#include <torch/extension.h>

torch::Tensor decode_attention_cuda(const torch::Tensor& query,
                                   const torch::Tensor& key, const torch::Tensor& value);
torch::Tensor decode_attention_online_cuda(const torch::Tensor& query,
                                          const torch::Tensor& key, const torch::Tensor& value);
torch::Tensor decode_attention_gqa_cuda(const torch::Tensor& query,
                                      const torch::Tensor& key, const torch::Tensor& value);


namespace {
void validate_tensor(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "decode_attention requires CUDA tensors");
    TORCH_CHECK(tensor.layout() == torch::kStrided, "decode_attention requires strided tensors");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "decode_attention requires float32 tensors");
    TORCH_CHECK(tensor.is_contiguous(), "decode_attention requires contiguous tensors");
    TORCH_CHECK(!tensor.requires_grad(), "decode_attention does not support gradients");
    TORCH_CHECK(!tensor.is_neg(), "decode_attention does not support lazy negative views");
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
    validate_tensor(query);
    validate_tensor(key);
    validate_tensor(value);
    TORCH_CHECK(query.device() == key.device() && key.device() == value.device(),
                "Q, K, V must use the same device");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
                "Q must have shape [1,Hq,1,D] and K/V must have shape [1,Hkv,L,D]");
    TORCH_CHECK(query.size(0) == 1 && key.size(0) == 1 && value.size(0) == 1,
                "FP32 GQA currently requires batch size 1");
    TORCH_CHECK(query.numel() > 0 && key.numel() > 0 && value.numel() > 0,
                "head counts, context length, and head dimension must be positive");
    TORCH_CHECK(query.size(2) == 1, "GQA decode requires exactly one query token");
    TORCH_CHECK(key.sizes() == value.sizes(), "K and V must have the same shape");
    TORCH_CHECK(query.size(3) == key.size(3), "Q and K/V must share the head dimension");
    TORCH_CHECK(query.size(1) % key.size(1) == 0,
                "query head count must be divisible by KV head count");
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("decode_attention", &decode_attention, "Single-head CUDA decode attention");
    module.def("decode_attention_online", &decode_attention_online, "Online single-head CUDA decode attention");
    module.def("decode_attention_gqa", &decode_attention_gqa, "FP32 grouped-query CUDA decode attention");
}

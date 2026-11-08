// Input contract and binding for the first single-head decode-attention kernel.
#include <torch/extension.h>

torch::Tensor decode_attention_cuda(const torch::Tensor& query,
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
}  // namespace

torch::Tensor decode_attention(const torch::Tensor& query,
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
    return decode_attention_cuda(query, key, value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("decode_attention", &decode_attention, "Single-head CUDA decode attention");
}

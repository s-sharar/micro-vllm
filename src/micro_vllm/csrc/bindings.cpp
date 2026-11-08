// CPU-side input validation and Python bindings for the CUDA exercises.
#include <torch/extension.h>

torch::Tensor vector_add_cuda(const torch::Tensor& a, const torch::Tensor& b);
torch::Tensor reduce_sum_cuda(const torch::Tensor& values);
torch::Tensor dot_product_cuda(const torch::Tensor& a, const torch::Tensor& b);

namespace {
void validate_vector(const torch::Tensor& values, const char* operation) {
    TORCH_CHECK(values.is_cuda(), operation, " requires a CUDA tensor");
    TORCH_CHECK(values.layout() == torch::kStrided, operation, " requires a strided tensor");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32, operation, " requires a float32 tensor");
    TORCH_CHECK(values.dim() == 1, operation, " requires a one-dimensional tensor");
    TORCH_CHECK(values.is_contiguous(), operation, " requires a contiguous tensor");
    TORCH_CHECK(!values.requires_grad(), operation, " does not support gradients");
    TORCH_CHECK(!values.is_neg(), operation, " does not support lazy negative views");
}

void validate_pair(const torch::Tensor& a, const torch::Tensor& b, const char* operation) {
    validate_vector(a, operation);
    validate_vector(b, operation);
    TORCH_CHECK(a.device() == b.device(), operation, " inputs must use the same device");
    TORCH_CHECK(a.sizes() == b.sizes(), operation, " inputs must have the same shape");
}
}  // namespace

torch::Tensor vector_add(const torch::Tensor& a, const torch::Tensor& b) {
    validate_pair(a, b, "vector_add");
    return vector_add_cuda(a, b);
}

torch::Tensor reduce_sum(const torch::Tensor& values) {
    validate_vector(values, "reduce_sum");
    return reduce_sum_cuda(values);
}

torch::Tensor dot_product(const torch::Tensor& a, const torch::Tensor& b) {
    validate_pair(a, b, "dot_product");
    return dot_product_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("vector_add", &vector_add, "Add two CUDA float32 vectors");
    module.def("reduce_sum", &reduce_sum, "Sum a CUDA float32 vector into a scalar");
    module.def("dot_product", &dot_product, "Dot product of two CUDA float32 vectors");
}

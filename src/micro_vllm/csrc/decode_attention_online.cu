// Contiguous/cache-prefix KV addressing for the shared online attention kernel.
#include "online_attention.cuh"

namespace {
template <typename scalar_t>
struct ContiguousKVReader {
    const scalar_t* key;
    const scalar_t* value;
    int64_t key_head_stride;
    int64_t key_token_stride;
    int64_t value_head_stride;
    int64_t value_token_stride;

    __device__ inline void select_head(int64_t head) {
        key += head * key_head_stride;
        value += head * value_head_stride;
    }
    __device__ inline scalar_t key_at(int64_t token, int64_t feature) const {
        return key[token * key_token_stride + feature];
    }
    __device__ inline scalar_t value_at(int64_t token, int64_t feature) const {
        return value[token * value_token_stride + feature];
    }
    __device__ inline bool valid_range(int64_t, int) const { return true; }
};

template <typename scalar_t>
torch::Tensor launch_contiguous(const torch::Tensor& query,
                               const torch::Tensor& key, const torch::Tensor& value) {
    const bool single_head = query.dim() == 1;
    const int64_t query_heads = single_head ? 1 : query.size(1);
    const int64_t kv_heads = single_head ? 1 : key.size(1);
    const ContiguousKVReader<scalar_t> reader{
        key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
        single_head ? 0 : key.stride(1), single_head ? key.stride(0) : key.stride(2),
        single_head ? 0 : value.stride(1), single_head ? value.stride(0) : value.stride(2)};
    return launch_online_attention<scalar_t>(query, reader, query_heads, query_heads / kv_heads,
        single_head ? key.size(0) : key.size(2), single_head ? query.size(0) : query.size(3));
}
}  // namespace

torch::Tensor decode_attention_online_cuda(const torch::Tensor& query,
                                          const torch::Tensor& key, const torch::Tensor& value) {
    return launch_contiguous<float>(query, key, value);
}

torch::Tensor decode_attention_gqa_cuda(const torch::Tensor& query,
                                      const torch::Tensor& key, const torch::Tensor& value) {
    return query.scalar_type() == torch::kFloat32
        ? launch_contiguous<float>(query, key, value)
        : launch_contiguous<c10::Half>(query, key, value);
}

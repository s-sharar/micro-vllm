// Paged KV addressing only; the attention algorithm lives in online_attention.cuh.
#include "online_attention.cuh"

namespace {
template <typename scalar_t>
struct PagedKVReader {
    const scalar_t* key;
    const scalar_t* value;
    const int64_t* page_table;
    int64_t page_count;
    int64_t page_size;
    int64_t head_dim;
    int64_t kv_heads;
    int64_t head_offset = 0;

    __device__ inline void select_head(int64_t head) {
        head_offset = head * page_size * head_dim;
    }
    __device__ inline int64_t offset(int64_t token, int64_t feature) const {
        const int64_t physical = page_table[token / page_size];
        return physical * kv_heads * page_size * head_dim
            + head_offset + (token % page_size) * head_dim + feature;
    }
    __device__ inline scalar_t key_at(int64_t token, int64_t feature) const {
        return key[offset(token, feature)];
    }
    __device__ inline scalar_t value_at(int64_t token, int64_t feature) const {
        return value[offset(token, feature)];
    }
    __device__ inline bool valid_range(int64_t start, int count) const {
        const int64_t last_page = (start + count - 1) / page_size;
        for (int64_t page = start / page_size; page <= last_page; ++page) {
            const int64_t physical = page_table[page];
            if (physical < 0 || physical >= page_count) return false;
        }
        return true;
    }
};

template <typename scalar_t>
torch::Tensor launch_paged(const torch::Tensor& query, const torch::Tensor& key,
                          const torch::Tensor& value, const torch::Tensor& table, int64_t length) {
    const PagedKVReader<scalar_t> reader{
        key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), table.data_ptr<int64_t>(),
        key.size(0), key.size(2), key.size(3), key.size(1)};
    return launch_online_attention<scalar_t>(query, reader, query.size(1),
        query.size(1) / key.size(1), length, query.size(3));
}
}  // namespace

torch::Tensor decode_attention_paged_cuda(const torch::Tensor& query,
    const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& table, int64_t length) {
    return query.scalar_type() == torch::kFloat32
        ? launch_paged<float>(query, key, value, table, length)
        : launch_paged<c10::Half>(query, key, value, table, length);
}

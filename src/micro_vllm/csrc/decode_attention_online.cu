// Online softmax in bounded 32-token chunks; output stores the normalized u/l state.
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "block_reduce.cuh"

namespace {
constexpr int kOnlineThreads = 256;
constexpr int kTokenTile = 32;

struct OnlineState {
    float maximum;
    float denominator;
    float old_mass;
    float old_output_scale;
};

// All threads participate; only thread 0 consumes the returned tile maximum.
__device__ float compute_tile_scores(
    const float* query, const float* key, int64_t head_dim,
    int64_t start, int count, float scale, float* tile_weights, float* scratch
) {
    const int thread = threadIdx.x;
    float tile_maximum = -CUDART_INF_F;
    for (int token = 0; token < count; ++token) {
        float local_dot = 0.0f;
        for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
            local_dot += query[feature] * key[(start + token) * head_dim + feature];
        }
        const float score = micro_vllm::block_reduce_sum(local_dot, scratch) * scale;
        if (thread == 0) {
            tile_weights[token] = score;
            tile_maximum = fmaxf(tile_maximum, score);
        }
        __syncthreads();  // Consume the reduction before reusing its scratch.
    }
    return tile_maximum;
}

__device__ void update_normalization(
    float tile_maximum, int count, float* tile_weights, float* scratch, OnlineState& state
) {
    const int thread = threadIdx.x;
    if (thread == 0) {
        const float next_maximum = fmaxf(state.maximum, tile_maximum);
        const float rescale = expf(state.maximum - next_maximum);
        state.old_mass = rescale * state.denominator;
        state.maximum = next_maximum;
    }
    __syncthreads();
    float local_mass = 0.0f;
    if (thread < count) {
        local_mass = expf(tile_weights[thread] - state.maximum);
        tile_weights[thread] = local_mass;
    }
    const float tile_mass = micro_vllm::block_reduce_sum(local_mass, scratch);
    if (thread == 0) {
        state.denominator = state.old_mass + tile_mass;
        state.old_output_scale = state.old_mass / state.denominator;
    }
    __syncthreads();
    if (thread < count) {
        tile_weights[thread] /= state.denominator;
    }
    __syncthreads();  // Publish every normalized tile weight before reading V.
}

__device__ void accumulate_tile_values(
    const float* value, float* output, int64_t head_dim, int64_t start,
    int count, const float* tile_weights, const OnlineState& state
) {
    for (int64_t feature = threadIdx.x; feature < head_dim; feature += blockDim.x) {
        // Keeping output = u/l avoids overflow in an unnormalized weighted numerator.
        float result = state.old_output_scale * output[feature];
        for (int token = 0; token < count; ++token) {
            result += tile_weights[token] * value[(start + token) * head_dim + feature];
        }
        output[feature] = result;
    }
}

__global__ void decode_attention_online_kernel(
    const float* query, const float* key, const float* value,
    float* output, int64_t length, int64_t head_dim, int64_t query_heads_per_kv_head
) {
    const int64_t query_head = blockIdx.x;
    const int64_t kv_head = query_head / query_heads_per_kv_head;
    const int64_t kv_offset = kv_head * length * head_dim;
    query += query_head * head_dim;
    key += kv_offset;
    value += kv_offset;
    output += query_head * head_dim;
    __shared__ float scratch[kOnlineThreads];
    __shared__ float tile_weights[kTokenTile];
    __shared__ OnlineState state;
    const int thread = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
        output[feature] = 0.0f;
    }
    if (thread == 0) {
        state.maximum = -CUDART_INF_F;
        state.denominator = 0.0f;
        state.old_mass = 0.0f;
        state.old_output_scale = 0.0f;
    }
    __syncthreads();
    for (int64_t start = 0; start < length; start += kTokenTile) {
        const int count = length - start < kTokenTile
            ? static_cast<int>(length - start) : kTokenTile;
        const float tile_maximum = compute_tile_scores(
            query, key, head_dim, start, count, scale, tile_weights, scratch);
        update_normalization(tile_maximum, count, tile_weights, scratch, state);
        accumulate_tile_values(value, output, head_dim, start, count, tile_weights, state);
        __syncthreads();  // All readers finish before tile storage/state are reused.
    }
}
torch::Tensor launch_online_attention(
    const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value,
    int64_t query_heads, int64_t heads_per_kv, int64_t length, int64_t head_dim
) {
    const c10::cuda::CUDAGuard device_guard(query.device());
    auto output = torch::empty_like(query);
    const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device());
    decode_attention_online_kernel<<<query_heads, kOnlineThreads, 0, stream>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        output.data_ptr<float>(), length, head_dim, heads_per_kv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
}  // namespace

torch::Tensor decode_attention_online_cuda(const torch::Tensor& query,
                                          const torch::Tensor& key, const torch::Tensor& value) {
    return launch_online_attention(query, key, value, 1, 1, key.size(0), query.size(0));
}

torch::Tensor decode_attention_gqa_cuda(const torch::Tensor& query,
                                      const torch::Tensor& key, const torch::Tensor& value) {
    const int64_t query_heads = query.size(1);
    const int64_t heads_per_kv = query_heads / key.size(1);
    return launch_online_attention(query, key, value, query_heads, heads_per_kv,
                                   key.size(2), query.size(3));
}

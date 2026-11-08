// Online softmax in bounded 32-token chunks; output stores the normalized u/l state.
#include <torch/types.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "block_reduce.cuh"

namespace {
constexpr int kOnlineThreads = 256;
constexpr int kTokenTile = 32;

// Element strides for the read-only input layouts; output/state are contiguous.
struct AttentionStrides {
    int64_t query_head;
    int64_t key_head;
    int64_t key_token;
    int64_t value_head;
    int64_t value_token;
};

struct OnlineState {
    float maximum;
    float denominator;
    float old_mass;
    float old_output_scale;
};

// All threads participate; only thread 0 consumes the returned tile maximum.
template <typename scalar_t>
__device__ float compute_tile_scores(
    const scalar_t* query, const scalar_t* key, int64_t head_dim,
    int64_t token_stride, int64_t start, int count, float scale, float* tile_weights, float* scratch
) {
    const int thread = threadIdx.x;
    float tile_maximum = -CUDART_INF_F;
    for (int token = 0; token < count; ++token) {
        float local_dot = 0.0f;
        for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
            local_dot += static_cast<float>(query[feature])
                       * static_cast<float>(key[(start + token) * token_stride + feature]);
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

template <typename scalar_t>
__device__ void accumulate_tile_values(
    const scalar_t* value, float* output, int64_t head_dim, int64_t token_stride, int64_t start,
    int count, const float* tile_weights, const OnlineState& state
) {
    for (int64_t feature = threadIdx.x; feature < head_dim; feature += blockDim.x) {
        // Keeping output = u/l avoids overflow in an unnormalized weighted numerator.
        float result = state.old_output_scale * output[feature];
        for (int token = 0; token < count; ++token) {
            result += tile_weights[token]
                    * static_cast<float>(value[(start + token) * token_stride + feature]);
        }
        output[feature] = result;
    }
}

template <typename scalar_t>
__global__ void decode_attention_online_kernel(
    const scalar_t* query, const scalar_t* key, const scalar_t* value,
    float* running_output, scalar_t* output, int64_t length, int64_t head_dim, int64_t query_heads_per_kv_head, AttentionStrides strides
) {
    const int64_t query_head = blockIdx.x;
    const int64_t kv_head = query_head / query_heads_per_kv_head;
    query += query_head * strides.query_head;
    key += kv_head * strides.key_head;
    value += kv_head * strides.value_head;
    running_output += query_head * head_dim;
    output += query_head * head_dim;
    __shared__ float scratch[kOnlineThreads];
    __shared__ float tile_weights[kTokenTile];
    __shared__ OnlineState state;
    const int thread = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
        running_output[feature] = 0.0f;
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
            query, key, head_dim, strides.key_token, start, count, scale, tile_weights, scratch);
        update_normalization(tile_maximum, count, tile_weights, scratch, state);
        accumulate_tile_values(value, running_output, head_dim, strides.value_token,
                               start, count, tile_weights, state);
        __syncthreads();  // All readers finish before tile storage/state are reused.
    }
    for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
        output[feature] = static_cast<scalar_t>(running_output[feature]);
    }
}

torch::Tensor launch_online_attention(
    const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value,
    int64_t query_heads, int64_t heads_per_kv, int64_t length, int64_t head_dim,
    AttentionStrides strides
) {
    const c10::cuda::CUDAGuard device_guard(query.device());
    auto output = torch::empty(query.sizes(), query.options());
    auto running_output = query.scalar_type() == torch::kFloat32
        ? output : torch::empty(query.sizes(), query.options().dtype(torch::kFloat32));
    const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device());
    if (query.scalar_type() == torch::kFloat32) {
        decode_attention_online_kernel<float><<<query_heads, kOnlineThreads, 0, stream>>>(
            query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
            running_output.data_ptr<float>(), output.data_ptr<float>(), length, head_dim, heads_per_kv, strides);
    } else {
        decode_attention_online_kernel<c10::Half><<<query_heads, kOnlineThreads, 0, stream>>>(
            query.data_ptr<c10::Half>(), key.data_ptr<c10::Half>(), value.data_ptr<c10::Half>(),
            running_output.data_ptr<float>(), output.data_ptr<c10::Half>(), length, head_dim, heads_per_kv, strides);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
}  // namespace

torch::Tensor decode_attention_online_cuda(const torch::Tensor& query,
                                          const torch::Tensor& key, const torch::Tensor& value) {
    const AttentionStrides strides{0, 0, key.stride(0), 0, value.stride(0)};
    return launch_online_attention(query, key, value, 1, 1, key.size(0), query.size(0), strides);
}

torch::Tensor decode_attention_gqa_cuda(const torch::Tensor& query,
                                      const torch::Tensor& key, const torch::Tensor& value) {
    const int64_t query_heads = query.size(1);
    const int64_t heads_per_kv = query_heads / key.size(1);
    const AttentionStrides strides{query.stride(1), key.stride(1), key.stride(2),
                                   value.stride(1), value.stride(2)};
    return launch_online_attention(query, key, value, query_heads, heads_per_kv,
                                   key.size(2), query.size(3), strides);
}

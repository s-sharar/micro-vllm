#pragma once
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

struct OnlineState {
    float maximum;
    float denominator;
    float old_mass;
    float old_output_scale;
};

// All threads participate; only thread 0 consumes the returned tile maximum.
template <typename scalar_t, typename KVReader>
__device__ inline float compute_tile_scores(
    const scalar_t* query, const KVReader& reader, int64_t head_dim,
    int64_t start, int count, float scale, float* tile_weights, float* scratch
) {
    const int thread = threadIdx.x;
    float tile_maximum = -CUDART_INF_F;
    for (int token = 0; token < count; ++token) {
        float local_dot = 0.0f;
        for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
            local_dot += static_cast<float>(query[feature])
                       * static_cast<float>(reader.key_at(start + token, feature));
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

__device__ inline void update_normalization(
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

template <typename KVReader>
__device__ inline void accumulate_tile_values(
    const KVReader& reader, float* output, int64_t head_dim, int64_t start,
    int count, const float* tile_weights, const OnlineState& state
) {
    for (int64_t feature = threadIdx.x; feature < head_dim; feature += blockDim.x) {
        // Keeping output = u/l avoids overflow in an unnormalized weighted numerator.
        float result = state.old_output_scale * output[feature];
        for (int token = 0; token < count; ++token) {
            result += tile_weights[token]
                    * static_cast<float>(reader.value_at(start + token, feature));
        }
        output[feature] = result;
    }
}

__device__ inline void initialize_online_state(
    float* running_output, int64_t head_dim, OnlineState& state
) {
    const int thread = threadIdx.x;
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
}

template <typename scalar_t, typename KVReader>
__global__ void decode_attention_online_kernel(
    const scalar_t* query, KVReader reader,
    float* running_output, scalar_t* output, int64_t length, int64_t head_dim, int64_t query_heads_per_kv_head, int64_t query_head_stride
) {
    const int64_t query_head = blockIdx.x;
    const int64_t kv_head = query_head / query_heads_per_kv_head;
    query += query_head * query_head_stride;
    reader.select_head(kv_head);
    running_output += query_head * head_dim;
    output += query_head * head_dim;
    __shared__ float scratch[kOnlineThreads];
    __shared__ float tile_weights[kTokenTile];
    __shared__ OnlineState state;
    const int thread = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    initialize_online_state(running_output, head_dim, state);
    for (int64_t start = 0; start < length; start += kTokenTile) {
        const int count = length - start < kTokenTile
            ? static_cast<int>(length - start) : kTokenTile;
        // This condition is block-uniform: all threads inspect the same immutable table.
        // The prior tile's barrier has completed before any invalid page can be read.
        if (!reader.valid_range(start, count)) {
            for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
                output[feature] = static_cast<scalar_t>(CUDART_NAN_F);
            }
            return;
        }
        const float tile_maximum = compute_tile_scores(
            query, reader, head_dim, start, count, scale, tile_weights, scratch);
        update_normalization(tile_maximum, count, tile_weights, scratch, state);
        accumulate_tile_values(reader, running_output, head_dim,
                               start, count, tile_weights, state);
        __syncthreads();  // All readers finish before tile storage/state are reused.
    }
    for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
        output[feature] = static_cast<scalar_t>(running_output[feature]);
    }
}

// The same launch/state allocation serves both physical KV layouts.
template <typename scalar_t, typename KVReader>
torch::Tensor launch_online_attention(
    const torch::Tensor& query, KVReader reader, int64_t query_heads,
    int64_t heads_per_kv, int64_t length, int64_t head_dim
) {
    const c10::cuda::CUDAGuard device_guard(query.device());
    auto output = torch::empty(query.sizes(), query.options());
    auto running_output = query.scalar_type() == torch::kFloat32
        ? output : torch::empty(query.sizes(), query.options().dtype(torch::kFloat32));
    const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device());
    const int64_t query_head_stride = query.dim() == 1 ? 0 : query.stride(1);
    decode_attention_online_kernel<scalar_t><<<query_heads, kOnlineThreads, 0, stream>>>(
        query.data_ptr<scalar_t>(), reader, running_output.data_ptr<float>(),
        output.data_ptr<scalar_t>(), length, head_dim, heads_per_kv, query_head_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
}  // namespace

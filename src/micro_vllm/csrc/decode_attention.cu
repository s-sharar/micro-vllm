// First dense decode kernel: score workspace, stable softmax, weighted values.
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "block_reduce.cuh"

namespace {
constexpr int kAttentionThreads = 256;

__global__ void decode_attention_kernel(
    const float* query, const float* key, const float* value,
    float* weights, float* output, int64_t length, int64_t head_dim
) {
    __shared__ float partial_sums[kAttentionThreads];
    __shared__ float score_maximum;
    const int thread = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float maximum = -CUDART_INF_F;

    for (int64_t token = 0; token < length; ++token) {
        float local_dot = 0.0f;
        for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
            local_dot += query[feature] * key[token * head_dim + feature];
        }
        const float score = micro_vllm::block_reduce_sum(local_dot, partial_sums) * scale;
        if (thread == 0) {
            weights[token] = score;
            maximum = fmaxf(maximum, score);
        }
        // Every thread consumes this reduction before its shared scratch is reused.
        __syncthreads();
    }
    if (thread == 0) {
        score_maximum = maximum;
    }
    __syncthreads();

    float local_weight_sum = 0.0f;
    for (int64_t token = thread; token < length; token += blockDim.x) {
        const float weight = expf(weights[token] - score_maximum);
        weights[token] = weight;
        local_weight_sum += weight;
    }
    // The helper's barriers also publish all workspace weights before V is read.
    const float denominator = micro_vllm::block_reduce_sum(local_weight_sum, partial_sums);
    // Normalize before multiplying V, so a finite weighted average avoids a huge numerator.
    for (int64_t token = thread; token < length; token += blockDim.x) {
        weights[token] /= denominator;
    }
    __syncthreads();

    for (int64_t feature = thread; feature < head_dim; feature += blockDim.x) {
        float weighted_value = 0.0f;
        for (int64_t token = 0; token < length; ++token) {
            weighted_value += weights[token] * value[token * head_dim + feature];
        }
        output[feature] = weighted_value;
    }
}
}  // namespace

torch::Tensor decode_attention_cuda(const torch::Tensor& query,
                                   const torch::Tensor& key, const torch::Tensor& value) {
    const c10::cuda::CUDAGuard device_guard(query.device());
    auto output = torch::empty_like(query);
    auto workspace = torch::empty({key.size(0)}, query.options());
    const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device());
    decode_attention_kernel<<<1, kAttentionThreads, 0, stream>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        workspace.data_ptr<float>(), output.data_ptr<float>(), key.size(0), query.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

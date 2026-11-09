"""CUDA decode attention and single-head references, independent of the exercises."""
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def _load_extension() -> ModuleType:
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent / 'csrc'
    return load(
        name='_micro_vllm_decode_attention',
        sources=[str(source_dir / 'decode_attention.cpp'),
                 str(source_dir / 'decode_attention.cu'),
                 str(source_dir / 'decode_attention_online.cu'),
                 str(source_dir / 'decode_attention_paged.cu')],
        extra_cflags=['-O2'],
        extra_cuda_cflags=['-O2'],
        with_cuda=True,
    )


def decode_attention(query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor) -> torch.Tensor:
    """Compute softmax(K @ q / sqrt(D)) @ V for the latest query of one head.

    q is [D], K/V are [L,D], and the result is [D]. L and D must be positive.
    All L rows must contain valid context, including the current token's K/V;
    no future positions or padding may be included. Q/K already have RoPE.
    Inputs are contiguous strided CUDA float32 tensors on the same device,
    without gradients or lazy negative views. Inputs are unchanged.

    One fused launch materializes L scores, applies stable softmax in FP32,
    and accumulates values in FP32 on the current stream. Import is lazy; first
    use builds only attention sources. This initial version is a correctness
    baseline; decode_attention_online provides the chunked alternative. The
    single-head references stay FP32; decode_attention_gqa supports FP16/FP32.
    GQA accepts outer strides and is used by model decode dispatch.
    """
    return _load_extension().decode_attention(query, key, value)


def decode_attention_online(query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor) -> torch.Tensor:
    """Single-head FP32 decode attention with online softmax in 32-token chunks.

    Inputs follow decode_attention's [D], [L,D], [L,D] contract. Scores occupy
    bounded shared memory instead of an L-element GPU workspace. The kernel
    tracks maximum m, denominator l, and the normalized weighted output u/l;
    each new chunk rescales prior contributions before adding its own. Keeping
    the output normalized reduces overflow risk with large finite V values.

    The required [D] output also stores running output between chunks. FP32
    operation order can differ from the materialized baseline. This first
    online implementation is a correctness baseline, not a speed claim.
    """
    return _load_extension().decode_attention_online(query, key, value)


def decode_attention_gqa(query: torch.Tensor, key: torch.Tensor,
                         value: torch.Tensor) -> torch.Tensor:
    """Online decode for Q [1,Hq,1,D] and original K/V [1,Hkv,L,D].

    Returns the same shape as Q. Batch size is 1 and only the newest query is
    processed. Hq must be divisible by positive Hkv; consecutive groups of
    Hq/Hkv query heads share each KV head. Each query head gets its own CUDA
    block and online-softmax state, without repeating K/V in GPU memory.

    Inputs are strided CUDA float32 or float16 tensors with the same dtype and
    device, without gradients or lazy negative views. Feature stride must be 1;
    outer strides can be nonnegative, including zero for read-only broadcasts.
    Q, K, and V use their own head/token strides, so cache-prefix views require
    no contiguous copies. All L positions must be
    valid and include the current token; Q/K already have RoPE. L and D must
    be positive. Loads convert to FP32; scores, softmax and the running output
    stay FP32 across chunks. The final output is cast once to the input dtype.
    Output is new contiguous storage. decode_attention_paged uses the same
    online computation with a reader for fixed physical pages.
    """
    return _load_extension().decode_attention_gqa(query, key, value)


def _checked_paged_query(query, cache, layer_idx):
    from .paged_kv_cache import PagedKVCache

    if not isinstance(cache, PagedKVCache):
        raise ValueError('paged decode requires an owned PagedKVCache')
    length = cache.length(layer_idx)
    if length <= 0:
        raise ValueError('paged decode requires a nonempty cache layer')
    if not isinstance(query, torch.Tensor):
        raise ValueError('query must be a tensor')
    if query.layout != torch.strided or query.is_nested:
        raise ValueError('query must use dense strided storage')
    if (query.ndim != 4 or query.shape[0] != 1 or query.shape[1] <= 0
            or query.shape[2] != 1 or query.shape[3] != cache.head_dim):
        raise ValueError('query shape must be [1, query_heads, 1, cache.head_dim]')
    if query.shape[1] % cache.num_kv_heads:
        raise ValueError('query head count must be divisible by KV head count')
    if cache.dtype not in (torch.float16, torch.float32) or query.dtype != cache.dtype:
        raise ValueError('query and cache require matching float16 or float32 dtype')
    if query.requires_grad:
        raise ValueError('paged decode does not support gradients')
    if query.is_neg():
        raise ValueError('paged decode does not support lazy negative views')
    if query.stride(-1) != 1 or any(stride < 0 for stride in query.stride()[:-1]):
        raise ValueError('query requires feature stride 1 and nonnegative outer strides')
    if cache.device.type != 'cuda' or query.device != cache.device:
        raise ValueError('paged decode requires a CUDA cache and query on the same device')
    return length


@torch.inference_mode()
def decode_attention_paged(query: torch.Tensor, cache, *, layer_idx: int) -> torch.Tensor:
    """Attend the latest query directly over one owned PagedKVCache layer.

    Q is [1,Hq,1,D]; Hq is divisible by the cache's original KV head count.
    Query and cached keys already have RoPE. Every valid cached token, including
    the current token, is visible. FP16/FP32 I/O uses the existing FP32 online
    computation, with a paged reader instead of a contiguous reader.

    The immutable fixed mapping is validated by cache allocation; the private
    device table stays on GPU. No gather, KV repetition, host table readback,
    or context-sized score buffer occurs. Only cache.length(layer_idx) tokens
    are read; unused pages/slots are ignored. Query, mapping, buffers and lengths
    are unchanged. Output is fresh contiguous storage with the query's shape,
    device and dtype. Cross-stream lifetime/reuse remains caller-coordinated.
    """
    length = _checked_paged_query(query, cache, layer_idx)
    return _load_extension()._decode_attention_paged(
        query, cache._keys[layer_idx], cache._values[layer_idx],
        cache._device_page_table, length)

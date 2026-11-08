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
                 str(source_dir / 'decode_attention_online.cu')],
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
    GQA accepts outer strides; paging and model dispatch come later.
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
    Output is new contiguous storage. Model integration is a separate step.
    """
    return _load_extension().decode_attention_gqa(query, key, value)

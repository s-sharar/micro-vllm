"""Single-head CUDA decode attention; independent of the introductory exercises."""
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
                 str(source_dir / 'decode_attention.cu')],
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
    baseline; FP16, GQA, paging, online softmax, and model dispatch come later.
    """
    return _load_extension().decode_attention(query, key, value)

"""Introductory inference-only CUDA operations, compiled on their first use.

Activate the project's virtual environment so nvcc and Ninja are on PATH.
Compilation uses PyTorch's extension cache; importing this module does not build.
"""
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def _load_extension() -> ModuleType:
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent / 'csrc'
    return load(
        name='_micro_vllm_cuda',
        sources=[
            str(source_dir / 'bindings.cpp'),
            str(source_dir / 'vector_add.cu'),
            str(source_dir / 'reduce_sum.cu'),
            str(source_dir / 'dot_product.cu'),
        ],
        extra_cflags=['-O2'],
        extra_cuda_cflags=['-O2'],
        with_cuda=True,
    )


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return a new CUDA vector containing a[i] + b[i], without changing inputs.

    Inputs must be equal-shape, contiguous, one-dimensional float32 strided
    tensors on the same CUDA device, with no gradients or lazy negative views.
    The kernel runs on PyTorch's current stream for that device. The first call
    builds the C++/CUDA extension; later calls reuse it. No backward is defined.
    """
    return _load_extension().vector_add(a, b)


def reduce_sum(values: torch.Tensor) -> torch.Tensor:
    """Sum a CUDA float32 vector into a new scalar tensor on the same device.

    The input follows vector_add's one-dimensional, contiguous, strided,
    inference-only contract. Empty vectors sum to zero. A single block combines
    per-thread FP32 sums in shared memory; rounding can differ from torch.sum.
    This introductory kernel prioritizes clarity over large-vector throughput.
    """
    return _load_extension().reduce_sum(values)


def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the unscaled sum of a[i] * b[i] as a new CUDA float32 scalar.

    Inputs follow vector_add's inference-only format and must match in shape
    and device. Empty vectors produce zero. Multiplication and accumulation use
    FP32 in one block; rounding may differ from torch.dot or an FP64 reference.
    This teaching kernel does not yet apply attention's 1/sqrt(head_dim) scale.
    """
    return _load_extension().dot_product(a, b)

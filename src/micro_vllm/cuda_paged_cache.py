"""Private native writes into pool pages owned by a PagedKVCache request."""
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def _load_extension() -> ModuleType:
    from torch.utils.cpp_extension import load

    sources = Path(__file__).resolve().parent / 'csrc'
    return load(
        name='_micro_vllm_paged_cache',
        sources=[str(sources / 'paged_kv_cache.cpp'), str(sources / 'paged_kv_cache.cu')],
        extra_cflags=['-O2'],
        extra_cuda_cflags=['-O2'],
        with_cuda=True,
    )


def _write_paged_kv(key: torch.Tensor, value: torch.Tensor,
                    key_pages: torch.Tensor, value_pages: torch.Tensor,
                    page_table: torch.Tensor, start: int) -> None:
    """Copy post-RoPE K/V [1,H,N,D] into fixed [P,H,B,D] owned pages.

    PagedKVCache is the sole supported public entry point. Its private contiguous
    CUDA int64 page table must map every written position to a unique in-range
    physical ID, validated by the pool and published by reserve() on the stream
    before this write; entries stay fixed while owned. This helper validates metadata,
    NOT arbitrary GPU table contents; it performs no GPU-to-host synchronization.
    A kernel range guard prevents out-of-bounds writes for corrupted physical IDs,
    but corrupted private tables are outside the supported cache contract.

    Sources may share read-only storage and have independent nonnegative outer
    strides; feature stride is one. Destinations are separate contiguous storage,
    without source/destination aliases. FP16/BF16/FP32 copies preserve bits without
    conversion. Gradients and lazy negative views are unsupported. Writes use the
    device's current CUDA stream. No attention or page reservation occurs here.
    """
    _load_extension().write_paged_kv(key, value, key_pages, value_pages, page_table, start)


def _write_paged_kv_batch(key: torch.Tensor, value: torch.Tensor,
                          key_pages: torch.Tensor, value_pages: torch.Tensor,
                          page_table: torch.Tensor, starts: torch.Tensor,
                          counts: torch.Tensor) -> None:
    """Copy each row's first counts[b] tokens of padded K/V [B,H,N,D] in one launch.

    PagedKVBatch is the supported caller. page_table is its contiguous [B,W] int64
    device table built from the rows' published tables and padded with -1; starts
    and counts are [B] int64 device tensors from host metadata. Row b writes
    positions starts[b] + t for t < counts[b]; padded tokens are never written.
    Only metadata is validated here, without reading device values. The kernel
    skips positions outside a row's table width or with an out-of-range physical
    ID, so a row cannot write into another row's table entries. The single-request
    write uses the same kernel with no per-row arrays.
    """
    _load_extension().write_paged_kv_batch(key, value, key_pages, value_pages,
                                           page_table, starts, counts)

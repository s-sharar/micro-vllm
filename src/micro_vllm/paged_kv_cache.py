"""Fixed-page KV storage for one request, with checked CPU/CUDA append.

Create through allocate(). Each layer owns [physical_pages, kv_heads,
page_size, head_dim] buffers. A fixed logical-to-physical mapping is validated
on the host once and copied into a private device table. Do not mutate private
buffers or metadata. Input K already has RoPE; V is its ordinary projection.

gather() allocates a contiguous logical-order copy for reference attention
and tests. Native paged decode reads pages directly; initial prefill uses local K/V.
This class does not allocate/reclaim individual pages dynamically or share
pages between requests. Reset hides old tokens while retaining all storage.
Operations use the current stream; callers coordinate cross-stream reuse.
Unexpected asynchronous failures require discarding/resetting the cache.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch


def _checked_page_ids(page_table, num_pages):
    if page_table is None:
        return tuple(range(num_pages))
    if not isinstance(page_table, Sequence) or isinstance(page_table, (str, bytes)):
        raise ValueError('page_table must be a sequence of physical page IDs')
    page_ids = tuple(page_table)
    if (not page_ids or any(type(page) is not int or not 0 <= page < num_pages
                            for page in page_ids)):
        raise ValueError('page_table must contain nonempty integer IDs within the physical pool')
    if len(set(page_ids)) != len(page_ids):
        raise ValueError('page_table must not assign a physical page more than once')
    return page_ids


def _storage_overlaps(left, right):
    # Interop handles may share memory without sharing a StorageImpl or base pointer.
    left_storage, right_storage = left.untyped_storage(), right.untyped_storage()
    left_start, right_start = left_storage.data_ptr(), right_storage.data_ptr()
    return (left_start < right_start + right_storage.nbytes()
            and right_start < left_start + left_storage.nbytes())


@dataclass
class PagedKVCache:
    """Fixed single-request mapping; append/reset mutate storage and return self."""

    _keys: tuple[torch.Tensor, ...] = field(repr=False)
    _values: tuple[torch.Tensor, ...] = field(repr=False)
    _lengths: list[int]
    _page_ids: tuple[int, ...]
    _device_page_table: torch.Tensor = field(repr=False)

    @classmethod
    def allocate(cls, *, num_layers, num_kv_heads, num_pages, page_size,
                 head_dim, dtype, device, page_table=None):
        dimensions = (num_layers, num_kv_heads, num_pages, page_size, head_dim)
        if any(type(size) is not int or size <= 0 for size in dimensions):
            raise ValueError('cache dimensions must be positive integers')
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('cache dtype must be float32, float16, or bfloat16')
        device = torch.device(device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('paged cache requires a CPU or CUDA device')
        page_ids = _checked_page_ids(page_table, num_pages)
        table = torch.tensor(page_ids, dtype=torch.int64, device=device)
        shape = (num_pages, num_kv_heads, page_size, head_dim)
        keys = tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers))
        values = tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers))
        return cls(keys, values, [0] * num_layers, page_ids, table)

    @property
    def num_layers(self):
        return len(self._keys)

    @property
    def batch_size(self):
        return 1

    @property
    def num_pages(self):
        return self._keys[0].shape[0]

    @property
    def num_kv_heads(self):
        return self._keys[0].shape[1]

    @property
    def page_size(self):
        return self._keys[0].shape[2]

    @property
    def head_dim(self):
        return self._keys[0].shape[3]

    @property
    def page_table(self):
        """Immutable host mapping; the independently owned device table stays private."""
        return self._page_ids

    @property
    def capacity(self):
        return len(self._page_ids) * self.page_size

    @property
    def dtype(self):
        return self._keys[0].dtype

    @property
    def device(self):
        return self._keys[0].device

    @property
    def sequence_length(self):
        if any(length != self._lengths[0] for length in self._lengths):
            raise ValueError('cache layers must have aligned lengths before a model call')
        return self._lengths[0]

    def _check_layer(self, layer_idx):
        if type(layer_idx) is not int or not 0 <= layer_idx < self.num_layers:
            raise ValueError('layer index must identify an existing cache layer')

    def length(self, layer_idx):
        self._check_layer(layer_idx)
        return self._lengths[layer_idx]

    def _append_end(self, layer_idx, key, value):
        self._check_layer(layer_idx)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError('keys and values must be tensors')
        if key.layout != torch.strided or value.layout != torch.strided:
            raise ValueError('keys and values must use dense strided storage')
        if (key.ndim != 4 or key.shape != value.shape or key.shape[2] <= 0
                or (key.shape[0], key.shape[1], key.shape[3])
                != (1, self.num_kv_heads, self.head_dim)):
            raise ValueError('K/V must match [1, kv_heads, new_tokens, head_dim] with new_tokens > 0')
        for source in (key, value):
            if source.device != self.device or source.dtype != self.dtype:
                raise ValueError('K/V must match the cache dtype and device')
            if source.requires_grad or source.is_neg():
                raise ValueError('K/V must not require gradients or use lazy negative views')
            if source.stride(-1) != 1 or any(stride < 0 for stride in source.stride()[:-1]):
                raise ValueError('K/V require feature stride 1 and nonnegative outer strides')
            if any(_storage_overlaps(source, target)
                   for target in (self._keys[layer_idx], self._values[layer_idx])):
                raise ValueError('K/V inputs must not overlap destination page storage')
        end = self._lengths[layer_idx] + key.shape[2]
        if end > self.capacity:
            raise ValueError('paged KV cache capacity exceeded')
        return end

    def _append_cpu(self, layer_idx, key, value, start):
        offset = 0
        while offset < key.shape[2]:
            logical_page, slot = divmod(start + offset, self.page_size)
            physical_page = self._page_ids[logical_page]
            count = min(self.page_size - slot, key.shape[2] - offset)
            self._keys[layer_idx][physical_page, :, slot:slot + count].copy_(
                key[0, :, offset:offset + count])
            self._values[layer_idx][physical_page, :, slot:slot + count].copy_(
                value[0, :, offset:offset + count])
            offset += count

    @torch.inference_mode()
    def append(self, layer_idx, key, value):
        """Write only new slots; validate predictable failures before any writes."""
        end = self._append_end(layer_idx, key, value)
        start = self._lengths[layer_idx]
        if self.device.type == 'cuda':
            from .cuda_paged_cache import _write_paged_kv

            _write_paged_kv(key, value, self._keys[layer_idx], self._values[layer_idx],
                            self._device_page_table, start)
        else:
            self._append_cpu(layer_idx, key, value, start)
        self._lengths[layer_idx] = end
        return self

    def _gather_buffer(self, pages, length):
        if length == 0:
            return torch.empty((1, self.num_kv_heads, 0, self.head_dim),
                               dtype=self.dtype, device=self.device)
        parts = []
        for logical_page in range((length + self.page_size - 1) // self.page_size):
            count = min(self.page_size, length - logical_page * self.page_size)
            parts.append(pages[self._page_ids[logical_page], :, :count])
        return torch.cat(parts, dim=1).unsqueeze(0)

    @torch.inference_mode()
    def gather(self, layer_idx):
        """Allocate logical-order K/V copies for reference attention and tests."""
        length = self.length(layer_idx)
        return (self._gather_buffer(self._keys[layer_idx], length),
                self._gather_buffer(self._values[layer_idx], length))

    def reset(self):
        """Clear lengths, preserving physical pages and the fixed mapping."""
        for layer_idx in range(self.num_layers):
            self._lengths[layer_idx] = 0
        return self

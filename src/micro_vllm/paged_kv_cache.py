"""Request-owned paged KV view over a shared KVPool, with checked CPU/CUDA append.

Create through for_pool() for dynamic pages or allocate() for a private pool
with a fixed mapping. Pool layers own [physical_pages, kv_heads, page_size,
head_dim] buffers; this request owns its logical-to-physical page list, a
device table preallocated at its maximum page count, and per-layer lengths.
reserve(new_tokens) obtains every extra page for one forward in a single
all-or-nothing pool reservation and publishes the IDs on the current stream
before that forward's K/V writes. Published entries never change while owned.

gather() allocates a contiguous logical-order copy for reference attention
and tests. Native paged decode reads pages directly; initial prefill uses local K/V.
release() returns pages to the pool once; afterwards the request is empty and
unusable. Reset hides old tokens while keeping owned pages. A forward uses
one stream; release joins every stream that reserved pages. Input K already
has RoPE. Unexpected asynchronous failures require releasing the request.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from .kv_pool import KVPool


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
    """One request's page table and lengths; append/reserve/reset return self."""

    _keys: tuple[torch.Tensor, ...] = field(repr=False)
    _values: tuple[torch.Tensor, ...] = field(repr=False)
    _lengths: list[int]
    _page_ids: list[int]
    _device_page_table: torch.Tensor = field(repr=False)
    _pool: KVPool = field(repr=False)
    _staging: torch.Tensor | None = field(default=None, repr=False)
    _owner: object = field(default_factory=object, repr=False)
    _streams: set = field(default_factory=set, repr=False)
    _released: bool = False

    @classmethod
    def allocate(cls, *, num_layers, num_kv_heads, num_pages, page_size,
                 head_dim, dtype, device, page_table=None):
        dimensions = (num_layers, num_kv_heads, num_pages, page_size, head_dim)
        if any(type(size) is not int or size <= 0 for size in dimensions):
            raise ValueError('cache dimensions must be positive integers')
        page_ids = _checked_page_ids(page_table, num_pages)
        pool = KVPool.allocate(num_layers=num_layers, num_kv_heads=num_kv_heads,
                               num_pages=num_pages, page_size=page_size, head_dim=head_dim,
                               dtype=dtype, device=device)
        table = torch.tensor(page_ids, dtype=torch.int64, device=pool.device)
        cache = cls(pool._keys, pool._values, [0] * num_layers, list(page_ids), table, pool)
        pool._claim(cache._owner, page_ids)
        return cache

    @classmethod
    def for_pool(cls, pool, *, max_tokens):
        """Start an empty request that may grow to max_tokens within the shared pool."""
        if not isinstance(pool, KVPool):
            raise ValueError('pool must be a KVPool')
        if type(max_tokens) is not int or not 0 < max_tokens <= pool.token_capacity:
            raise ValueError('max_tokens must be a positive integer within pool token capacity')
        max_pages = -(-max_tokens // pool.page_size)
        table = torch.full((max_pages,), -1, dtype=torch.int64, device=pool.device)
        staging = (torch.empty(max_pages, dtype=torch.int64, pin_memory=True)
                   if pool.device.type == 'cuda' else None)
        return cls(pool._keys, pool._values, [0] * pool.num_layers, [], table, pool, staging)

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
        """Snapshot of owned pages in logical order; the device table stays private."""
        return tuple(self._page_ids)

    @property
    def capacity(self):
        """Maximum request tokens; reserve() must own a page before writes reach it."""
        return self._device_page_table.numel() * self.page_size

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

    def _check_active(self):
        if self._released:
            raise ValueError('paged KV cache was released')

    def _append_end(self, layer_idx, key, value):
        self._check_active()
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
        if end > len(self._page_ids) * self.page_size:
            raise ValueError('paged KV cache capacity exceeded; reserve pages first')
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
        """Clear lengths, preserving owned pages and their mapping."""
        for layer_idx in range(self.num_layers):
            self._lengths[layer_idx] = 0
        return self

    @torch.inference_mode()
    def reserve(self, new_tokens):
        """Own and publish every page the next aligned new_tokens need, or change nothing."""
        self._check_active()
        if type(new_tokens) is not int or new_tokens <= 0:
            raise ValueError('new_tokens must be a positive integer')
        end = self.sequence_length + new_tokens
        if end > self.capacity:
            raise ValueError('paged KV cache capacity exceeded')
        if self.device.type == 'cuda':
            self._streams.add(torch.cuda.current_stream(self.device))
        first = len(self._page_ids)
        count = -(-end // self.page_size) - first
        if count > 0:
            self._publish(first, self._pool._reserve(self._owner, count))
        return self

    def _publish(self, first, pages):
        end = first + len(pages)
        ids = torch.tensor(pages, dtype=torch.int64)
        if self._staging is None:
            self._device_page_table[first:end] = ids
        else:
            # Staging slots are written once per request; pending copies never see edits.
            self._staging[first:end] = ids
            with torch.cuda.device(self.device):
                self._device_page_table[first:end].copy_(
                    self._staging[first:end], non_blocking=True)
        self._page_ids.extend(pages)

    def release(self):
        """Return owned pages to the pool once; CUDA pages retire until queued work ends."""
        if self._released:
            return
        self._pool._release(self._owner, self._page_ids, self._streams,
                            (self._device_page_table, self._staging))
        self._released, self._page_ids = True, []
        self.reset()

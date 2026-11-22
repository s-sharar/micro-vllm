"""Shared physical KV page pool with per-page Free/Owned/Retiring ownership.

Create through allocate(). The pool owns every layer's [num_pages, kv_heads,
page_size, head_dim] K/V buffers; page ID p names the same slot range in all
layers. Requests (PagedKVCache.for_pool) own their page tables and lengths and
obtain pages only through all-or-nothing reservations. Reservation reclaims
completed retirements first, then waits on the oldest retirement events, and
raises KVPoolExhausted without changing ownership if pages are still short.
Freed IDs are reused in FIFO order.

CPU release frees pages immediately. CUDA release records one event on the
request's bound stream (or the current stream if it never queued work) and
keeps the pages Retiring, plus the request's table and host staging memory
alive, until that event completes. Single-threaded host use only.
"""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch


class KVPoolExhausted(RuntimeError):
    """Not enough Free or reclaimable pages for an all-or-nothing reservation."""


class PageState(Enum):
    FREE = 'free'
    OWNED = 'owned'
    RETIRING = 'retiring'


@dataclass
class _Retirement:
    event: torch.cuda.Event
    pages: list[int]
    keepalive: tuple


@dataclass
class KVPool:
    """Physical page storage plus ownership; requests hold logical mappings."""

    _keys: tuple[torch.Tensor, ...] = field(repr=False)
    _values: tuple[torch.Tensor, ...] = field(repr=False)
    _states: list[PageState] = field(repr=False)
    _owners: list[object | None] = field(repr=False)
    _free: deque = field(repr=False)
    _retiring: deque = field(default_factory=deque, repr=False)

    @classmethod
    def allocate(cls, *, num_layers, num_kv_heads, num_pages, page_size, head_dim,
                 dtype, device):
        dimensions = (num_layers, num_kv_heads, num_pages, page_size, head_dim)
        if any(type(size) is not int or size <= 0 for size in dimensions):
            raise ValueError('pool dimensions must be positive integers')
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('pool dtype must be float32, float16, or bfloat16')
        device = torch.device(device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('KV pool requires a CPU or CUDA device')
        shape = (num_pages, num_kv_heads, page_size, head_dim)
        keys = tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers))
        values = tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers))
        return cls(keys, values, [PageState.FREE] * num_pages, [None] * num_pages,
                   deque(range(num_pages)))

    @property
    def num_layers(self):
        return len(self._keys)

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
    def dtype(self):
        return self._keys[0].dtype

    @property
    def device(self):
        return self._keys[0].device

    @property
    def token_capacity(self):
        return self.num_pages * self.page_size

    @property
    def num_free(self):
        return len(self._free)

    def state(self, page_id):
        if type(page_id) is not int or not 0 <= page_id < self.num_pages:
            raise ValueError('page ID must identify a physical page in this pool')
        return self._states[page_id]

    def _set_free(self, pages):
        for page in pages:
            self._states[page], self._owners[page] = PageState.FREE, None
            self._free.append(page)

    def reclaim(self):
        """Free Retiring pages whose release events completed; return the page count."""
        reclaimed, pending = 0, deque()
        while self._retiring:
            retirement = self._retiring.popleft()
            if retirement.event.query():
                self._set_free(retirement.pages)
                reclaimed += len(retirement.pages)
            else:
                pending.append(retirement)
        self._retiring = pending
        return reclaimed

    def _reserve(self, owner, count):
        self.reclaim()
        while count > self.num_free and self._retiring:
            retirement = self._retiring.popleft()
            retirement.event.synchronize()
            self._set_free(retirement.pages)
        if count > self.num_free:
            raise KVPoolExhausted(
                f'KV pool has {self.num_free} free pages; reservation needs {count}')
        pages = [self._free.popleft() for _ in range(count)]
        for page in pages:
            self._states[page], self._owners[page] = PageState.OWNED, owner
        return pages

    def _claim(self, owner, pages):
        if any(self._states[page] is not PageState.FREE for page in pages):
            raise ValueError('claimed pages must be free')
        claimed = set(pages)
        self._free = deque(page for page in self._free if page not in claimed)
        for page in pages:
            self._states[page], self._owners[page] = PageState.OWNED, owner

    def _release(self, owner, pages, stream, keepalive):
        if any(self._states[page] is not PageState.OWNED or self._owners[page] is not owner
               for page in pages):
            raise RuntimeError('released pages must be owned by the releasing request')
        if not pages:
            return
        if self.device.type != 'cuda':
            self._set_free(pages)
            return
        with torch.cuda.device(self.device):
            event = torch.cuda.Event()
            event.record(stream or torch.cuda.current_stream())
        for page in pages:
            self._states[page], self._owners[page] = PageState.RETIRING, None
        self._retiring.append(_Retirement(event, list(pages), keepalive))

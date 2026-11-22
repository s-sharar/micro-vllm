"""One batched forward's view over several PagedKVCache requests sharing a KVPool.

PreparedModel.forward_batch is the entry point. A batch is either all
full-prompt prefill (empty caches, unequal prompt lengths, right-padded to the
longest row) or all single-token decode (unequal cached lengths). prepare()
validates every row without changing anything. reserve() binds the rows to
the current stream, obtains all of their extra pages in one all-or-nothing
pool reservation and publishes each row's IDs; KVPoolExhausted leaves every
row's ownership, page table and lengths unchanged and the rows usable.

After reserve() succeeds the batch is live and owns the forward's outcome.
commit() advances every layer length only after the whole forward succeeded;
during it layers read their just-written tokens through the batch's own row
lengths. abort() marks every row failed and releases it through
PagedKVCache.release(), including reserved but unpublished pages, so partly
written pages are never read again. A finished batch, or one whose rows were
released, rejects further writes and reads. Writes copy only each row's
valid tokens; padding never enters the cache.

Device metadata never needs device-to-host reads: a [3, B] int64 tensor of
row starts, counts and ends is staged in pinned host memory and copied without
blocking on the current stream (PyTorch's pinned host allocator does not reuse
that memory until the copy's recorded event completes), and the -1-padded
[B, max_pages] table is copied on the device from the rows' published tables.
"""
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch.nn.utils.rnn import pad_sequence

from . import cuda_paged_cache
from .kv_pool import KVPool
from .paged_kv_cache import PagedKVCache


def _report_cleanup_failure(error, message):
    add_note = getattr(error, 'add_note', None)
    if callable(add_note):
        add_note(message)
    else:
        warnings.warn(message, RuntimeWarning, stacklevel=3)


@dataclass
class PagedKVBatch:
    """Rows' request caches plus per-row starts and new-token counts for one forward."""

    caches: tuple[PagedKVCache, ...]
    starts: tuple[int, ...]
    counts: tuple[int, ...]
    _pool: KVPool = field(repr=False)
    _stream: torch.cuda.Stream | None = field(default=None, repr=False)
    _metadata: torch.Tensor | None = field(default=None, repr=False)
    _staging: torch.Tensor | None = field(default=None, repr=False)
    _table: torch.Tensor | None = field(default=None, repr=False)
    _decode_mask: torch.Tensor | None = field(default=None, repr=False)
    _live: bool = field(default=False, repr=False)

    @classmethod
    def prepare(cls, caches, counts, *, num_layers, num_kv_heads, head_dim, dtype, device,
                max_positions):
        """Validate the whole batch against the model before any reservation or write."""
        if (not isinstance(caches, Sequence) or isinstance(caches, (str, bytes))
                or not caches or len(caches) != len(counts)):
            raise ValueError('caches must provide one cache per row')
        if any(not isinstance(cache, PagedKVCache) for cache in caches):
            raise ValueError('caches must be PagedKVCache requests')
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError('batch caches must be distinct requests')
        pool = caches[0]._pool
        if any(cache._pool is not pool for cache in caches):
            raise ValueError('batch caches must share the same KVPool')
        expected = (num_layers, num_kv_heads, head_dim, dtype, torch.device(device))
        if (pool.num_layers, pool.num_kv_heads, pool.head_dim, pool.dtype, pool.device) != expected:
            raise ValueError('cache geometry must match the prepared model')
        for cache in caches:
            cache._check_active()
            cache._check_stream()
        starts = tuple(cache.sequence_length for cache in caches)
        if any(starts) and not all(starts):
            raise ValueError('mixed prefill and decode rows are not supported in one batch')
        if starts[0] and any(count != 1 for count in counts):
            raise ValueError('decode rows must contain exactly one token')
        for cache, start, count in zip(caches, starts, counts):
            if start + count > max_positions:
                raise ValueError('sequence length exceeds the configured context length')
            if start + count > cache.capacity:
                raise ValueError('paged KV cache capacity exceeded')
        return cls(tuple(caches), starts, tuple(counts), pool)

    @property
    def size(self):
        return len(self.caches)

    @property
    def device(self):
        return self._pool.device

    @property
    def is_prefill(self):
        return self.starts[0] == 0

    @property
    def max_tokens(self):
        return max(self.counts)

    @property
    def ends(self):
        return tuple(start + count for start, count in zip(self.starts, self.counts))

    @property
    def device_starts(self):
        return self._metadata[0]

    @property
    def device_counts(self):
        return self._metadata[1]

    @property
    def device_ends(self):
        return self._metadata[2]

    def _check_live(self):
        """Pages may be touched only between a successful reserve() and commit()/abort()."""
        if not self._live or any(cache._released or cache._failed for cache in self.caches):
            raise ValueError('batched paged work requires a live reserved PagedKVBatch')
        if self._stream is not None and torch.cuda.current_stream(self.device) != self._stream:
            raise ValueError('paged KV batch work must stay on the request stream')

    @torch.inference_mode()
    def reserve(self):
        """Own and publish every row's pages in one reservation; exhaustion changes no row."""
        for cache in self.caches:
            cache._check_active()
            cache._check_stream()
        page_size = self._pool.page_size
        firsts = [len(cache._page_ids) for cache in self.caches]
        needed = [max(0, -(-end // page_size) - first) for end, first in zip(self.ends, firsts)]
        grants = self._pool._reserve_many(
            [(cache._owner, count) for cache, count in zip(self.caches, needed)])
        published = 0
        try:
            for cache in self.caches:
                cache._bind_stream()
            self._stream = self.caches[0]._stream
            for cache, first, pages in zip(self.caches, firsts, grants):
                if pages:
                    cache._publish(first, pages)
                published += 1
            self._stage_metadata()
        except BaseException as error:
            # Unpublished grants join their page lists so release() frees or retries them.
            for cache, pages in zip(self.caches[published:], grants[published:]):
                cache._page_ids.extend(pages)
            self.abort(error)
            raise
        self._live = True
        return self

    def _stage_metadata(self):
        rows = [self.starts, self.counts, self.ends]
        if self.device.type != 'cuda':
            self._metadata = torch.tensor(rows, dtype=torch.int64)
            return
        self._staging = torch.tensor(rows, dtype=torch.int64, pin_memory=True)
        self._metadata = self._staging.to(self.device, non_blocking=True)
        self._table = pad_sequence(
            [cache._device_page_table[:len(cache._page_ids)] for cache in self.caches],
            batch_first=True, padding_value=-1)

    def positions(self):
        """RoPE positions: shared [1, T] for prefill, each row's [B, 1] start for decode."""
        if self.is_prefill:
            return torch.arange(self.max_tokens, device=self.device).unsqueeze(0)
        return self.device_starts.unsqueeze(1)

    @torch.inference_mode()
    def append(self, layer_idx, key, value):
        """Write each row's first counts[b] tokens of padded rotated K/V [B, Hkv, T, D]."""
        self._check_live()
        shape = (self.size, self._pool.num_kv_heads, self.max_tokens, self._pool.head_dim)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f'batched K/V must have shape {shape}')
        if self.device.type == 'cuda':
            cuda_paged_cache._write_paged_kv_batch(
                key, value, self._pool._keys[layer_idx], self._pool._values[layer_idx],
                self._table, self.device_starts, self.device_counts)
            return
        for row, (cache, start, count) in enumerate(zip(self.caches, self.starts, self.counts)):
            cache._append_cpu(layer_idx, key[row:row + 1, :, :count],
                              value[row:row + 1, :, :count], start)

    def zero_padding(self, value):
        """Zero padded prefill V rows; masked probabilities are 0, but 0 * NaN is not."""
        if all(count == self.max_tokens for count in self.counts):
            return value
        positions = torch.arange(self.max_tokens, device=self.device)
        padding = positions.unsqueeze(0) >= self.device_counts.unsqueeze(1)
        return value.masked_fill(padding[:, None, :, None], 0)

    @torch.inference_mode()
    def gather(self, layer_idx):
        """Allocate zero-padded logical-order K/V [B, Hkv, max_end, D] for reference decode."""
        self._check_live()
        length = max(self.ends)
        shape = (self.size, self._pool.num_kv_heads, length, self._pool.head_dim)
        keys = torch.zeros(shape, dtype=self._pool.dtype, device=self.device)
        values = torch.zeros_like(keys)
        for row, (cache, end) in enumerate(zip(self.caches, self.ends)):
            keys[row:row + 1, :, :end] = cache._gather_buffer(self._pool._keys[layer_idx], end)
            values[row:row + 1, :, :end] = cache._gather_buffer(self._pool._values[layer_idx], end)
        return keys, values

    def decode_mask(self):
        """Boolean [B, 1, max_end] mask blocking each row's positions at or past its end."""
        if self._decode_mask is None:
            positions = torch.arange(max(self.ends), device=self.device)
            self._decode_mask = positions[None, None, :] >= self.device_ends[:, None, None]
        return self._decode_mask

    def last_hidden(self, hidden):
        """Select each row's last valid position from padded [B, T, hidden]."""
        if all(count == self.max_tokens for count in self.counts):
            return hidden[:, -1]
        rows = torch.arange(self.size, device=self.device)
        return hidden[rows, self.device_counts - 1]

    def commit(self):
        self._live = False
        for cache, end in zip(self.caches, self.ends):
            cache._lengths[:] = [end] * cache.num_layers

    def abort(self, error):
        """Fail and release every row; a release error is reported and leaves that row retryable."""
        self._live = False
        for row, cache in enumerate(self.caches):
            cache._failed = True
            try:
                cache.release()
            except Exception as release_error:
                _report_cleanup_failure(error, f'releasing batch row {row} failed: '
                                               f'{release_error!r}; call release() on that cache again')

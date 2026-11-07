"""Fixed-capacity, contiguous KV storage with in-place writes.

Each layer owns K/V buffers [batch, kv_heads, capacity, head_dim]. Keys must
already have RoPE applied; values are stored directly from their projection.
Append writes only new slots without reallocating buffers. Read returns
borrowed prefix views: callers must not modify them, and reset/reuse may
change their contents. Append inputs must not overlap destination slots.
One cache belongs to one request; concurrent writes require coordination.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class ContiguousKVCache:
    """Create via allocate(); append() and reset() mutate this cache and return self."""

    _keys: tuple[torch.Tensor, ...] = field(repr=False)
    _values: tuple[torch.Tensor, ...] = field(repr=False)
    _lengths: list[int]

    @classmethod
    def allocate(
        cls, *, num_layers: int, batch_size: int, num_kv_heads: int,
        capacity: int, head_dim: int, dtype: torch.dtype,
        device: torch.device | str,
    ) -> 'ContiguousKVCache':
        dimensions = (num_layers, batch_size, num_kv_heads, capacity, head_dim)
        if any(type(size) is not int or size <= 0 for size in dimensions):
            raise ValueError('cache dimensions must be positive integers')
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('cache dtype must be float32, float16, or bfloat16')
        shape = (batch_size, num_kv_heads, capacity, head_dim)
        keys = tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers))
        values = tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers))
        return cls(keys, values, [0] * num_layers)

    @property
    def num_layers(self) -> int:
        return len(self._keys)

    @property
    def batch_size(self) -> int:
        return self._keys[0].shape[0]

    @property
    def num_kv_heads(self) -> int:
        return self._keys[0].shape[1]

    @property
    def head_dim(self) -> int:
        return self._keys[0].shape[3]

    @property
    def capacity(self) -> int:
        return self._keys[0].shape[2]

    @property
    def dtype(self) -> torch.dtype:
        return self._keys[0].dtype

    @property
    def device(self) -> torch.device:
        return self._keys[0].device

    @property
    def sequence_length(self) -> int:
        """Read the shared length only when every layer has been populated equally."""
        if any(length != self._lengths[0] for length in self._lengths):
            raise ValueError('cache layers must have aligned lengths before a model call')
        return self._lengths[0]

    def _check_layer(self, layer_idx: int) -> None:
        if type(layer_idx) is not int or not 0 <= layer_idx < self.num_layers:
            raise ValueError('layer index must identify an existing cache layer')

    def length(self, layer_idx: int) -> int:
        self._check_layer(layer_idx)
        return self._lengths[layer_idx]

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Borrow views of valid entries only; consume before resetting/reusing this cache."""
        length = self.length(layer_idx)
        return (self._keys[layer_idx][:, :, :length, :],
                self._values[layer_idx][:, :, :length, :])

    def _append_end(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> int:
        self._check_layer(layer_idx)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError('keys and values must be tensors')
        if key.layout != torch.strided or value.layout != torch.strided:
            raise ValueError('keys and values must use dense strided storage')
        expected = self._keys[layer_idx].shape
        if (key.ndim != 4 or key.shape != value.shape or key.shape[2] <= 0
                or (key.shape[0], key.shape[1], key.shape[3])
                != (expected[0], expected[1], expected[3])):
            raise ValueError('K/V must match [batch, kv_heads, new_tokens, head_dim] with new_tokens > 0')
        if (key.dtype != self.dtype or value.dtype != self.dtype
                or key.device != self.device or value.device != self.device):
            raise ValueError('K/V must match the cache dtype and device')
        end = self._lengths[layer_idx] + key.shape[2]
        if end > self.capacity:
            raise ValueError('KV cache capacity exceeded')
        return end

    @torch.inference_mode()
    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> 'ContiguousKVCache':
        """Validate both chunks, copy new entries into owned buffers, then advance length."""
        end = self._append_end(layer_idx, key, value)
        start = self._lengths[layer_idx]
        self._keys[layer_idx][:, :, start:end, :].copy_(key)
        self._values[layer_idx][:, :, start:end, :].copy_(value)
        self._lengths[layer_idx] = end
        return self

    def reset(self) -> 'ContiguousKVCache':
        """Clear populated lengths in place, retaining allocations and hiding old data."""
        for layer_idx in range(self.num_layers):
            self._lengths[layer_idx] = 0
        return self

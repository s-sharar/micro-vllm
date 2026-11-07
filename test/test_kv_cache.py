"""Contiguous KV storage: in-place writes, borrowed views, and stable buffers."""
import importlib.util
import unittest

import torch

if importlib.util.find_spec('micro_vllm.kv_cache') is not None:
    from micro_vllm.kv_cache import ContiguousKVCache
else:
    ContiguousKVCache = None


def allocate(**overrides):
    settings = dict(num_layers=2, batch_size=2, num_kv_heads=2, capacity=5,
                    head_dim=4, dtype=torch.float32, device='cpu')
    return ContiguousKVCache.allocate(**{**settings, **overrides})


def entries(tokens, offset=0., *, dtype=torch.float32, device='cpu'):
    return (torch.arange(2 * tokens * 2 * 4, dtype=dtype, device=device)
            .reshape(2, tokens, 2, 4).transpose(1, 2) + offset)


def buffer_addresses(cache):
    return tuple(tensor.data_ptr() for tensor in (*cache._keys, *cache._values))


class KVCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ContiguousKVCache, 'kv_cache.py must provide ContiguousKVCache')

    def test_empty_cache_has_capacity_but_no_readable_tokens(self):
        cache = allocate()
        self.assertEqual(cache.num_layers, 2)
        self.assertEqual(cache.capacity, 5)
        self.assertEqual(cache.sequence_length, 0)
        self.assertEqual(cache.dtype, torch.float32)
        self.assertEqual(cache.device, torch.device('cpu'))
        for layer in range(2):
            self.assertEqual(cache.length(layer), 0)
            key, value = cache.read(layer)
            self.assertEqual(key.shape, (2, 2, 0, 4))
            self.assertEqual(value.shape, key.shape)

    def test_prefill_decode_and_chunk_append_reuse_buffers_without_changing_inputs(self):
        cache = allocate()
        addresses = buffer_addresses(cache)
        first_k, first_v = entries(2), entries(2, 100.)
        original_k, original_v = first_k.clone(), first_v.clone()
        self.assertIs(cache.append(0, first_k, first_v), cache)
        self.assertEqual(cache.length(0), 2)
        cache.append(0, entries(1, 200.), entries(1, 300.))
        self.assertEqual(cache.length(0), 3)
        cache.append(0, entries(2, 400.), entries(2, 500.))
        self.assertEqual(cache.length(0), 5)
        expected_k = torch.cat((first_k, entries(1, 200.), entries(2, 400.)), dim=2)
        expected_v = torch.cat((first_v, entries(1, 300.), entries(2, 500.)), dim=2)
        actual_k, actual_v = cache.read(0)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)
        self.assertEqual(buffer_addresses(cache), addresses)
        torch.testing.assert_close(first_k, original_k, rtol=0, atol=0)
        torch.testing.assert_close(first_v, original_v, rtol=0, atol=0)

    def test_layers_are_independent_and_sequence_length_requires_alignment(self):
        cache = allocate()
        cache.append(0, entries(2), entries(2, 100.))
        self.assertEqual(cache.length(1), 0)
        self.assertEqual(cache.read(1)[0].shape[2], 0)
        with self.assertRaises(ValueError):
            _ = cache.sequence_length
        cache.append(1, entries(2, 200.), entries(2, 300.))
        self.assertEqual(cache.sequence_length, 2)
        torch.testing.assert_close(cache.read(0)[0], entries(2), rtol=0, atol=0)
        torch.testing.assert_close(cache.read(1)[0], entries(2, 200.), rtol=0, atol=0)

    def test_read_returns_borrowed_views_without_copying_populated_entries(self):
        cache = allocate()
        cache.append(0, entries(2), entries(2, 100.))
        key, value = cache.read(0)
        self.assertEqual(key.data_ptr(), cache._keys[0].data_ptr())
        self.assertEqual(value.data_ptr(), cache._values[0].data_ptr())
        cache.append(0, entries(1, 200.), entries(1, 300.))
        self.assertEqual(key.shape, (2, 2, 2, 4))
        torch.testing.assert_close(key, entries(2), rtol=0, atol=0)
        torch.testing.assert_close(value, entries(2, 100.), rtol=0, atol=0)

    def test_reset_reuses_buffers_and_shorter_reuse_hides_stale_values(self):
        cache = allocate()
        cache.append(0, entries(3), entries(3, 100.))
        addresses = buffer_addresses(cache)
        self.assertIs(cache.reset(), cache)
        self.assertEqual(cache.sequence_length, 0)
        self.assertEqual(cache.read(0)[0].shape, (2, 2, 0, 4))
        self.assertEqual(buffer_addresses(cache), addresses)
        cache.append(0, entries(1, 900.), entries(1, 1000.))
        self.assertEqual(cache.length(0), 1)
        torch.testing.assert_close(cache.read(0)[0], entries(1, 900.), rtol=0, atol=0)
        torch.testing.assert_close(cache.read(0)[1], entries(1, 1000.), rtol=0, atol=0)
        self.assertEqual(buffer_addresses(cache), addresses)

    def test_independent_requests_do_not_share_storage(self):
        first, second = allocate(), allocate()
        first.append(0, entries(2), entries(2, 100.))
        second.append(0, entries(2, 200.), entries(2, 300.))
        self.assertTrue(set(buffer_addresses(first)).isdisjoint(buffer_addresses(second)))
        first.reset()
        first.append(0, entries(1, 900.), entries(1, 1000.))
        self.assertEqual(second.length(0), 2)
        torch.testing.assert_close(second.read(0)[0], entries(2, 200.), rtol=0, atol=0)

    def test_exact_capacity_succeeds_and_overflow_preserves_cache(self):
        full = allocate(capacity=3)
        full.append(0, entries(3), entries(3, 100.))
        addresses = buffer_addresses(full)
        with self.assertRaises(ValueError):
            full.append(0, entries(1), entries(1))
        self.assertEqual(full.length(0), 3)
        torch.testing.assert_close(full.read(0)[0], entries(3), rtol=0, atol=0)
        self.assertEqual(buffer_addresses(full), addresses)

    def test_rejects_invalid_allocations(self):
        for field in ('num_layers', 'batch_size', 'num_kv_heads', 'capacity', 'head_dim'):
            for invalid in (0, -1, True, 1.5):
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    allocate(**{field: invalid})
        for dtype in (torch.long, torch.complex64, 'float32'):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                allocate(dtype=dtype)

    def test_rejects_invalid_chunks_before_changing_buffers_or_lengths(self):
        cache = allocate()
        for tensor in (*cache._keys, *cache._values):
            tensor.fill_(7.)
        cache.append(0, entries(1), entries(1, 100.))
        originals = tuple(tensor.clone() for tensor in (*cache._keys, *cache._values))
        key, value = entries(1), entries(1, 100.)
        for index in (-1, 2, True, 0.5):
            for operation in (lambda: cache.read(index), lambda: cache.length(index),
                              lambda: cache.append(index, key, value)):
                with self.subTest(layer=index), self.assertRaises(ValueError):
                    operation()
        invalid = (
            (None, value), (key, None), (key[0], value), (key, value[0]),
            (entries(0), entries(0)),
            (torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4)),
            (torch.zeros(2, 1, 1, 4), torch.zeros(2, 1, 1, 4)),
            (torch.zeros(2, 2, 1, 8), torch.zeros(2, 2, 1, 8)),
            (key, entries(2)), (key.half(), value), (key, value.half()),
            (key.to('meta'), value), (key, value.to('meta')),
            (key.to_sparse(), value), (key, value.to_sparse()),
        )
        for index, (keys, values) in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                cache.append(0, keys, values)
            self.assertEqual(cache.length(0), 1)
            self.assertEqual(cache.length(1), 0)
            for current, original in zip((*cache._keys, *cache._values), originals):
                torch.testing.assert_close(current, original, rtol=0, atol=0)

    def test_append_is_inference_only_and_does_not_record_autograd(self):
        cache = allocate()
        keys = entries(2).requires_grad_()
        values = entries(2, 100.).requires_grad_()
        cache.append(0, keys, values)
        self.assertEqual(cache.length(0), 2)
        for tensor in cache.read(0):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
        self.assertTrue(keys.requires_grad)
        self.assertTrue(values.requires_grad)

    def test_cpu_storage_preserves_supported_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                cache = allocate(dtype=dtype)
                keys, values = entries(2, dtype=dtype), entries(2, 100., dtype=dtype)
                cache.append(0, keys, values)
                actual_k, actual_v = cache.read(0)
                self.assertEqual(actual_k.dtype, dtype)
                self.assertEqual(actual_k.device.type, 'cpu')
                torch.testing.assert_close(actual_k, keys, rtol=0, atol=0)
                torch.testing.assert_close(actual_v, values, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_append_and_reset_preserve_allocations(self):
        with torch.inference_mode():
            cache = allocate(dtype=torch.float16, device='cuda')
        addresses = buffer_addresses(cache)
        keys = entries(2, dtype=torch.float16, device='cuda')
        values = entries(2, 100., dtype=torch.float16, device='cuda')
        self.assertIs(cache.append(0, keys, values), cache)
        actual_k, actual_v = cache.read(0)
        self.assertEqual(cache.device, keys.device)
        torch.testing.assert_close(actual_k, keys, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, values, rtol=0, atol=0)
        self.assertEqual(buffer_addresses(cache), addresses)
        cache.reset()
        self.assertEqual(cache.sequence_length, 0)
        self.assertEqual(buffer_addresses(cache), addresses)


if __name__ == '__main__':
    unittest.main()

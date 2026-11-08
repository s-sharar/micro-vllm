"""Grouped-query CUDA decode accepts dense feature rows with independent outer strides."""
import importlib
import unittest

import torch

from micro_vllm.kv_cache import ContiguousKVCache
from test import test_cuda_gqa as gqa_reference


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class GroupedQueryAttentionStrideTests(unittest.TestCase):
    def setUp(self):
        self.attention = importlib.import_module('micro_vllm.cuda_attention').decode_attention_gqa
        self.generator = torch.Generator(device='cuda').manual_seed(617)

    def random(self, shape, dtype):
        return torch.randn(shape, dtype=dtype, device='cuda', generator=self.generator)

    def assert_decode(self, query, key, value, backing=()):
        saved = [tensor.clone() for tensor in backing]
        expected = gqa_reference.fp64_reference(query, key, value)
        actual = self.attention(query, key, value)
        self.assertEqual(actual.shape, query.shape)
        self.assertEqual(actual.dtype, query.dtype)
        self.assertEqual(actual.device, query.device)
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.stride(-1), 1)
        self.assertEqual(actual.stride(1), query.shape[-1])
        self.assertEqual(actual.storage_offset(), 0)
        self.assertTrue(torch.isfinite(actual).all())
        tolerance = dict(rtol=2e-3, atol=2e-3) if query.dtype == torch.float16 else dict(rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(actual, expected, **tolerance)
        for tensor, original in zip(backing, saved):
            torch.testing.assert_close(tensor, original, rtol=0, atol=0, equal_nan=True)
            self.assertNotEqual(actual.untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())
        return actual

    def test_cache_prefix_views_skip_unused_capacity(self):
        for dtype in (torch.float32, torch.float16):
            for length in (31, 32, 33, 65):
                with self.subTest(dtype=dtype, length=length):
                    capacity = length + 19
                    cache = ContiguousKVCache.allocate(
                        num_layers=1, batch_size=1, num_kv_heads=2,
                        capacity=capacity, head_dim=64, dtype=dtype, device='cuda')
                    cache._keys[0].fill_(float('nan'))
                    cache._values[0].fill_(float('nan'))
                    key = self.random((1, 2, length, 64), dtype)
                    value = self.random(key.shape, dtype)
                    cache.append(0, key[:, :, :13], value[:, :, :13])
                    cache.append(0, key[:, :, 13:], value[:, :, 13:])
                    key_view, value_view = cache.read(0)
                    self.assertFalse(key_view.is_contiguous())
                    self.assertEqual(key_view.stride(1), capacity * 64)
                    self.assertEqual(value_view.stride(1), capacity * 64)
                    query = self.random((1, 14, 1, 64), dtype)
                    self.assert_decode(query, key_view, value_view,
                                       (query, cache._keys[0], cache._values[0]))
                    self.assertTrue(torch.isnan(cache._keys[0][:, :, length:]).all())
                    self.assertTrue(torch.isnan(cache._values[0][:, :, length:]).all())

    def test_independent_offset_and_padded_views(self):
        for dtype in (torch.float32, torch.float16):
            for dimension in (64, 257):
                with self.subTest(dtype=dtype, dimension=dimension):
                    length = 33
                    query_storage = torch.full((1, 30, 1, dimension), 1234., dtype=dtype, device='cuda')
                    key_storage = torch.full((1, length + 2, 4, dimension), 1234., dtype=dtype, device='cuda')
                    value_storage = torch.full((1, 6, (length + 2) * 3, dimension),
                                               1234., dtype=dtype, device='cuda')
                    query = query_storage[:, 1:29:2]
                    key = key_storage[:, 1:length + 1, 1:3].transpose(1, 2)
                    value = value_storage[:, 1:5:2, 2:2 + length * 3:3]
                    for tensor in (query, key, value):
                        tensor.copy_(self.random(tensor.shape, dtype))
                        self.assertGreater(tensor.storage_offset(), 0)
                        self.assertEqual(tensor.stride(-1), 1)
                        self.assertFalse(tensor.is_contiguous())
                    self.assertNotEqual(key.stride(), value.stride())
                    first = self.assert_decode(query, key, value,
                                               (query_storage, key_storage, value_storage))
                    preserved = first.clone()
                    second = self.attention(query, key, value)
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    torch.testing.assert_close(first, preserved, rtol=0, atol=0)

    def test_stepped_key_and_value_heads_and_tokens(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                length, dimension = 65, 64
                query_storage = self.random((1, 1, 14, dimension), dtype)
                query = query_storage.transpose(1, 2)
                key_storage = torch.full((1, 7, length * 2 + 3, dimension),
                                         777., dtype=dtype, device='cuda')
                value_storage = torch.full((1, 9, length * 3 + 4, dimension),
                                           -777., dtype=dtype, device='cuda')
                key = key_storage[:, 1:5:2, 1:1 + length * 2:2]
                value = value_storage[:, 2:8:3, 2:2 + length * 3:3]
                key.copy_(self.random(key.shape, dtype))
                value.copy_(self.random(value.shape, dtype))
                self.assertEqual(key.stride(2), 2 * dimension)
                self.assertEqual(value.stride(2), 3 * dimension)
                self.assertNotEqual(key.stride(1), value.stride(1))
                self.assert_decode(query, key, value, (query_storage, key_storage, value_storage))

    def test_zero_query_head_stride(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                query_storage = self.random((1, 1, 1, 64), dtype)
                query = query_storage.expand(1, 14, 1, 64)
                key = self.random((1, 2, 33, 64), dtype)
                value = self.random(key.shape, dtype)
                self.assertEqual(query.stride(1), 0)
                self.assert_decode(query, key, value, (query_storage, key, value))

    def test_zero_kv_head_stride_independently(self):
        for dtype in (torch.float32, torch.float16):
            for expanded in ('key', 'value', 'both'):
                with self.subTest(dtype=dtype, expanded=expanded):
                    query = self.random((1, 14, 1, 64), dtype)
                    key_storage = self.random((1, 1 if expanded in ('key', 'both') else 2, 33, 64), dtype)
                    value_storage = self.random((1, 1 if expanded in ('value', 'both') else 2, 33, 64), dtype)
                    key = key_storage.expand(1, 2, 33, 64)
                    value = value_storage.expand(1, 2, 33, 64)
                    if expanded in ('key', 'both'):
                        self.assertEqual(key.stride(1), 0)
                    if expanded in ('value', 'both'):
                        self.assertEqual(value.stride(1), 0)
                    self.assert_decode(query, key, value, (query, key_storage, value_storage))

    def test_zero_kv_token_stride_independently(self):
        for dtype in (torch.float32, torch.float16):
            for expanded in ('key', 'value', 'both'):
                with self.subTest(dtype=dtype, expanded=expanded):
                    query = self.random((1, 14, 1, 64), dtype)
                    key_storage = self.random((1, 2, 1 if expanded in ('key', 'both') else 33, 64), dtype)
                    value_storage = self.random((1, 2, 1 if expanded in ('value', 'both') else 33, 64), dtype)
                    key = key_storage.expand(1, 2, 33, 64)
                    value = value_storage.expand(1, 2, 33, 64)
                    if expanded in ('key', 'both'):
                        self.assertEqual(key.stride(2), 0)
                    if expanded in ('value', 'both'):
                        self.assertEqual(value.stride(2), 0)
                    self.assert_decode(query, key, value, (query, key_storage, value_storage))

    def test_unused_outer_dimensions_may_have_zero_stride(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                query_storage = self.random((1, 14, 1, 64), dtype)
                key_storage = self.random((1, 2, 33, 64), dtype)
                value_storage = self.random(key_storage.shape, dtype)
                query = query_storage.as_strided(query_storage.shape, (0, 64, 0, 1))
                key = key_storage.as_strided(key_storage.shape, (0, 33 * 64, 64, 1))
                value = value_storage.as_strided(value_storage.shape, (0, 33 * 64, 64, 1))
                self.assert_decode(query, key, value, (query_storage, key_storage, value_storage))

    def test_rejects_nonunit_feature_stride_for_every_argument(self):
        for dtype in (torch.float32, torch.float16):
            valid = [self.random((1, 14, 1, 64), dtype),
                     self.random((1, 2, 33, 64), dtype), self.random((1, 2, 33, 64), dtype)]
            for index, tensor in enumerate(valid):
                with self.subTest(dtype=dtype, argument=index):
                    shape = (*tensor.shape[:-1], 128)
                    invalid = self.random(shape, dtype)[..., ::2]
                    args = valid.copy()
                    args[index] = invalid
                    with self.assertRaisesRegex(RuntimeError, 'feature stride'):
                        self.attention(*args)


if __name__ == '__main__':
    unittest.main()

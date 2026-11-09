"""Fixed-table paged storage agrees with contiguous logical token order."""
import unittest
import torch

from micro_vllm.kv_cache import ContiguousKVCache

try:
    from micro_vllm.paged_kv_cache import PagedKVCache
except ImportError:
    PagedKVCache = None


class PagedKVCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(PagedKVCache, 'the fixed-table paged cache must exist')

    def cache(self, **kwargs):
        options = dict(num_layers=2, num_kv_heads=2, num_pages=7, page_size=4,
                       head_dim=6, dtype=torch.float32, device='cpu', page_table=[4, 1, 5])
        options.update(kwargs)
        return PagedKVCache.allocate(**options)

    def chunk(self, count, dtype=torch.float32):
        key = torch.arange(2 * count * 6, dtype=dtype).reshape(1, 2, count, 6)
        return key, -key - 1

    def poison(self, cache):
        for buffer in cache._keys + cache._values:
            buffer.fill_(float('nan'))

    def assert_unchanged(self, cache, snapshots, lengths):
        for actual, expected in zip(cache._keys + cache._values, snapshots):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertEqual([cache.length(i) for i in range(cache.num_layers)], lengths)

    def snapshot(self, cache):
        return [buffer.clone() for buffer in cache._keys + cache._values]

    def test_allocation_metadata_identity_mapping_and_empty_gather(self):
        cache = self.cache(page_table=None)
        self.assertEqual(cache.page_table, tuple(range(7)))
        self.assertEqual((cache.num_layers, cache.batch_size, cache.num_pages,
                          cache.num_kv_heads, cache.page_size, cache.head_dim,
                          cache.capacity), (2, 1, 7, 2, 4, 6, 28))
        self.assertEqual(cache.dtype, torch.float32)
        self.assertEqual(cache.device, torch.device('cpu'))
        self.assertEqual(cache.sequence_length, 0)
        for buffer in cache._keys + cache._values:
            self.assertEqual(tuple(buffer.shape), (7, 2, 4, 6))
            self.assertTrue(buffer.is_contiguous())
        for tensor in cache.gather(0):
            self.assertEqual(tuple(tensor.shape), (1, 2, 0, 6))

    def test_page_mapping_is_a_defensive_immutable_snapshot(self):
        mapping = [4, 1, 5]
        cache = self.cache(page_table=mapping)
        mapping[0] = 0
        self.assertEqual(cache.page_table, (4, 1, 5))
        self.assertEqual(cache._device_page_table.tolist(), [4, 1, 5])
        with self.assertRaises(AttributeError):
            cache.page_table = (0,)
        self.assertEqual(cache.capacity, 12)

    def test_shuffled_pages_boundary_chunks_and_contiguous_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                cache = self.cache(dtype=dtype)
                self.poison(cache)
                reference = ContiguousKVCache.allocate(
                    num_layers=2, batch_size=1, num_kv_heads=2, capacity=12,
                    head_dim=6, dtype=dtype, device='cpu')
                key, value = self.chunk(9, dtype)
                saved = (key.clone(), value.clone())
                offset = 0
                for count in (3, 4, 2):
                    self.assertIs(cache.append(0, key[:, :, offset:offset + count],
                                               value[:, :, offset:offset + count]), cache)
                    reference.append(0, key[:, :, offset:offset + count],
                                     value[:, :, offset:offset + count])
                    for actual, expected in zip(cache.gather(0), reference.read(0)):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    offset += count
                for actual, expected in zip((key, value), saved):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for pages, source in zip((cache._keys[0], cache._values[0]), (key, value)):
                    expected = torch.full_like(pages, float('nan'))
                    for position in range(9):
                        physical = (4, 1, 5)[position // 4]
                        expected[physical, :, position % 4] = source[0, :, position]
                    torch.testing.assert_close(pages, expected, rtol=0, atol=0, equal_nan=True)
                self.assertTrue(torch.isnan(cache._keys[1]).all())

    def test_gather_allocates_independent_storage(self):
        cache = self.cache()
        key, value = self.chunk(5)
        cache.append(0, key, value)
        gathered = cache.gather(0)
        for tensor, pages in zip(gathered, (cache._keys[0], cache._values[0])):
            self.assertNotEqual(tensor.untyped_storage().data_ptr(),
                                pages.untyped_storage().data_ptr())
            with torch.inference_mode():
                tensor.fill_(999)
        for actual, expected in zip(cache.gather(0), (key, value)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_layers_have_independent_storage_and_aligned_sequence_length(self):
        cache = self.cache()
        key, value = self.chunk(3)
        cache.append(0, key, value)
        self.assertEqual((cache.length(0), cache.length(1)), (3, 0))
        with self.assertRaisesRegex(ValueError, 'aligned'):
            _ = cache.sequence_length
        cache.append(1, key + 100, value - 100)
        self.assertEqual(cache.sequence_length, 3)
        torch.testing.assert_close(cache.gather(0)[0], key, rtol=0, atol=0)
        torch.testing.assert_close(cache.gather(1)[0], key + 100, rtol=0, atol=0)
        pointers = [p.data_ptr() for p in cache._keys + cache._values]
        self.assertEqual(len(set(pointers)), 4)

    def test_reset_and_shorter_reuse_preserve_allocations_and_table(self):
        cache = self.cache()
        buffers = cache._keys + cache._values + (cache._device_page_table,)
        pointers = [p.data_ptr() for p in buffers]
        cache.append(0, *self.chunk(9))
        self.assertIs(cache.reset(), cache)
        self.assertEqual(cache.sequence_length, 0)
        key, value = self.chunk(2)
        cache.append(0, key + 7, value - 7)
        self.assertEqual(cache.length(0), 2)
        self.assertEqual([p.data_ptr() for p in buffers], pointers)
        self.assertEqual(cache.page_table, (4, 1, 5))
        torch.testing.assert_close(cache.gather(0)[0], key + 7, rtol=0, atol=0)

    def test_exact_capacity_and_overflow_are_checked_before_writes(self):
        cache = self.cache()
        cache.append(0, *self.chunk(12))
        snapshots = self.snapshot(cache)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            cache.append(0, *self.chunk(1))
        self.assert_unchanged(cache, snapshots, [12, 0])

    def test_independent_noncontiguous_source_strides_and_broadcast_reads(self):
        cache = self.cache()
        key = torch.arange(60.).reshape(1, 5, 2, 6).transpose(1, 2)
        value = torch.arange(180.).reshape(1, 2, 15, 6)[:, :, 1::3]
        self.assertFalse(key.is_contiguous())
        self.assertNotEqual(key.stride(), value.stride())
        cache.append(0, key, value)
        for actual, expected in zip(cache.gather(0), (key, value)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        cache.reset()
        key = torch.arange(30.).reshape(1, 1, 5, 6).expand(1, 2, 5, 6)
        cache.append(0, key, value)
        torch.testing.assert_close(cache.gather(0)[0], key, rtol=0, atol=0)

    def test_invalid_dimensions_dtype_device_and_page_mappings(self):
        for name in ('num_layers', 'num_kv_heads', 'num_pages', 'page_size', 'head_dim'):
            for bad in (0, -1, True, 1.5):
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    self.cache(**{name: bad})
        for dtype in (torch.int64, torch.float64):
            with self.assertRaises(ValueError):
                self.cache(dtype=dtype)
        with self.assertRaises(ValueError):
            self.cache(device='meta')
        for mapping in ([], [1, 1], [-1], [7], [1.5], [True], '123',
                        torch.tensor([1, 2]), iter([1, 2])):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.cache(page_table=mapping)

    def test_invalid_append_inputs_preserve_data_and_lengths(self):
        cache = self.cache()
        self.poison(cache)
        cache.append(0, *self.chunk(1))
        snapshots = self.snapshot(cache)
        key, value = self.chunk(2)
        pairs = [
            (None, value), (key, None), (key[0], value[0]),
            (key.expand(2, -1, -1, -1), value.expand(2, -1, -1, -1)),
            (key[:, :1], value[:, :1]), (key[..., :5], value[..., :5]),
            (key[:, :, :0], value[:, :, :0]), (key, value[:, :, :1]),
            (key.half(), value.half()), (key, value.half()),
            (key.detach().requires_grad_(), value),
            (key, value.detach().requires_grad_()),
            (key.to_sparse(), value.to_sparse()),
            (torch.zeros(1, 2, 2, 12)[..., ::2], value),
            (key.to('meta'), value.to('meta')),
            (torch._neg_view(key), value),
        ]
        for index, pair in enumerate(pairs):
            with self.subTest(index=index), self.assertRaises(ValueError):
                cache.append(0, *pair)
            self.assert_unchanged(cache, snapshots, [1, 0])

    def test_invalid_layer_indices_are_rejected(self):
        cache = self.cache()
        for layer in (-1, 2, True, 1.5, None):
            for operation in (cache.length, cache.gather):
                with self.subTest(layer=layer), self.assertRaises(ValueError):
                    operation(layer)
            with self.assertRaises(ValueError):
                cache.append(layer, *self.chunk(1))

    def test_dlpack_offset_alias_is_rejected_before_any_write(self):
        cache = self.cache()
        self.poison(cache)
        snapshots = self.snapshot(cache)
        key, value = self.chunk(2)
        for pages in (cache._keys[0], cache._values[0]):
            source = torch.utils.dlpack.from_dlpack(pages[4, :, :2].unsqueeze(0))
            self.assertNotEqual(source.untyped_storage().data_ptr(),
                                pages.untyped_storage().data_ptr())
            self.assertFalse(torch._C._is_alias_of(source, pages))
            for pair in ((source, value), (key, source)):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    cache.append(0, *pair)
                self.assert_unchanged(cache, snapshots, [0, 0])

    def test_sources_may_not_alias_either_destination_buffer(self):
        cache = self.cache()
        self.poison(cache)
        snapshots = self.snapshot(cache)
        key, value = self.chunk(2)
        for pages in (cache._keys[0], cache._values[0]):
            alias = pages[4, :, :2].unsqueeze(0)
            for pair in ((alias, value), (key, alias)):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    cache.append(0, *pair)
                self.assert_unchanged(cache, snapshots, [0, 0])


if __name__ == '__main__':
    unittest.main()

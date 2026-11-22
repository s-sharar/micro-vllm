"""Shared KV pool ownership, dynamic request page tables, and safe page reuse."""
import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from micro_vllm.kv_cache import ContiguousKVCache
from micro_vllm.kv_pool import KVPool, KVPoolExhausted, PageState
from micro_vllm.paged_kv_cache import PagedKVCache

FREE, OWNED, RETIRING = PageState.FREE, PageState.OWNED, PageState.RETIRING


def make_pool(device='cpu', **kwargs):
    options = dict(num_layers=2, num_kv_heads=2, num_pages=6, page_size=3,
                   head_dim=4, dtype=torch.float32, device=device)
    options.update(kwargs)
    return KVPool.allocate(**options)


def chunk(count, start=0, device='cpu', dtype=torch.float32):
    key = torch.arange(start * 8, (start + count) * 8, dtype=dtype, device=device)
    key = key.reshape(1, count, 2, 4).transpose(1, 2)
    return key, -key - 1


def append_all(cache, count, start=0):
    for layer in range(cache.num_layers):
        cache.append(layer, *chunk(count, start, cache.device, cache.dtype))


class KVPoolTests(unittest.TestCase):
    def states(self, pool):
        return [pool.state(page) for page in range(pool.num_pages)]

    def test_allocation_metadata_and_initial_free_pages(self):
        pool = make_pool()
        self.assertEqual((pool.num_layers, pool.num_kv_heads, pool.num_pages, pool.page_size,
                          pool.head_dim, pool.token_capacity, pool.num_free),
                         (2, 2, 6, 3, 4, 18, 6))
        self.assertEqual((pool.dtype, pool.device), (torch.float32, torch.device('cpu')))
        self.assertEqual(self.states(pool), [FREE] * 6)
        for buffer in pool._keys + pool._values:
            self.assertEqual(tuple(buffer.shape), (6, 2, 3, 4))
            self.assertTrue(buffer.is_contiguous())
        self.assertEqual(len({buffer.data_ptr() for buffer in pool._keys + pool._values}), 4)

    def test_allocation_rejects_invalid_geometry_dtype_and_device(self):
        for options in ({'num_pages': 0}, {'page_size': True}, {'head_dim': 1.5},
                        {'num_layers': -1}, {'dtype': torch.int32}, {'device': 'meta'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_pool(**options)

    def test_requests_share_buffers_and_own_independent_tables(self):
        pool = make_pool()
        first = PagedKVCache.for_pool(pool, max_tokens=7)
        second = PagedKVCache.for_pool(pool, max_tokens=4)
        self.assertIs(first._keys, pool._keys)
        self.assertIs(second._values, pool._values)
        self.assertEqual((first.capacity, second.capacity), (9, 6))
        self.assertEqual((first.page_table, second.page_table), ((), ()))
        self.assertEqual(first._device_page_table.tolist(), [-1, -1, -1])
        self.assertEqual(first.num_pages, 6)
        self.assertEqual(self.states(pool), [FREE] * 6)

    def test_request_capacity_must_fit_the_whole_pool(self):
        pool = make_pool()
        PagedKVCache.for_pool(pool, max_tokens=18)
        for tokens in (19, 0, -1, True, 2.0):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                PagedKVCache.for_pool(pool, max_tokens=tokens)
        with self.assertRaises(ValueError):
            PagedKVCache.for_pool(object(), max_tokens=3)

    def test_page_boundary_growth_reserves_each_forward_in_one_operation(self):
        pool = make_pool()
        cache = PagedKVCache.for_pool(pool, max_tokens=13)
        with patch.object(pool, '_reserve', wraps=pool._reserve) as reserve:
            self.assertIs(cache.reserve(2), cache)
            self.assertEqual(cache.page_table, (0,))
            append_all(cache, 2)
            cache.reserve(1)
            self.assertEqual(cache.page_table, (0,))
            append_all(cache, 1, 2)
            cache.reserve(7)
            self.assertEqual(cache.page_table, (0, 1, 2, 3))
        self.assertEqual([call.args[1] for call in reserve.call_args_list], [1, 3])
        self.assertEqual(cache._device_page_table.tolist(), [0, 1, 2, 3, -1])
        self.assertEqual(self.states(pool), [OWNED] * 4 + [FREE] * 2)
        append_all(cache, 7, 3)
        self.assertEqual(cache.sequence_length, 10)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            cache.reserve(6)

    def test_append_requires_reserved_pages(self):
        pool = make_pool()
        cache = PagedKVCache.for_pool(pool, max_tokens=9)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            cache.append(0, *chunk(1))
        cache.reserve(3)
        cache.append(0, *chunk(3))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            cache.append(0, *chunk(1, 3))
        self.assertEqual(cache.length(0), 3)

    def test_reservation_requires_aligned_layers(self):
        cache = PagedKVCache.for_pool(make_pool(), max_tokens=9)
        cache.reserve(2)
        cache.append(0, *chunk(2))
        with self.assertRaisesRegex(ValueError, 'aligned'):
            cache.reserve(2)
        self.assertEqual(cache.page_table, (0,))

    def test_failed_reservation_is_all_or_nothing(self):
        pool = make_pool()
        holder = PagedKVCache.for_pool(pool, max_tokens=12).reserve(12)
        request = PagedKVCache.for_pool(pool, max_tokens=9).reserve(3)
        append_all(request, 3)
        table = request._device_page_table.clone()
        with self.assertRaises(KVPoolExhausted):
            request.reserve(6)
        self.assertTrue(issubclass(KVPoolExhausted, RuntimeError))
        self.assertEqual(request.page_table, (4,))
        self.assertTrue(torch.equal(request._device_page_table, table))
        self.assertEqual(self.states(pool), [OWNED] * 5 + [FREE])
        self.assertEqual(pool.num_free, 1)
        request.reserve(3)
        self.assertEqual(request.page_table, (4, 5))
        self.assertEqual(holder.page_table, (0, 1, 2, 3))

    def test_cpu_release_frees_immediately_in_fifo_order_and_is_idempotent(self):
        pool = make_pool()
        first = PagedKVCache.for_pool(pool, max_tokens=9).reserve(9)
        second = PagedKVCache.for_pool(pool, max_tokens=6).reserve(6)
        append_all(first, 4)
        self.assertIsNone(second.release())
        self.assertIsNone(first.release())
        first.release()
        self.assertEqual(self.states(pool), [FREE] * 6)
        self.assertEqual((first.page_table, first.sequence_length), ((), 0))
        third = PagedKVCache.for_pool(pool, max_tokens=18).reserve(18)
        self.assertEqual(third.page_table, (5, 3, 4, 0, 1, 2))
        for operation in (lambda: first.reserve(1), lambda: first.append(0, *chunk(1))):
            with self.assertRaisesRegex(ValueError, 'released'):
                operation()

    def test_failed_publication_returns_new_pages_and_requires_release(self):
        pool = make_pool()
        cache = PagedKVCache.for_pool(pool, max_tokens=12).reserve(3)
        append_all(cache, 3)
        table = cache._device_page_table.clone()
        with (patch.object(cache, '_publish', side_effect=RuntimeError('copy failed')),
              self.assertRaisesRegex(RuntimeError, 'copy failed')):
            cache.reserve(6)
        self.assertEqual(cache.page_table, (0,))
        self.assertTrue(torch.equal(cache._device_page_table, table))
        self.assertEqual(self.states(pool), [OWNED] + [FREE] * 5)
        for operation in (lambda: cache.reserve(1), lambda: cache.append(0, *chunk(1, 3))):
            with self.assertRaisesRegex(ValueError, 'release'):
                operation()
        cache.release()
        cache.release()
        self.assertEqual(self.states(pool), [FREE] * 6)

    def test_invalid_arguments_and_empty_release_leave_pool_unchanged(self):
        pool = make_pool()
        cache = PagedKVCache.for_pool(pool, max_tokens=6)
        for page in (-1, 6, True, 1.0):
            with self.subTest(page=page), self.assertRaises(ValueError):
                pool.state(page)
        for tokens in (0, -1, True, 1.5):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                cache.reserve(tokens)
        cache.release()
        fixed = PagedKVCache.allocate(num_layers=1, num_kv_heads=2, num_pages=2, page_size=3,
                                      head_dim=4, dtype=torch.float32, device='cpu')
        with self.assertRaisesRegex(ValueError, 'free'):
            fixed._pool._claim(object(), [0])
        self.assertEqual(self.states(pool), [FREE] * 6)
        self.assertEqual(pool.num_free, 6)

    def test_release_rejects_pages_not_owned_by_the_request(self):
        pool = make_pool()
        cache = PagedKVCache.for_pool(pool, max_tokens=3).reserve(3)
        with self.assertRaisesRegex(RuntimeError, 'own'):
            pool._release(object(), [0], None, ())
        cache.release()
        with self.assertRaisesRegex(RuntimeError, 'own'):
            pool._release(cache._owner, [0], None, ())

    def test_gather_follows_dynamic_non_identity_mapping_and_valid_length(self):
        pool = make_pool()
        request = PagedKVCache.for_pool(pool, max_tokens=12)
        other = PagedKVCache.for_pool(pool, max_tokens=6)
        reference = ContiguousKVCache.allocate(num_layers=2, batch_size=1, num_kv_heads=2,
                                               capacity=12, head_dim=4, dtype=torch.float32,
                                               device='cpu')
        for pages in pool._keys + pool._values:
            pages.fill_(float('nan'))
        offset = 0
        for count in (2, 2, 3):
            request.reserve(count)
            if offset == 0:
                other.reserve(1)
            for target in (request, reference):
                append_all(target, count, offset)
            offset += count
            for layer in range(2):
                for actual, expected in zip(request.gather(layer), reference.read(layer)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(request.page_table, (0, 2, 3))
        self.assertEqual(other.page_table, (1,))
        self.assertEqual(request._device_page_table.tolist(), [0, 2, 3, -1])

    def test_fixed_allocate_remains_a_private_pool_with_owned_mapping(self):
        cache = PagedKVCache.allocate(num_layers=1, num_kv_heads=2, num_pages=7, page_size=4,
                                      head_dim=4, dtype=torch.float32, device='cpu',
                                      page_table=[4, 1, 5])
        pool = cache._pool
        self.assertEqual([pool.state(page) for page in (4, 1, 5)], [OWNED] * 3)
        self.assertEqual(pool.num_free, 4)
        self.assertIs(cache.reserve(12), cache)
        self.assertEqual(cache.page_table, (4, 1, 5))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            cache.reserve(13)
        cache.release()
        self.assertEqual(pool.num_free, 7)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class CUDAKVPoolTests(unittest.TestCase):
    SLEEP_CYCLES = 400_000_000

    @classmethod
    def setUpClass(cls):
        from micro_vllm import cuda_attention, cuda_paged_cache

        cuda_paged_cache._load_extension()
        cuda_attention._load_extension()

    def states(self, pool):
        return [pool.state(page) for page in range(pool.num_pages)]

    def test_release_retires_until_the_release_event_completes(self):
        pool = make_pool('cuda', dtype=torch.float16)
        cache = PagedKVCache.for_pool(pool, max_tokens=6).reserve(6)
        append_all(cache, 6)
        torch.cuda._sleep(self.SLEEP_CYCLES)
        cache.release()
        self.assertEqual(self.states(pool), [RETIRING] * 2 + [FREE] * 4)
        self.assertEqual(pool.reclaim(), 0)
        self.assertEqual(pool.num_free, 4)
        torch.cuda.synchronize()
        self.assertEqual(pool.reclaim(), 2)
        self.assertEqual(self.states(pool), [FREE] * 6)
        self.assertEqual(pool.reclaim(), 0)

    def test_reservation_reclaims_then_waits_for_oldest_retirement(self):
        pool = make_pool('cuda')
        first = PagedKVCache.for_pool(pool, max_tokens=9).reserve(9)
        second = PagedKVCache.for_pool(pool, max_tokens=9).reserve(9)
        torch.cuda._sleep(self.SLEEP_CYCLES)
        first.release()
        second.release()
        self.assertEqual(pool.num_free, 0)
        request = PagedKVCache.for_pool(pool, max_tokens=6)
        with patch.object(torch.cuda.Event, 'synchronize', autospec=True,
                          side_effect=torch.cuda.Event.synchronize) as wait:
            request.reserve(6)
        self.assertEqual(wait.call_count, 1)
        self.assertEqual(request.page_table, (0, 1))
        self.assertEqual(self.states(pool)[:3], [OWNED, OWNED, FREE])
        self.assertEqual(self.states(pool)[3:], [RETIRING] * 3)

    def test_exhaustion_after_waiting_preserves_state(self):
        pool = make_pool('cuda')
        holder = PagedKVCache.for_pool(pool, max_tokens=12).reserve(12)
        retiring = PagedKVCache.for_pool(pool, max_tokens=3).reserve(3)
        torch.cuda._sleep(self.SLEEP_CYCLES)
        retiring.release()
        request = PagedKVCache.for_pool(pool, max_tokens=9)
        table = request._device_page_table.clone()
        with self.assertRaises(KVPoolExhausted):
            request.reserve(9)
        self.assertEqual(request.page_table, ())
        self.assertTrue(torch.equal(request._device_page_table, table))
        self.assertEqual(self.states(pool), [OWNED] * 4 + [FREE] * 2)
        self.assertEqual(holder.page_table, (0, 1, 2, 3))

    def test_table_publication_is_stream_ordered_before_native_writes(self):
        pool = make_pool('cuda')
        cache = PagedKVCache.for_pool(pool, max_tokens=12)
        reference = PagedKVCache.for_pool(make_pool(), max_tokens=12)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(self.SLEEP_CYCLES)
            for count, start in ((2, 0), (5, 2), (3, 7)):
                cache.reserve(count)
                append_all(cache, count, start)
                reference.reserve(count)
                append_all(reference, count, start)
            staging = cache._staging.clone()
        stream.synchronize()
        self.assertTrue(cache._staging.is_pinned())
        self.assertTrue(torch.equal(staging, cache._staging))
        self.assertEqual(cache._device_page_table.tolist(), [0, 1, 2, 3])
        with torch.cuda.stream(stream):
            for layer in range(2):
                for actual, expected in zip(cache.gather(layer), reference.gather(layer)):
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_request_binds_to_first_stream_and_rejects_others_before_work(self):
        from micro_vllm.cuda_attention import decode_attention_paged

        pool = make_pool('cuda', dtype=torch.float16, head_dim=64)
        cache = PagedKVCache.for_pool(pool, max_tokens=9)
        filler = torch.ones((1, 2, 2, 64), device='cuda', dtype=torch.float16)
        query = torch.ones((1, 14, 1, 64), device='cuda', dtype=torch.float16)
        bound, other = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(bound):
            cache.reserve(2)
            cache.append(0, filler, filler)
            cache.append(1, filler, filler)
        operations = (lambda: cache.reserve(1), lambda: cache.append(0, filler, filler),
                      lambda: cache.gather(0),
                      lambda: decode_attention_paged(query, cache, layer_idx=0))
        for stream in (other, torch.cuda.default_stream()):
            for operation in operations:
                with self.subTest(stream=stream), torch.cuda.stream(stream):
                    with self.assertRaisesRegex(ValueError, 'stream'):
                        operation()
        self.assertEqual((cache.page_table, cache.sequence_length), ((0,), 2))
        with torch.cuda.stream(bound):
            cache.reserve(1)
            self.assertEqual(decode_attention_paged(query, cache, layer_idx=1).shape, query.shape)
            cache.gather(0)

    def test_release_records_on_the_bound_stream_after_standalone_decode(self):
        from micro_vllm.cuda_attention import decode_attention_paged

        pool = make_pool('cuda', dtype=torch.float16, head_dim=64)
        cache = PagedKVCache.for_pool(pool, max_tokens=6)
        filler = torch.ones((1, 2, 3, 64), device='cuda', dtype=torch.float16)
        bound = torch.cuda.Stream()
        with torch.cuda.stream(bound):
            cache.reserve(3)
            cache.append(0, filler, filler)
            torch.cuda._sleep(self.SLEEP_CYCLES)
            decode_attention_paged(torch.ones((1, 2, 1, 64), device='cuda',
                                              dtype=torch.float16), cache, layer_idx=0)
        cache.release()
        self.assertEqual(pool.reclaim(), 0)
        self.assertEqual(pool.state(0), RETIRING)
        bound.synchronize()
        self.assertEqual(pool.reclaim(), 1)

    def test_release_from_another_stream_waits_for_the_bound_stream(self):
        pool = make_pool('cuda')
        cache = PagedKVCache.for_pool(pool, max_tokens=6)
        side, releaser = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(side):
            cache.reserve(6)
            append_all(cache, 6)
            torch.cuda._sleep(self.SLEEP_CYCLES)
        with torch.cuda.stream(releaser):
            cache.release()
        self.assertEqual(pool.reclaim(), 0)
        self.assertEqual(self.states(pool)[:2], [RETIRING] * 2)
        side.synchronize()
        releaser.synchronize()
        self.assertEqual(pool.reclaim(), 2)

    def test_failed_cuda_publication_retires_new_pages_on_the_bound_stream(self):
        pool = make_pool('cuda')
        cache = PagedKVCache.for_pool(pool, max_tokens=12).reserve(3)
        torch.cuda._sleep(self.SLEEP_CYCLES)
        with (patch.object(cache, '_publish', side_effect=RuntimeError('copy failed')),
              self.assertRaises(RuntimeError)):
            cache.reserve(9)
        self.assertEqual(self.states(pool), [OWNED] + [RETIRING] * 2 + [FREE] * 3)
        self.assertEqual(cache.page_table, (0,))
        cache.release()
        torch.cuda.synchronize()
        self.assertEqual(pool.reclaim(), 3)
        self.assertEqual(self.states(pool), [FREE] * 6)

    def test_retirement_keeps_request_table_and_staging_alive(self):
        pool = make_pool('cuda')
        cache = PagedKVCache.for_pool(pool, max_tokens=6)
        torch.cuda._sleep(self.SLEEP_CYCLES)
        cache.reserve(6)
        staging, table = weakref.ref(cache._staging), weakref.ref(cache._device_page_table)
        cache.release()
        del cache
        gc.collect()
        self.assertIsNotNone(staging())
        self.assertIsNotNone(table())
        torch.cuda.synchronize()
        pool.reclaim()
        gc.collect()
        self.assertIsNone(staging())
        self.assertIsNone(table())

    def test_native_paged_decode_reads_dynamic_non_identity_mapping(self):
        from micro_vllm.cuda_attention import decode_attention_gqa, decode_attention_paged

        pool = make_pool('cuda', dtype=torch.float16, num_pages=9, head_dim=64)
        request = PagedKVCache.for_pool(pool, max_tokens=13)
        other = PagedKVCache.for_pool(pool, max_tokens=12)
        for pages in pool._keys + pool._values:
            pages.fill_(float('nan'))
        keys = torch.randn((1, 2, 13, 64), device='cuda', dtype=torch.float16)
        values = torch.randn_like(keys)
        filler = torch.zeros((1, 2, 3, 64), device='cuda', dtype=torch.float16)
        offset = 0
        for count in (4, 1, 3, 5):
            request.reserve(count)
            other.reserve(3)
            for layer in range(2):
                request.append(layer, keys[:, :, offset:offset + count],
                               values[:, :, offset:offset + count])
                other.append(layer, filler, filler)
            offset += count
            query = torch.randn((1, 14, 1, 64), device='cuda', dtype=torch.float16)
            actual = decode_attention_paged(query, request, layer_idx=1)
            expected = decode_attention_gqa(query, keys[:, :, :offset], values[:, :, :offset])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(request.page_table, (0, 1, 4, 6, 7))
        self.assertEqual(other.page_table, (2, 3, 5, 8))


if __name__ == '__main__':
    unittest.main()

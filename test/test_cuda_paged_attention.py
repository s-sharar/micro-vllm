"""Native-only paged decode: page addressing, shared math, and metadata safety."""
import unittest
import subprocess
import sys

import torch

from micro_vllm import cuda_attention
from test.test_cuda_gqa import fp64_reference


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class NativePagedAttentionTests(unittest.TestCase):
    def setUp(self):
        self.extension = cuda_attention._load_extension()
        self.assertTrue(hasattr(self.extension, '_decode_attention_paged'),
                        'provide the internal paged operation in the attention extension')
        self.attention = self.extension._decode_attention_paged

    def inputs(self, length=65, block_size=4, heads=(6, 2), dimension=33,
               dtype=torch.float32):
        hq, hkv = heads
        generator = torch.Generator(device='cuda').manual_seed(519)
        query = torch.randn(1, hq, 1, dimension, dtype=dtype,
                            device='cuda', generator=generator)
        key = torch.randn(1, hkv, length, dimension, dtype=dtype,
                          device='cuda', generator=generator)
        value = torch.randn(key.shape, dtype=dtype, device='cuda', generator=generator)
        count = (length + block_size - 1) // block_size
        # Physical IDs are shuffled and leave three entire pages unassigned.
        ids = list(range(count + 3))[2:count + 2][::-1]
        pages = torch.full((count + 3, hkv, block_size, dimension), float('nan'),
                           dtype=dtype, device='cuda')
        values = torch.full_like(pages, float('nan'))
        for logical, physical in enumerate(ids):
            start, stop = logical * block_size, min((logical + 1) * block_size, length)
            pages[physical, :, :stop - start] = key[0, :, start:stop]
            values[physical, :, :stop - start] = value[0, :, start:stop]
        table = torch.tensor(ids, dtype=torch.int64, device='cuda')
        return query, pages, values, table, key, value

    def assert_matches(self, query, pages, values, table, key, value):
        actual = self.attention(query, pages, values, table, key.shape[2])
        tolerance = dict(rtol=2e-3, atol=2e-3) if query.dtype == torch.float16 else dict(rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(actual, fp64_reference(query, key, value), **tolerance)
        # Identical math/order is shared by the two KV address readers.
        torch.testing.assert_close(actual, cuda_attention.decode_attention_gqa(query, key, value),
                                   rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(actual.is_contiguous())
        self.assertEqual((actual.shape, actual.dtype, actual.device),
                         (query.shape, query.dtype, query.device))
        return actual

    def test_internal_native_binding_is_available(self):
        self.assertTrue(hasattr(self.extension, '_decode_attention_paged'))

    def test_shuffled_pages_partial_pages_and_online_tile_boundaries(self):
        for dtype in (torch.float32, torch.float16):
            for block_size in (3, 4, 16, 64):
                lengths = sorted({block_size - 1, block_size, block_size + 1,
                                  31, 32, 33, 65, 257})
                for length in lengths:
                    with self.subTest(dtype=dtype, block=block_size, length=length):
                        self.assert_matches(*self.inputs(length, block_size, dtype=dtype))

    def test_generic_gqa_and_feature_tails(self):
        for dtype in (torch.float32, torch.float16):
            for heads in ((1, 1), (4, 4), (6, 2), (14, 2), (14, 1)):
                for dimension in (33, 257):
                    with self.subTest(dtype=dtype, heads=heads, dimension=dimension):
                        self.assert_matches(*self.inputs(heads=heads, dimension=dimension,
                                                         dtype=dtype))

    def test_independent_query_strides_including_broadcast_heads(self):
        query, pages, values, table, key, value = self.inputs()
        storage = torch.empty(1, 15, 2, 40, device='cuda')
        strided = storage[:, 1:13:2, 1:2, 3:36]
        strided.copy_(query)
        self.assertFalse(strided.is_contiguous())
        self.assert_matches(strided, pages, values, table, key, value)
        broadcast = query[:, :1].expand_as(query)
        self.assertEqual(broadcast.stride(1), 0)
        self.assert_matches(broadcast, pages, values, table, key, value)

    def test_inputs_unchanged_outputs_fresh_and_readonly_aliases_allowed(self):
        query, pages, values, table, key, value = self.inputs()
        originals = [tensor.clone() for tensor in (query, pages, values, table)]
        first = self.assert_matches(query, pages, values, table, key, value)
        saved = first.clone()
        second = self.attention(query, pages, values, table, key.shape[2])
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        for tensor, original in zip((query, pages, values, table), originals):
            torch.testing.assert_close(tensor, original, rtol=0, atol=0, equal_nan=True)
            self.assertNotEqual(first.data_ptr(), tensor.data_ptr())
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
        self.assert_matches(query, pages, pages, table, key, key)
        # Q may also be a read-only slice of a page used by K and V.
        alias_query = pages[table[0], :, :1].unsqueeze(0)
        self.assert_matches(alias_query, pages, pages, table, key, key)

    def test_normalized_state_keeps_large_values_finite(self):
        query, pages, values, table, key, value = self.inputs()
        query.zero_()
        key.zero_()
        pages.zero_()
        for head, sign in ((0, 1), (1, -1)):
            values[:, head].fill_(sign * 2e38)
            value[:, head].fill_(sign * 2e38)
        actual = self.attention(query, pages, values, table, key.shape[2])
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, fp64_reference(query, key, value), rtol=5e-6, atol=0)

    def test_current_nondefault_stream_orders_table_and_pages(self):
        query, pages, values, table, key, value = self.inputs()
        self.attention(query, pages, values, table, 65)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10000000)
            query.zero_()
            pages.zero_()
            values.fill_(7)
            table.copy_(table.flip(0))
            actual = self.attention(query, pages, values, table, 65) * 2
        stream.synchronize()
        torch.testing.assert_close(actual, torch.full_like(query, 14), rtol=2e-5, atol=2e-5)

    def test_invalid_used_private_page_ids_cannot_read_out_of_bounds(self):
        query, pages, values, table, _, _ = self.inputs()
        for invalid in (-1, pages.shape[0], 2**63 - 1):
            bad_table = table.clone()
            bad_table[8] = invalid  # A later tile must also guard before K/V reads.
            actual = self.attention(query, pages, values, bad_table, 65)
            self.assertTrue(torch.isnan(actual).all())

    def test_rejects_invalid_native_metadata_before_launch(self):
        query, pages, values, table, _, _ = self.inputs()
        valid = [query, pages, values, table, 65]
        cases = []
        for index, tensor in enumerate(valid[:4]):
            cases.extend([(index, tensor.cpu(), 'CUDA'),
                          (index, torch._neg_view(tensor), 'negative views'),
                          (index, tensor.to_sparse(), 'strided')])
        for index, tensor in enumerate(valid[:3]):
            cases.extend([(index, tensor.double(), 'float32 or float16'),
                          (index, tensor.detach().requires_grad_(), 'gradients')])
        cases.extend([
            (0, query.half(), 'same dtype'), (1, pages.half(), 'same dtype'),
            (2, values.half(), 'same dtype'), (0, query.squeeze(0), 'shape'),
            (0, query.repeat(2, 1, 1, 1), 'batch'),
            (0, query.repeat(1, 1, 2, 1), 'one query'),
            (0, query[:, :0], 'positive'),
            (0, query[:, :3], 'divisible'), (0, query[..., :32], 'head dimension'),
            (0, torch.empty(1, 6, 1, 66, device='cuda')[..., ::2], 'feature stride'),
            (1, pages[:, :, :0].contiguous(), 'positive'),
            (1, pages[:0], 'positive'),
            (1, pages[:, :0].contiguous(), 'positive'),
            (1, pages[..., :0].contiguous(), 'positive'),
            (1, pages[..., :32], 'contiguous'),
            (2, values[:1], 'same shape'),
            (1, pages.squeeze(1).flatten(0, 1), 'shape'),
            (3, table.int(), 'int64'),
            (3, table.unsqueeze(0), 'one-dimensional'),
            (3, table[:0], 'positive'),
            (3, torch.zeros((pages.shape[0] + 1,), device='cuda', dtype=torch.int64), 'physical pages'),
            (3, table.repeat_interleave(2)[::2], 'contiguous'),
            (4, 0, 'positive'), (4, -1, 'positive'),
            (4, table.numel() * pages.shape[2] + 1, 'capacity'),
        ])
        for index, invalid, message in cases:
            args = valid.copy()
            args[index] = invalid
            with self.subTest(argument=index, message=message), self.assertRaisesRegex(RuntimeError, message):
                self.attention(*args)

    @unittest.skipUnless(torch.cuda.device_count() > 1, 'multiple CUDA GPUs required')
    def test_rejects_mixed_devices_for_every_native_argument(self):
        query, pages, values, table, _, _ = self.inputs()
        valid = [query, pages, values, table, 65]
        for index in range(4):
            args = valid.copy()
            args[index] = args[index].to('cuda:1')
            with self.subTest(argument=index), self.assertRaisesRegex(RuntimeError, 'same device'):
                self.attention(*args)

    def test_no_context_sized_allocations_or_gather_kernels(self):
        for dtype in (torch.float32, torch.float16):
            for length in (8193, 16385):
                query, pages, values, table, _, _ = self.inputs(
                    length, 16, heads=(14, 2), dimension=64, dtype=dtype)
                self.attention(query, pages, values, table, length)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                initial = torch.cuda.memory_allocated()
                actual = self.attention(query, pages, values, table, length)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - initial
                self.assertLess(peak, length * query.element_size())
                self.assertEqual(actual.shape, query.shape)
        self.assert_single_kernel_in_clean_process()

    def assert_single_kernel_in_clean_process(self):
        # A broader sequence of earlier GPU tests can lose CUPTI activities.
        # Require real kernel evidence from a clean profiler process.
        script = '''
import torch
from micro_vllm.cuda_attention import decode_attention_paged
from micro_vllm.paged_kv_cache import PagedKVCache
cache = PagedKVCache.allocate(
    num_layers=1,num_kv_heads=2,num_pages=5,page_size=32,head_dim=64,
    dtype=torch.float16,device='cuda',page_table=[3,0,2])
q = torch.zeros(1,14,1,64,device='cuda',dtype=torch.float16)
k = torch.zeros(1,2,65,64,device='cuda',dtype=torch.float16)
cache.append(0,k,torch.ones_like(k))
decode_attention_paged(q,cache,layer_idx=0)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA]) as profile:
    actual = decode_attention_paged(q,cache,layer_idx=0)
    torch.cuda.synchronize()
kernels = [event.name for event in profile.events()
           if event.device_type == torch.autograd.DeviceType.CUDA]
assert len(kernels) == 1, kernels
assert 'decode_attention_online_kernel' in kernels[0], kernels
torch.testing.assert_close(actual,torch.ones_like(q))
print('PAGED_SINGLE_KERNEL_OK')
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PAGED_SINGLE_KERNEL_OK', result.stdout)


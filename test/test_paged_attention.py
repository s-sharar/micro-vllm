"""Public paged attention validates owned caches and reads their pages directly."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from micro_vllm import cuda_attention
from micro_vllm.kv_cache import ContiguousKVCache
from micro_vllm.paged_kv_cache import PagedKVCache


class PagedAttentionValidationTests(unittest.TestCase):
    def setUp(self):
        self.attention = getattr(cuda_attention, 'decode_attention_paged', None)
        self.assertIsNotNone(self.attention, 'provide the public direct paged decode operation')

    def cache(self, dtype=torch.float32):
        cache = PagedKVCache.allocate(
            num_layers=2, num_kv_heads=2, num_pages=7, page_size=4,
            head_dim=8, dtype=dtype, device='cpu', page_table=[4, 1, 5])
        key = torch.arange(80., dtype=dtype).reshape(1, 2, 5, 8)
        cache.append(0, key, -key)
        return cache

    def test_rejects_foreign_cache_types_before_loading_cuda(self):
        q = torch.zeros(1, 4, 1, 8)
        contiguous = ContiguousKVCache.allocate(
            num_layers=1, batch_size=1, num_kv_heads=2, capacity=4,
            head_dim=8, dtype=torch.float32, device='cpu')
        with patch.object(cuda_attention, '_load_extension') as load:
            for cache in (None, contiguous, object()):
                with self.assertRaisesRegex(ValueError, 'PagedKVCache'):
                    self.attention(q, cache, layer_idx=0)
            load.assert_not_called()

    def test_layer_and_empty_cache_rejection_preserve_lengths(self):
        cache = self.cache()
        q = torch.zeros(1, 4, 1, 8)
        with patch.object(cuda_attention, '_load_extension') as load:
            for layer in (-1, 2, True, 1.5, None):
                with self.assertRaisesRegex(ValueError, 'layer'):
                    self.attention(q, cache, layer_idx=layer)
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                self.attention(q, cache, layer_idx=1)
            load.assert_not_called()
        self.assertEqual((cache.length(0), cache.length(1)), (5, 0))

    def test_query_formats_fail_before_cuda_loading(self):
        cache = self.cache()
        q = torch.zeros(1, 4, 1, 8)
        cases = [(None, 'tensor'), (q.to_sparse(), 'strided'),
                 (q[0], 'shape'), (q.expand(2, -1, -1, -1), 'shape'),
                 (q.expand(-1, -1, 2, -1), 'shape'), (q[:, :0], 'shape'),
                 (q[..., :7], 'shape'), (q[:, :3], 'divisible'),
                 (q.half(), 'dtype'), (q.double(), 'dtype'),
                 (q.detach().requires_grad_(), 'gradients'),
                 (torch._neg_view(q), 'negative'),
                 (torch.zeros(1, 4, 1, 16)[..., ::2], 'feature stride')]
        with patch.object(cuda_attention, '_load_extension') as load:
            for invalid, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    self.attention(invalid, cache, layer_idx=0)
            load.assert_not_called()

    def test_cpu_cache_is_not_silently_gathered_or_fallback_executed(self):
        cache = self.cache()
        before = [p.clone() for p in cache._keys + cache._values]
        with patch.object(cache, 'gather', side_effect=AssertionError('must not gather')), \
                patch.object(cuda_attention, '_load_extension') as load:
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                self.attention(torch.zeros(1, 4, 1, 8), cache, layer_idx=0)
            load.assert_not_called()
        for actual, expected in zip(cache._keys + cache._values, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_bfloat16_decode_is_explicitly_unsupported(self):
        cache = self.cache(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'float16 or float32'):
            self.attention(torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16), cache, layer_idx=0)

    def test_import_has_no_native_compile_side_effect(self):
        code = ('from micro_vllm.cuda_attention import decode_attention_paged,_load_extension; '
                'assert callable(decode_attention_paged); '
                'assert _load_extension.cache_info().currsize == 0')
        run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             env=os.environ.copy(), timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class PagedAttentionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.attention = getattr(cuda_attention, 'decode_attention_paged', None)
        self.assertIsNotNone(self.attention, 'provide the public direct paged decode operation')

    def cache(self, dtype, length=33, page_size=4):
        needed = (length + page_size - 1) // page_size
        mapping = list(reversed(range(1, needed + 1)))
        return PagedKVCache.allocate(
            num_layers=2, num_kv_heads=2, num_pages=needed + 2, page_size=page_size,
            head_dim=64, dtype=dtype, device='cuda', page_table=mapping)

    def test_direct_reads_match_contiguous_for_each_layer_without_gather(self):
        generator = torch.Generator(device='cuda').manual_seed(471)
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                cache = self.cache(dtype)
                for pages in cache._keys + cache._values:
                    pages.fill_(float('nan'))
                q = torch.randn(1, 14, 1, 64, device='cuda', dtype=dtype, generator=generator)
                q_saved = q.clone()
                sources, references = [], []
                for layer, length in ((0, 33), (1, 5)):
                    k = torch.randn(1, 2, length, 64, device='cuda', dtype=dtype, generator=generator)
                    v = torch.randn(k.shape, device='cuda', dtype=dtype, generator=generator)
                    cache.append(layer, k[:, :, :3], v[:, :, :3])
                    cache.append(layer, k[:, :, 3:], v[:, :, 3:])
                    sources.append((k, v))
                    references.append(cuda_attention.decode_attention_gqa(q, k, v))
                snapshots = [p.clone() for p in cache._keys + cache._values]
                pointers = [p.data_ptr() for p in cache._keys + cache._values]
                with patch.object(cache, 'gather', side_effect=AssertionError('must not gather')), \
                        patch.object(cache, '_gather_buffer', side_effect=AssertionError('must not gather')):
                    outputs = [self.attention(q, cache, layer_idx=layer) for layer in (0, 1)]
                tolerance = dict(rtol=2e-3, atol=2e-3) if dtype == torch.float16 else dict(rtol=2e-4, atol=2e-5)
                for actual, expected in zip(outputs, references):
                    torch.testing.assert_close(actual, expected, **tolerance)
                    self.assertEqual(actual.shape, q.shape)
                    self.assertTrue(actual.is_contiguous())
                    self.assertNotEqual(actual.data_ptr(), q.data_ptr())
                self.assertNotEqual(outputs[0].data_ptr(), outputs[1].data_ptr())
                torch.testing.assert_close(q, q_saved, rtol=0, atol=0)
                for actual, expected in zip(cache._keys + cache._values, snapshots):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                self.assertEqual([p.data_ptr() for p in cache._keys + cache._values], pointers)
                self.assertEqual((cache.length(0), cache.length(1)), (33, 5))

    def test_reset_and_shorter_reuse_exclude_stale_values(self):
        cache = self.cache(torch.float32)
        q = torch.zeros(1, 14, 1, 64, device='cuda')
        cache.append(0, torch.ones(1, 2, 33, 64, device='cuda'),
                     torch.full((1, 2, 33, 64), 99., device='cuda'))
        self.attention(q, cache, layer_idx=0)
        cache.reset()
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.attention(q, cache, layer_idx=0)
        cache.append(0, torch.zeros(1, 2, 3, 64, device='cuda'),
                     torch.full((1, 2, 3, 64), 7., device='cuda'))
        actual = self.attention(q, cache, layer_idx=0)
        torch.testing.assert_close(actual, torch.full_like(q, 7.), rtol=1e-6, atol=1e-6)

    def test_workspace_is_output_sized_not_cache_sized(self):
        cache = self.cache(torch.float16, length=1025, page_size=16)
        k = torch.zeros(1, 2, 1025, 64, device='cuda', dtype=torch.float16)
        v = torch.ones_like(k)
        cache.append(0, k, v)
        q = torch.zeros(1, 14, 1, 64, device='cuda', dtype=torch.float16)
        self.attention(q, cache, layer_idx=0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        actual = self.attention(q, cache, layer_idx=0)
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - before
        self.assertLess(extra, k.numel() * k.element_size())
        torch.testing.assert_close(actual, torch.ones_like(q), rtol=2e-3, atol=2e-3)

    def test_mismatched_query_device_is_rejected(self):
        cache = self.cache(torch.float32)
        cache.append(0, torch.zeros(1, 2, 1, 64, device='cuda'),
                     torch.ones(1, 2, 1, 64, device='cuda'))
        with self.assertRaisesRegex(ValueError, 'device'):
            self.attention(torch.zeros(1, 14, 1, 64), cache, layer_idx=0)


if __name__ == '__main__':
    unittest.main()

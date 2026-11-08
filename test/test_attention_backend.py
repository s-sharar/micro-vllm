"""Backend policy, blocked masks, and real accelerated attention operators."""
import os
import unittest
from unittest.mock import patch

import torch

from micro_vllm import attention_backend as backend
from micro_vllm.attention import causal_attention
from micro_vllm import cuda_attention

GPU_TESTS = os.environ.get('RUN_GPU_TESTS') == '1' and torch.cuda.is_available()


class AttentionPolicyTests(unittest.TestCase):
    def select(self, mode='auto', **changes):
        options = dict(device=torch.device('cuda'), dtype=torch.float16,
                       batch_size=1, query_length=4, head_dim=64,
                       past_length=0, has_cache=False)
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.get_device_capability', return_value=(8, 6)), \
                patch('torch.backends.cuda.is_flash_attention_available', return_value=True):
            return backend.select_attention_backend(mode, **{**options, **changes})

    def test_public_backend_names_are_strictly_validated(self):
        for name in ('auto', 'torch', 'cuda'):
            self.assertEqual(backend.validate_attention_backend(name), name)
        for invalid in ('flash', 'CUDA', '', None, True, 3, []):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                backend.validate_attention_backend(invalid)

    def test_cpu_auto_and_torch_always_use_reference(self):
        for mode in ('auto', 'torch'):
            self.assertEqual(self.select(mode, device=torch.device('cpu')), 'torch')
        for changes in ({}, dict(query_length=1, past_length=3, has_cache=True),
                        dict(dtype=torch.bfloat16), dict(batch_size=2)):
            self.assertEqual(self.select('torch', **changes), 'torch')

    def test_native_decode_requires_nonempty_cache_and_single_new_token(self):
        for mode in ('auto', 'cuda'):
            for dtype in (torch.float16, torch.float32):
                self.assertEqual(self.select(mode, dtype=dtype, query_length=1,
                                             past_length=3, has_cache=True), 'cuda')
        for changes in (dict(query_length=4, past_length=0, has_cache=True),
                        dict(query_length=3, past_length=4, has_cache=True),
                        dict(query_length=1, past_length=0, has_cache=True),
                        dict(query_length=1, past_length=0, has_cache=False)):
            self.assertEqual(self.select('cuda', **changes), 'torch')

    def test_forced_cuda_rejects_unsupported_device_batch_and_dtype(self):
        for changes in (dict(device=torch.device('cpu')), dict(batch_size=2),
                        dict(dtype=torch.bfloat16), dict(dtype=torch.float64)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.select('cuda', **changes)

    def test_auto_prefill_and_cached_chunks_use_flash_when_eligible(self):
        for changes in ({}, dict(past_length=4, has_cache=True),
                        dict(query_length=1), dict(head_dim=256), dict(head_dim=72)):
            self.assertEqual(self.select(**changes), 'flash')

    def test_auto_reference_fallbacks_are_deterministic(self):
        for changes in (dict(dtype=torch.bfloat16), dict(dtype=torch.float32),
                        dict(batch_size=2), dict(head_dim=258), dict(head_dim=66)):
            with self.subTest(changes=changes):
                self.assertEqual(self.select(**changes), 'torch')
        options = dict(device=torch.device('cuda'), dtype=torch.float16, batch_size=1,
                       query_length=4, head_dim=64, past_length=0, has_cache=False)
        with patch('torch.cuda.get_device_capability', return_value=(7, 5)):
            self.assertEqual(backend.select_attention_backend('auto', **options), 'torch')
        with patch('torch.backends.cuda.is_flash_attention_available', return_value=False):
            self.assertEqual(backend.select_attention_backend('auto', **options), 'torch')


class ReferenceDispatchTests(unittest.TestCase):
    def test_explicit_true_blocked_mask_is_respected_for_every_implementation(self):
        query = torch.zeros(1, 4, 2, 8)
        key = torch.zeros(1, 2, 4, 8)
        value = torch.arange(4.).reshape(1, 1, 4, 1).expand(1, 2, 4, 8)
        mask = torch.tensor([[False, True, True, True], [False, False, True, True]])
        expected = causal_attention(query, key, value, past_length=2, attention_mask=mask)
        self.assertTrue(torch.equal(expected[:, :, 0], torch.zeros(1, 4, 8)))
        self.assertTrue(torch.equal(expected[:, :, 1], torch.full((1, 4, 8), .5)))
        with patch('micro_vllm.cuda_attention.decode_attention_gqa') as native, \
                patch('torch.nn.functional.scaled_dot_product_attention') as flash:
            for implementation in ('torch', 'cuda', 'flash'):
                actual = backend.attention_forward(query, key, value, past_length=2,
                                                   attention_mask=mask,
                                                   implementation=implementation)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        native.assert_not_called()
        flash.assert_not_called()

    def test_unknown_internal_implementation_rejected(self):
        query = torch.zeros(1, 2, 1, 8)
        for invalid in ('auto', 'bogus', None):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                backend.attention_forward(query, query, query, implementation=invalid)

    def test_reference_preserves_cached_lower_right_causality(self):
        generator = torch.Generator().manual_seed(42)
        query = torch.randn(2, 14, 3, 16, generator=generator)
        key = torch.randn(2, 2, 8, 16, generator=generator)
        value = torch.randn(key.shape, generator=generator)
        expected = causal_attention(query, key, value, past_length=5)
        actual = backend.attention_forward(query, key, value, past_length=5,
                                           implementation='torch')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(GPU_TESTS, 'set RUN_GPU_TESTS=1 for real CUDA operators')
class AcceleratedAttentionTests(unittest.TestCase):
    def tensors(self, tokens, past, dtype):
        generator = torch.Generator(device='cuda').manual_seed(932)
        query = torch.randn((1, 14, tokens, 64), dtype=dtype, device='cuda', generator=generator)
        capacity = past + tokens + 19
        key_storage = torch.randn((1, 2, capacity, 64), dtype=dtype,
                                  device='cuda', generator=generator)
        value_storage = torch.randn(key_storage.shape, dtype=dtype,
                                    device='cuda', generator=generator)
        return query, key_storage[:, :, :past + tokens], value_storage[:, :, :past + tokens]

    def require_flash(self):
        if (not torch.backends.cuda.is_flash_attention_available()
                or torch.cuda.get_device_capability()[0] < 8):
            self.skipTest('FlashAttention requires an enabled build and SM80+')

    def test_native_decode_calls_actual_gqa_with_strided_cache_views(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                query, key, value = self.tensors(1, 128, dtype)
                expected = causal_attention(query, key, value, past_length=128)
                with patch('micro_vllm.cuda_attention.decode_attention_gqa',
                           wraps=cuda_attention.decode_attention_gqa) as native:
                    actual = backend.attention_forward(query, key, value, past_length=128,
                                                       implementation='cuda')
                native.assert_called_once()
                self.assertIs(native.call_args.args[1], key)
                self.assertEqual(key.stride(1), (129 + 19) * 64)
                tolerance = dict(rtol=2e-3, atol=2e-3) if dtype == torch.float16 else dict(rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(actual, expected, **tolerance)

    def test_flash_full_and_rectangular_prefills_use_real_fused_operator(self):
        self.require_flash()
        for tokens, past in ((17, 0), (5, 12), (1, 0)):
            with self.subTest(tokens=tokens, past=past):
                query, key, value = self.tensors(tokens, past, torch.float16)
                expected = causal_attention(query, key, value, past_length=past)
                with patch('micro_vllm.attention_backend.flash_attention',
                           wraps=backend.flash_attention) as flash:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                        actual = backend.attention_forward(query, key, value, past_length=past,
                                                           implementation='flash')
                flash.assert_called_once()
                names = {event.key for event in profile.key_averages()}
                self.assertTrue(any('_scaled_dot_product_flash_attention' in name for name in names), names)
                self.assertFalse(any('_scaled_dot_product_attention_math' in name for name in names), names)
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-3)


if __name__ == '__main__':
    unittest.main()

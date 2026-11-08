"""Single-head decode attention checked against an independent FP64 oracle."""
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import torch


def reference_attention(query, key, value):
    scores = (key.double() @ query.double()) * (query.numel() ** -0.5)
    return (torch.softmax(scores, dim=0) @ value.double()).float()


class AttentionImportTests(unittest.TestCase):
    def test_import_requires_no_toolkit_or_exercise_extension(self):
        self.assertIsNotNone(importlib.util.find_spec('micro_vllm.cuda_attention'))
        script = (
            'import sys; import micro_vllm.cuda_attention; '
            'assert "_micro_vllm_decode_attention" not in sys.modules; '
            'assert "micro_vllm.cuda_ops" not in sys.modules; '
            'print("ATTENTION_IMPORT_OK")'
        )
        with tempfile.TemporaryDirectory(prefix='micro-vllm-attention-import-') as directory:
            cache = os.path.join(directory, 'cache')
            env = dict(os.environ, CUDA_HOME=os.path.join(directory, 'no-toolkit'),
                       TORCH_EXTENSIONS_DIR=cache)
            result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                    text=True, cwd=directory, env=env, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('ATTENTION_IMPORT_OK', result.stdout)
            self.assertFalse(os.path.exists(cache))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class SingleHeadAttentionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('micro_vllm.cuda_attention'),
                             'provide the single-head CUDA decode-attention module')
        self.attention = importlib.import_module('micro_vllm.cuda_attention').decode_attention

    def test_zero_query_averages_all_valid_values(self):
        query = torch.zeros(64, device='cuda')
        key = torch.randn(5, 64, device='cuda')
        value = torch.arange(320, device='cuda', dtype=torch.float32).reshape(5, 64)
        actual = self.attention(query, key, value)
        self.assertEqual(actual.shape, query.shape)
        self.assertEqual(actual.device, query.device)
        self.assertEqual(actual.dtype, query.dtype)
        torch.testing.assert_close(actual, value.mean(0), rtol=0, atol=2e-5)

    def test_one_visible_token_returns_its_value(self):
        query = torch.randn(64, device='cuda')
        key, value = torch.randn(1, 64, device='cuda'), torch.randn(1, 64, device='cuda')
        torch.testing.assert_close(self.attention(query, key, value), value[0], rtol=0, atol=0)

    def test_nonuniform_weights_use_head_dimension_scale(self):
        query = torch.tensor([2., 0., 0., 0.], device='cuda')
        key = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.]], device='cuda')
        value = torch.tensor([[1., 2., 3., 4.], [10., 20., 30., 40.]], device='cuda')
        weight = torch.sigmoid(torch.tensor(1., device='cuda'))
        expected = value[0] * (1 - weight) + value[1] * weight
        torch.testing.assert_close(self.attention(query, key, value), expected, rtol=1e-5, atol=1e-6)

    def test_matches_fp64_oracle_across_context_boundaries(self):
        generator = torch.Generator(device='cuda').manual_seed(71)
        for length in (1, 2, 3, 31, 32, 33, 255, 256, 257, 513, 1025):
            with self.subTest(length=length):
                query = torch.randn(64, device='cuda', generator=generator)
                key = torch.randn(length, 64, device='cuda', generator=generator)
                value = torch.randn(length, 64, device='cuda', generator=generator)
                expected = reference_attention(query, key, value)
                torch.testing.assert_close(self.attention(query, key, value), expected,
                                           rtol=1e-4, atol=1e-5)

    def test_head_dimensions_and_feature_tails(self):
        generator = torch.Generator(device='cuda').manual_seed(73)
        for dimension in (1, 2, 31, 33, 63, 65, 128, 257):
            with self.subTest(dimension=dimension):
                query = torch.randn(dimension, device='cuda', generator=generator)
                key = torch.randn(7, dimension, device='cuda', generator=generator)
                value = torch.randn(7, dimension, device='cuda', generator=generator)
                torch.testing.assert_close(self.attention(query, key, value),
                                           reference_attention(query, key, value),
                                           rtol=1e-4, atol=1e-5)

    def test_large_finite_scores_use_stable_softmax(self):
        query = torch.tensor([2., 0., 0., 0.], device='cuda')
        key = torch.tensor([[1000., 0., 0., 0.], [1001., 0., 0., 0.]], device='cuda')
        value = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]], device='cuda')
        actual = self.attention(query, key, value)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, reference_attention(query, key, value), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(self.attention(query, -key, value),
                                   reference_attention(query, -key, value), rtol=1e-5, atol=1e-6)

    def test_finite_large_values_do_not_overflow_before_normalization(self):
        query = torch.zeros(1, device='cuda')
        key = torch.zeros(2, 1, device='cuda')
        for sign in (1., -1.):
            with self.subTest(sign=sign):
                value = torch.full((2, 1), sign * 2e38, device='cuda')
                actual = self.attention(query, key, value)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, reference_attention(query, key, value),
                                           rtol=1e-6, atol=0)

    def test_offsets_inputs_unchanged_and_output_storage_independent(self):
        query = (torch.arange(70, device='cuda', dtype=torch.float32) / 64)[3:67]
        key = (torch.arange(704, device='cuda', dtype=torch.float32) / 1000).reshape(11, 64)[2:9]
        value = torch.arange(832, device='cuda', dtype=torch.float32).reshape(13, 64)[3:10]
        original = [tensor.clone() for tensor in (query, key, value)]
        first = self.attention(query, key, value)
        expected = reference_attention(query, key, value)
        torch.testing.assert_close(first, expected, rtol=1e-4, atol=1e-5)
        before = first.clone()
        second = self.attention(query, key, value + 1)
        torch.testing.assert_close(second, expected + 1, rtol=1e-4, atol=1e-5)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        for tensor, saved in zip((query, key, value), original):
            torch.testing.assert_close(tensor, saved, rtol=0, atol=0)
            self.assertNotEqual(first.data_ptr(), tensor.data_ptr())
        torch.testing.assert_close(first, before, rtol=0, atol=0)

    def test_last_visible_token_and_value_cancellation(self):
        query = torch.zeros(33, device='cuda')
        key, value = torch.zeros(3, 33, device='cuda'), torch.zeros(3, 33, device='cuda')
        value[-1].fill_(12)
        torch.testing.assert_close(self.attention(query, key, value),
                                   torch.full_like(query, 4), rtol=0, atol=0)
        base = torch.arange(99, device='cuda', dtype=torch.float32).reshape(3, 33) / 100
        value = torch.cat((base, -base, torch.zeros(1, 33, device='cuda')))
        key = torch.zeros_like(value)
        torch.testing.assert_close(self.attention(query, key, value),
                                   reference_attention(query, key, value), rtol=0, atol=1e-6)

    def test_current_stream_orders_cache_writes_and_attention(self):
        query = torch.zeros(64, device='cuda')
        key, value = torch.zeros(5, 64, device='cuda'), torch.zeros(5, 64, device='cuda')
        self.attention(query, key, value)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10000000)
            query.fill_(1)
            key.fill_(1)
            value.fill_(3)
            actual = self.attention(query, key, value) * 2
        stream.synchronize()
        torch.testing.assert_close(actual, torch.full_like(query, 6), rtol=0, atol=1e-6)

    def test_rejects_unsupported_formats_in_each_argument(self):
        valid = [torch.ones(64, device='cuda'), torch.ones(3, 64, device='cuda'),
                 torch.ones(3, 64, device='cuda')]
        for index in range(3):
            tensor = valid[index]
            noncontiguous = (torch.ones(128, device='cuda')[::2] if index == 0 else
                             torch.ones(3, 128, device='cuda')[:, ::2])
            cases = [(tensor.cpu(), 'CUDA'), (tensor.half(), 'float32'),
                     (tensor.double(), 'float32'), (tensor.long(), 'float32'),
                     (tensor.to_sparse(), 'strided'), (noncontiguous, 'contiguous'),
                     (torch._neg_view(tensor), 'negative views'),
                     (tensor.detach().requires_grad_(), 'gradients'),
                     (tensor.unsqueeze(0), 'shape')]
            for invalid, message in cases:
                args = valid.copy()
                args[index] = invalid
                with self.subTest(argument=index, message=message), self.assertRaisesRegex(RuntimeError, message):
                    self.attention(*args)

    def test_rejects_empty_context_and_mismatched_dimensions(self):
        query = torch.ones(64, device='cuda')
        key = torch.ones(3, 64, device='cuda')
        cases = [
            (query, key[:0], key[:0], 'positive'),
            (query[:0], key[:, :0], key[:, :0], 'positive'),
            (query[:63], key, key, 'head dimension'),
            (query, key, key[:2], 'same shape'),
            (query, key[:, :63].contiguous(), key, 'same shape'),
        ]
        for q, k, v, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.attention(q, k, v)

    def test_public_api_in_an_independent_process(self):
        script = (
            'import sys,torch; from micro_vllm.cuda_attention import decode_attention; '
            'q=torch.zeros(64,device="cuda"); '
            'k=torch.ones(3,64,device="cuda"); v=torch.ones_like(k)*2; '
            'torch.testing.assert_close(decode_attention(q,k,v), torch.ones_like(q)*2); '
            'assert "micro_vllm.cuda_ops" not in sys.modules; '
            'assert "_micro_vllm_cuda" not in sys.modules; '
            'print("DECODE_ATTENTION_OK")'
        )
        with tempfile.TemporaryDirectory(prefix='micro-vllm-attention-') as directory:
            result = subprocess.run([sys.executable, '-c', script], cwd=directory,
                                    env=os.environ.copy(), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('DECODE_ATTENTION_OK', result.stdout)

    @unittest.skipUnless(torch.cuda.device_count() > 1, 'two CUDA GPUs required')
    def test_device_guard_and_mismatched_devices(self):
        with torch.cuda.device(0):
            query = torch.ones(64, device='cuda:1')
            key = torch.ones(3, 64, device='cuda:1')
            torch.testing.assert_close(self.attention(query, key, key), key[0], rtol=0, atol=0)
            self.assertEqual(torch.cuda.current_device(), 0)
            for index in (0, 1, 2):
                args = [query, key, key]
                args[index] = args[index].to('cuda:0')
                with self.subTest(argument=index), self.assertRaisesRegex(RuntimeError, 'same device'):
                    self.attention(*args)


if __name__ == '__main__':
    unittest.main()

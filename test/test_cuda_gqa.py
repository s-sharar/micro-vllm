"""FP32 grouped-query decode: head ownership, online state, and PyTorch parity."""
import importlib
import os
import subprocess
import sys
import tempfile
import unittest

import torch

from micro_vllm.attention import causal_attention


def fp64_reference(query, key, value):
    query_heads, kv_heads = query.shape[1], key.shape[1]
    group = query_heads // kv_heads
    output = torch.empty_like(query)
    for head in range(query_heads):
        kv_head = head // group
        scores = key[0, kv_head].double() @ query[0, head, 0].double()
        probabilities = torch.softmax(scores * query.shape[-1] ** -0.5, dim=0)
        output[0, head, 0] = (probabilities @ value[0, kv_head].double()).float()
    return output


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class GroupedQueryAttentionTests(unittest.TestCase):
    def setUp(self):
        self.ops = importlib.import_module('micro_vllm.cuda_attention')
        self.assertTrue(hasattr(self.ops, 'decode_attention_gqa'),
                        'provide the FP32 multi-head CUDA decode operation')
        self.attention = self.ops.decode_attention_gqa

    def test_seven_query_heads_share_each_original_kv_head(self):
        query = torch.zeros(1, 14, 1, 64, device='cuda')
        key = torch.randn(1, 2, 5, 64, device='cuda')
        features = torch.arange(64, device='cuda', dtype=torch.float32)
        value = torch.empty_like(key)
        value[0, 0] = features
        value[0, 1] = features + 1000
        expected = torch.empty_like(query)
        expected[0, :7, 0] = features
        expected[0, 7:, 0] = features + 1000
        actual = self.attention(query, key, value)
        self.assertEqual(actual.shape, query.shape)
        self.assertEqual(actual.dtype, query.dtype)
        self.assertEqual(actual.device, query.device)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)

    def test_queries_in_the_same_group_can_choose_different_positions(self):
        query = torch.tensor([[[[1., 0.]], [[-1., 0.]], [[0., 1.]], [[0., -1.]]]], device='cuda')
        key = torch.tensor([[[[3., 0.], [-3., 0.]], [[0., 3.], [0., -3.]]]], device='cuda')
        value = torch.tensor([[[[10., 0.], [0., 20.]], [[100., 0.], [0., 200.]]]], device='cuda')
        actual = self.attention(query, key, value)
        torch.testing.assert_close(actual, fp64_reference(query, key, value), rtol=2e-4, atol=2e-5)
        self.assertGreater(actual[0, 0, 0, 0].item(), actual[0, 1, 0, 0].item())
        self.assertGreater(actual[0, 2, 0, 0].item(), actual[0, 3, 0, 0].item())

    def test_qwen_heads_match_pytorch_and_fp64_across_chunk_boundaries(self):
        generator = torch.Generator(device='cuda').manual_seed(89)
        for length in (1, 31, 32, 33, 65, 257, 1025):
            with self.subTest(length=length):
                query = torch.randn(1, 14, 1, 64, device='cuda', generator=generator)
                key = torch.randn(1, 2, length, 64, device='cuda', generator=generator)
                value = torch.randn(key.shape, device='cuda', generator=generator)
                actual = self.attention(query, key, value)
                torch.testing.assert_close(actual, fp64_reference(query, key, value),
                                           rtol=2e-4, atol=2e-5)
                expected = causal_attention(query, key, value, past_length=length - 1)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_each_head_matches_the_existing_single_head_kernel(self):
        generator = torch.Generator(device='cuda').manual_seed(97)
        query = torch.randn(1, 14, 1, 64, device='cuda', generator=generator)
        key = torch.randn(1, 2, 65, 64, device='cuda', generator=generator)
        value = torch.randn(key.shape, device='cuda', generator=generator)
        actual = self.attention(query, key, value)
        for head in range(14):
            expected = self.ops.decode_attention_online(query[0, head, 0], key[0, head // 7], value[0, head // 7])
            torch.testing.assert_close(actual[0, head, 0], expected, rtol=0, atol=0)

    def test_other_head_group_sizes_and_feature_tails(self):
        generator = torch.Generator(device='cuda').manual_seed(101)
        for query_heads, kv_heads, dimension in ((1, 1, 33), (4, 4, 64), (6, 2, 33),
                                                 (14, 1, 64), (6, 2, 257)):
            with self.subTest(heads=(query_heads, kv_heads), dimension=dimension):
                query = torch.randn(1, query_heads, 1, dimension, device='cuda', generator=generator)
                key = torch.randn(1, kv_heads, 33, dimension, device='cuda', generator=generator)
                value = torch.randn(key.shape, device='cuda', generator=generator)
                torch.testing.assert_close(self.attention(query, key, value), fp64_reference(query, key, value),
                                           rtol=2e-4, atol=2e-5)

    def test_offset_views_preserve_inputs_and_outputs_are_independent(self):
        generator = torch.Generator(device='cuda').manual_seed(103)
        query = torch.randn(1, 16, 1, 64, device='cuda', generator=generator)[:, 1:15]
        key = torch.randn(1, 4, 33, 64, device='cuda', generator=generator)[:, 1:3]
        value = torch.randn(1, 4, 33, 64, device='cuda', generator=generator)[:, 1:3]
        self.assertTrue(all(tensor.is_contiguous() for tensor in (query, key, value)))
        originals = [tensor.clone() for tensor in (query, key, value)]
        first = self.attention(query, key, value)
        before = first.clone()
        second = self.attention(query, key, value + 1)
        torch.testing.assert_close(first, fp64_reference(query, key, value), rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(second, first + 1, rtol=2e-4, atol=2e-5)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        for tensor, saved in zip((query, key, value), originals):
            torch.testing.assert_close(tensor, saved, rtol=0, atol=0)
            self.assertNotEqual(first.data_ptr(), tensor.data_ptr())
        torch.testing.assert_close(first, before, rtol=0, atol=0)

    def test_large_values_and_independent_state_per_block(self):
        query = torch.zeros(1, 14, 1, 64, device='cuda')
        key = torch.zeros(1, 2, 65, 64, device='cuda')
        value = torch.empty_like(key)
        value[:, 0].fill_(2e38)
        value[:, 1].fill_(-2e38)
        actual = self.attention(query, key, value)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, fp64_reference(query, key, value), rtol=5e-6, atol=0)

    def test_current_stream_orders_all_head_inputs_and_consumers(self):
        query = torch.zeros(1, 14, 1, 64, device='cuda')
        key = torch.zeros(1, 2, 65, 64, device='cuda')
        value = torch.zeros_like(key)
        self.attention(query, key, value)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10000000)
            query.fill_(1)
            key.fill_(1)
            value[:, 0].fill_(3)
            value[:, 1].fill_(7)
            actual = self.attention(query, key, value) * 2
        stream.synchronize()
        expected = torch.empty_like(query)
        expected[:, :7].fill_(6)
        expected[:, 7:].fill_(14)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_rejects_unsupported_formats_in_each_argument(self):
        valid = [torch.ones(1, 14, 1, 64, device='cuda'), torch.ones(1, 2, 5, 64, device='cuda'),
                 torch.ones(1, 2, 5, 64, device='cuda')]
        for index, tensor in enumerate(valid):
            shape = list(tensor.shape)
            shape[-1] *= 2
            noncontiguous = torch.ones(shape, device='cuda')[..., ::2]
            cases = [(tensor.cpu(), 'CUDA'),
                     (tensor.double(), 'float32'), (tensor.long(), 'float32'),
                     (tensor.to_sparse(), 'strided'), (noncontiguous, 'feature stride'),
                     (torch._neg_view(tensor), 'negative views'),
                     (tensor.detach().requires_grad_(), 'gradients'), (tensor.squeeze(0), 'shape')]
            for invalid, message in cases:
                args = valid.copy()
                args[index] = invalid
                with self.subTest(argument=index, message=message), self.assertRaisesRegex(RuntimeError, message):
                    self.attention(*args)

    def test_rejects_invalid_batch_query_length_and_head_dimensions(self):
        query = torch.ones(1, 14, 1, 64, device='cuda')
        key = torch.ones(1, 2, 5, 64, device='cuda')
        cases = [
            (query.repeat(2, 1, 1, 1), key.repeat(2, 1, 1, 1), key.repeat(2, 1, 1, 1), 'batch'),
            (query, key.repeat(2, 1, 1, 1), key, 'batch'),
            (query.repeat(1, 1, 2, 1), key, key, 'one query'),
            (query[:, :0], key, key, 'positive'),
            (query, key[:, :0], key[:, :0], 'positive'),
            (query, key[:, :, :0], key[:, :, :0], 'positive'),
            (query[..., :0], key[..., :0], key[..., :0], 'positive'),
            (query[:, :3].contiguous(), key, key, 'divisible'),
            (query[:, :1].contiguous(), key, key, 'divisible'),
            (query, key, key[:, :1], 'same shape'),
            (query[..., :63].contiguous(), key, key, 'head dimension'),
        ]
        for q, k, v, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.attention(q, k, v)

    def test_does_not_allocate_repeated_kv_storage(self):
        query = torch.zeros(1, 14, 1, 64, device='cuda')
        key = torch.zeros(1, 2, 257, 64, device='cuda')
        value = torch.ones_like(key)
        self.attention(query, key, value)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial = torch.cuda.memory_allocated()
        actual = self.attention(query, key, value)
        torch.cuda.synchronize()
        extra_peak = torch.cuda.max_memory_allocated() - initial
        self.assertLess(extra_peak, key.numel() * key.element_size())
        torch.testing.assert_close(actual, torch.ones_like(query), rtol=2e-5, atol=2e-5)

    def test_public_api_in_an_independent_process(self):
        script = (
            'import sys,torch; from micro_vllm.cuda_attention import decode_attention_gqa; '
            'q=torch.zeros(1,14,1,64,device="cuda"); k=torch.zeros(1,2,33,64,device="cuda"); '
            'v=torch.ones_like(k)*3; '
            'torch.testing.assert_close(decode_attention_gqa(q,k,v), torch.ones_like(q)*3); '
            'assert "micro_vllm.cuda_ops" not in sys.modules; '
            'print("GQA_API_OK")'
        )
        with tempfile.TemporaryDirectory(prefix='micro-vllm-gqa-') as directory:
            result = subprocess.run([sys.executable, '-c', script], cwd=directory,
                                    env=os.environ.copy(), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('GQA_API_OK', result.stdout)

    @unittest.skipUnless(torch.cuda.device_count() > 1, 'two CUDA GPUs required')
    def test_device_guard_and_mismatched_devices(self):
        with torch.cuda.device(0):
            query = torch.ones(1, 14, 1, 64, device='cuda:1')
            key = torch.ones(1, 2, 5, 64, device='cuda:1')
            torch.testing.assert_close(self.attention(query, key, key), torch.ones_like(query), rtol=1e-6, atol=1e-6)
            self.assertEqual(torch.cuda.current_device(), 0)
            for index in (0, 1, 2):
                args = [query, key, key]
                args[index] = args[index].to('cuda:0')
                with self.subTest(argument=index), self.assertRaisesRegex(RuntimeError, 'same device'):
                    self.attention(*args)


if __name__ == '__main__':
    unittest.main()

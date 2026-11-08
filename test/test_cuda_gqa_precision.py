"""FP16 grouped-query decode keeps attention arithmetic in FP32 until output."""
import importlib
import unittest

import torch

from micro_vllm.attention import causal_attention


def fp64_reference(query, key, value):
    """Compute from already-quantized inputs; round only the final output."""
    query_heads, kv_heads = query.shape[1], key.shape[1]
    group = query_heads // kv_heads
    output = torch.empty_like(query, dtype=torch.float64)
    for head in range(query_heads):
        kv_head = head // group
        scores = key[0, kv_head].double() @ query[0, head, 0].double()
        probabilities = torch.softmax(scores * query.shape[-1] ** -0.5, dim=0)
        output[0, head, 0] = probabilities @ value[0, kv_head].double()
    return output.to(query.dtype)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class GroupedQueryAttentionPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.attention = importlib.import_module(
            'micro_vllm.cuda_attention').decode_attention_gqa

    def inputs(self, length=65, heads=(14, 2), dimension=64, seed=271):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        query = torch.randn(1, heads[0], 1, dimension, device='cuda',
                            dtype=torch.float16, generator=generator)
        key = torch.randn(1, heads[1], length, dimension, device='cuda',
                          dtype=torch.float16, generator=generator)
        value = torch.randn(key.shape, device='cuda', dtype=torch.float16,
                            generator=generator)
        return query, key, value

    def test_half_qwen_matches_quantized_input_fp64_reference(self):
        query, key, value = self.inputs()
        actual = self.attention(query, key, value)
        self.assertEqual(actual.shape, query.shape)
        self.assertEqual(actual.dtype, torch.float16)
        self.assertEqual(actual.device, query.device)
        torch.testing.assert_close(actual, fp64_reference(query, key, value),
                                   rtol=1e-3, atol=2e-4)

    def test_half_tile_tails_match_fp64_and_existing_attention(self):
        for length in (1, 31, 32, 33, 65, 257, 1025):
            with self.subTest(length=length):
                query, key, value = self.inputs(length=length, seed=277 + length)
                actual = self.attention(query, key, value)
                torch.testing.assert_close(actual, fp64_reference(query, key, value),
                                           rtol=1e-3, atol=2e-4)
                # PyTorch rounds softmax probabilities to half before the V matmul.
                expected = causal_attention(query, key, value, past_length=length - 1)
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=1e-3)

    def test_half_general_head_ratios_and_feature_tails(self):
        for heads, dimension in (((1, 1), 33), ((4, 4), 64), ((6, 2), 33),
                                 ((14, 1), 64), ((6, 2), 257)):
            with self.subTest(heads=heads, dimension=dimension):
                query, key, value = self.inputs(length=33, heads=heads,
                                                dimension=dimension, seed=281)
                actual = self.attention(query, key, value)
                torch.testing.assert_close(actual, fp64_reference(query, key, value),
                                           rtol=1e-3, atol=2e-4)

    def test_half_online_state_does_not_round_at_each_tile(self):
        # At tile two the mean is exactly a half-way value. Rounding the
        # normalized running output to half there loses the later small updates.
        query = torch.zeros(1, 14, 1, 64, device='cuda', dtype=torch.float16)
        key = torch.zeros(1, 2, 4097, 64, device='cuda', dtype=torch.float16)
        value = torch.full_like(key, 1 + 2 ** -10)
        value[:, :, :32] = 1
        expected = fp64_reference(query, key, value)
        self.assertTrue(torch.equal(expected, torch.full_like(query, 1 + 2 ** -10)))
        actual = self.attention(query, key, value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_half_large_finite_values_keep_fp32_accumulators(self):
        query = torch.zeros(1, 14, 1, 64, device='cuda', dtype=torch.float16)
        key = torch.zeros(1, 2, 1025, 64, device='cuda', dtype=torch.float16)
        value = torch.empty_like(key)
        value[:, 0].fill_(60000)
        value[:, 1].fill_(-60000)
        actual = self.attention(query, key, value)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, fp64_reference(query, key, value),
                                   rtol=0, atol=0)

    def test_half_large_finite_logits_use_stable_fp32_softmax(self):
        query = torch.full((1, 14, 1, 64), 500, device='cuda', dtype=torch.float16)
        query[:, 1::2].neg_()
        key = torch.full((1, 2, 65, 64), 500, device='cuda', dtype=torch.float16)
        key[:, :, 0] = 501
        key[:, :, -1] = 499
        value = self.inputs(length=65, seed=283)[2]
        actual = self.attention(query, key, value)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, fp64_reference(query, key, value),
                                   rtol=0, atol=0)

    def test_half_inputs_preserved_and_output_storage_is_fresh(self):
        query, key, value = self.inputs(seed=293)
        originals = [tensor.clone() for tensor in (query, key, value)]
        first = self.attention(query, key, value)
        saved = first.clone()
        second = self.attention(query, key, value + 1)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(second, fp64_reference(query, key, value + 1),
                                   rtol=1e-3, atol=2e-4)
        for tensor, original in zip((query, key, value), originals):
            self.assertNotEqual(first.data_ptr(), tensor.data_ptr())
            self.assertNotEqual(second.data_ptr(), tensor.data_ptr())
            torch.testing.assert_close(tensor, original, rtol=0, atol=0)
        torch.testing.assert_close(first, saved, rtol=0, atol=0)

    def test_half_current_stream_orders_inputs_and_consumers(self):
        query, key, value = self.inputs()
        self.attention(query, key, value)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10000000)
            query.zero_()
            key.zero_()
            value[:, 0].fill_(3)
            value[:, 1].fill_(7)
            actual = self.attention(query, key, value) * 2
        stream.synchronize()
        expected = torch.empty_like(query)
        expected[:, :7].fill_(6)
        expected[:, 7:].fill_(14)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_each_argument_rejects_mixed_supported_dtypes(self):
        for base_dtype, other_dtype in ((torch.float16, torch.float32),
                                        (torch.float32, torch.float16)):
            valid = [tensor.to(base_dtype) for tensor in self.inputs(length=5)]
            for index in range(3):
                args = valid.copy()
                args[index] = args[index].to(other_dtype)
                with self.subTest(base_dtype=base_dtype, argument=index):
                    with self.assertRaisesRegex(RuntimeError, 'same dtype'):
                        self.attention(*args)

    def test_each_argument_rejects_unsupported_dtypes(self):
        valid = list(self.inputs(length=5))
        for dtype in (torch.bfloat16, torch.float64, torch.int64):
            for index in range(3):
                args = valid.copy()
                args[index] = args[index].to(dtype)
                with self.subTest(dtype=dtype, argument=index):
                    with self.assertRaisesRegex(RuntimeError, 'float32 or float16'):
                        self.attention(*args)
            with self.subTest(dtype=dtype, all_arguments=True):
                with self.assertRaisesRegex(RuntimeError, 'float32 or float16'):
                    self.attention(*(tensor.to(dtype) for tensor in valid))

    @unittest.skipUnless(torch.cuda.device_count() > 1, 'two CUDA GPUs required')
    def test_half_device_guard_and_mismatched_devices(self):
        with torch.cuda.device(0):
            query = torch.ones(1, 14, 1, 64, device='cuda:1', dtype=torch.float16)
            key = torch.ones(1, 2, 33, 64, device='cuda:1', dtype=torch.float16)
            actual = self.attention(query, key, key)
            torch.testing.assert_close(actual, query, rtol=0, atol=0)
            self.assertEqual(torch.cuda.current_device(), 0)
            for index in range(3):
                args = [query, key, key]
                args[index] = args[index].to('cuda:0')
                with self.subTest(argument=index):
                    with self.assertRaisesRegex(RuntimeError, 'same device'):
                        self.attention(*args)


if __name__ == '__main__':
    unittest.main()

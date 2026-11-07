"""Rectangular causal attention, grouped-head mapping, and allocation checks."""
import importlib
import math
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, eager_attention_forward


class CachedAttentionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('micro_vllm.attention'),
                             'cached attention module must exist')
        self.attention = importlib.import_module('micro_vllm.attention')

    def test_mask_offsets_and_diagonal(self):
        actual = self.attention.build_causal_mask(3, 5, 2, torch.device('cpu'))
        expected = torch.tensor([[False, False, False, True, True],
                                 [False, False, False, False, True],
                                 [False, False, False, False, False]])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.dtype, torch.bool)
        self.assertEqual(actual.device, torch.device('cpu'))
        torch.testing.assert_close(self.attention.build_causal_mask(3, 3, 0, 'cpu'),
                                   torch.ones(3, 3, dtype=torch.bool).triu(1))

    def test_single_query_needs_no_mask_allocation(self):
        with patch.object(torch, 'arange', side_effect=AssertionError('mask allocation')):
            self.assertIsNone(self.attention.build_causal_mask(1, 5, 4, 'cpu'))
            query = torch.zeros(1, 14, 1, 2)
            key = torch.zeros(1, 2, 5, 2)
            value = torch.tensor(
                [[[[0., 1.], [2., 3.], [4., 5.], [6., 7.], [8., 9.]],
                  [[10., 11.], [12., 13.], [14., 15.], [16., 17.], [18., 19.]]]])
            actual = self.attention.causal_attention(query, key, value, past_length=4)
        expected = torch.tensor([[[[4., 5.]]] * 7 + [[[14., 15.]]] * 7])
        torch.testing.assert_close(actual, expected)

    def test_prefix_averages_map_seven_query_heads_per_kv_head(self):
        query = torch.zeros(2, 14, 3, 2)
        key = torch.zeros(2, 2, 5, 2)
        value = torch.arange(40.).reshape(2, 2, 5, 2)
        originals = [item.clone() for item in (query, key, value)]
        actual = self.attention.causal_attention(query, key, value, past_length=2)
        expected = torch.empty_like(query)
        for batch in range(2):
            for head in range(14):
                for position in range(3):
                    expected[batch, head, position] = value[batch, head // 7, :position + 3].mean(0)
        torch.testing.assert_close(actual, expected)
        for item, original in zip((query, key, value), originals):
            torch.testing.assert_close(item, original, rtol=0, atol=0)

    def test_nonuniform_scores_use_head_dimension_scaling(self):
        query = torch.tensor([[[[2., 0., 0., 0.], [2., 0., 0., 0.]]]])
        key = torch.tensor([[[[0., 0., 0., 0.], [1., 0., 0., 0.], [2., 0., 0., 0.]]]])
        value = torch.tensor([[[[1., 2., 3., 4.], [10., 20., 30., 40.], [100., 200., 300., 400.]]]])
        first = (1 + 10 * math.e) / (1 + math.e)
        second = (1 + 10 * math.e + 100 * math.e ** 2) / (1 + math.e + math.e ** 2)
        expected = torch.tensor([[[[first, 2 * first, 3 * first, 4 * first],
                                   [second, 2 * second, 3 * second, 4 * second]]]])
        torch.testing.assert_close(self.attention.causal_attention(query, key, value, past_length=1), expected)

    def test_future_values_do_not_leak_into_rectangular_prefix(self):
        query = torch.ones(1, 4, 2, 4)
        key = torch.zeros(1, 2, 5, 4)
        value = torch.ones_like(key)
        expected = self.attention.causal_attention(query, key, value, past_length=3)
        key[:, :, -1] = 100
        value[:, :, -1] = 100
        actual = self.attention.causal_attention(query, key, value, past_length=3)
        torch.testing.assert_close(actual[:, :, :1], expected[:, :, :1], rtol=0, atol=0)
        self.assertTrue((actual[:, :, 1:] > 1).all())

    def test_shared_mask_is_reused_and_kv_storage_is_not_expanded(self):
        query = torch.randn(2, 14, 3, 4)
        key, value = torch.randn(2, 2, 5, 4), torch.randn(2, 2, 5, 4)
        mask = self.attention.build_causal_mask(3, 5, 2, query.device)
        expected = self.attention.causal_attention(query, key, value, past_length=2)
        matmul = torch.matmul
        operands = []
        def checked_matmul(left, right):
            operands.append((left.shape, right.shape, right.untyped_storage().data_ptr()))
            return matmul(left, right)
        with patch.object(self.attention, 'build_causal_mask', side_effect=AssertionError('mask rebuilt')), \
             patch.object(torch.Tensor, 'repeat_interleave', side_effect=AssertionError('KV repeated')), \
             patch.object(torch, 'matmul', side_effect=checked_matmul):
            actual = self.attention.causal_attention(query, key, value, past_length=2, attention_mask=mask)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(operands, [
            (torch.Size([2, 2, 21, 4]), torch.Size([2, 2, 4, 5]), key.untyped_storage().data_ptr()),
            (torch.Size([2, 2, 21, 5]), torch.Size([2, 2, 5, 4]), value.untyped_storage().data_ptr()),
        ])

    def test_hugging_face_eager_parity_for_cached_and_uncached_noncontiguous_inputs(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for kv_heads in (2, 14):
                    config = Qwen2Config(hidden_size=56, num_attention_heads=14,
                                         num_key_value_heads=kv_heads, attention_dropout=0.)
                    module = Qwen2Attention(config, layer_idx=0).eval()
                    for query_length, past_length in ((1, 0), (3, 0), (1, 5), (3, 5)):
                        with self.subTest(device=device, dtype=dtype, kv_heads=kv_heads,
                                          query_length=query_length, past_length=past_length), torch.random.fork_rng():
                            torch.manual_seed(31)
                            key_length = query_length + past_length
                            query = torch.randn(2, query_length, 14, 4, dtype=dtype, device=device).transpose(1, 2)
                            key = torch.randn(2, key_length, kv_heads, 4, dtype=dtype, device=device).transpose(1, 2)
                            value = torch.randn(2, key_length, kv_heads, 4, dtype=dtype, device=device).transpose(1, 2)
                            blocked = torch.arange(key_length, device=device)[None, :] > (
                                past_length + torch.arange(query_length, device=device)[:, None])
                            mask = torch.zeros(query_length, key_length, device=device, dtype=dtype).masked_fill(blocked, float('-inf'))
                            expected, _ = eager_attention_forward(module, query, key, value,
                                mask[None, None], scaling=4 ** -0.5, dropout=0.)
                            actual = self.attention.causal_attention(query, key, value, past_length=past_length)
                            self.assertEqual(actual.dtype, dtype)
                            self.assertEqual(actual.device, query.device)
                            torch.testing.assert_close(actual, expected.transpose(1, 2))

    def test_rejects_invalid_lengths_and_nonintegral_dimensions(self):
        for args in ((0, 2, 2), (2, 0, 0), (2, 2, -1), (2, 3, 0),
                     (1, 2, 0), (2.0, 2, 0), (2, 2.0, 0), (2, 2, 0.0),
                     (True, 1, 0), (1, True, 0), (1, 1, False)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.attention.build_causal_mask(*args, device='cpu')
        query, key = torch.zeros(1, 4, 2, 4), torch.zeros(1, 2, 2, 4)
        for past_length in (-1, 1, 0.0, False):
            with self.subTest(past_length=past_length), self.assertRaises(ValueError):
                self.attention.causal_attention(query, key, key, past_length=past_length)

    def test_rejects_invalid_tensor_shapes_dtypes_devices_and_layouts(self):
        query, key = torch.zeros(1, 4, 2, 4), torch.zeros(1, 2, 2, 4)
        invalid = [
            (query.squeeze(0), key, key), (query, key, key.squeeze(0)),
            (query, key, torch.zeros(1, 2, 3, 4)),
            (query, torch.zeros(2, 2, 2, 4), torch.zeros(2, 2, 2, 4)),
            (query, torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),
            (query, torch.zeros(1, 2, 2, 8), torch.zeros(1, 2, 2, 8)),
            (torch.zeros(1, 3, 2, 4), key, key),
            (query.long(), key.long(), key.long()), (query, key.half(), key.half()),
            (query, key, key.half()),
            (query, key.to('meta'), key.to('meta')),
            (query.to_sparse(), key, key), (query, key.to_sparse(), key.to_sparse()),
        ]
        for axis in range(4):
            shape = list(query.shape)
            shape[axis] = 0
            invalid.append((torch.zeros(shape), key, key))
            shape = list(key.shape)
            shape[axis] = 0
            invalid.append((query, torch.zeros(shape), torch.zeros(shape)))
        for index, args in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.attention.causal_attention(*args)

    def test_rejects_invalid_shared_masks(self):
        query, key = torch.zeros(1, 4, 2, 4), torch.zeros(1, 2, 2, 4)
        masks = [torch.zeros(2, 2), torch.zeros(1, 2, 2, dtype=torch.bool),
                 torch.zeros(2, 3, dtype=torch.bool),
                 torch.zeros(2, 2, dtype=torch.bool, device='meta'),
                 torch.zeros(2, 2, dtype=torch.bool).to_sparse()]
        for mask in masks:
            with self.subTest(shape=mask.shape, dtype=mask.dtype, device=mask.device), self.assertRaises(ValueError):
                self.attention.causal_attention(query, key, key, attention_mask=mask)


if __name__ == '__main__':
    unittest.main()

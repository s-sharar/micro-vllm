"""Uncached causal attention checks, including grouped query heads."""
import math
import os
import unittest

import torch
from transformers import AutoModelForCausalLM, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, eager_attention_forward

import model


def reference_mask(tokens, dtype):
    return torch.full((tokens, tokens), float('-inf'), dtype=dtype).triu(1)[None, None]


class CausalAttentionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(model, 'causal_attention'),
                        'model.py must provide causal_attention')

    def test_uniform_scores_average_only_visible_values_with_correct_kv_groups(self):
        query = torch.zeros(1, 4, 3, 2)
        key = torch.zeros(1, 2, 3, 2)
        value = torch.tensor([[[[2., 4.], [6., 8.], [10., 12.]],
                               [[20., 40.], [60., 80.], [100., 120.]]]])
        originals = tuple(item.clone() for item in (query, key, value))
        expected = torch.tensor([[[[2., 4.], [4., 6.], [6., 8.]],
                                  [[2., 4.], [4., 6.], [6., 8.]],
                                  [[20., 40.], [40., 60.], [60., 80.]],
                                  [[20., 40.], [40., 60.], [60., 80.]]]])
        actual = model.causal_attention(query, key, value)
        torch.testing.assert_close(actual, expected)
        for current, original in zip((query, key, value), originals):
            torch.testing.assert_close(current, original, rtol=0, atol=0)

    def test_scaled_scores_produce_hand_calculated_weighted_values(self):
        query = torch.tensor([[[[2., 0., 0., 0.], [2., 0., 0., 0.]]]])
        key = torch.tensor([[[[0., 0., 0., 0.], [1., 0., 0., 0.]]]])
        value = torch.tensor([[[[1., 2., 3., 4.], [10., 20., 30., 40.]]]])
        second_weight = math.e / (1 + math.e)
        expected = torch.tensor([[[[1., 2., 3., 4.],
                                  [1 + 9 * second_weight, 2 + 18 * second_weight,
                                   3 + 27 * second_weight, 4 + 36 * second_weight]]]])
        torch.testing.assert_close(model.causal_attention(query, key, value), expected)

    def test_changing_future_keys_and_values_cannot_change_prefix(self):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            query = torch.randn(2, 4, 4, 4)
            key, value = torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4)
        expected = model.causal_attention(query, key, value)
        changed_key = torch.cat((key[:, :, :2], key[:, :, 2:] + 100), dim=2)
        changed_value = torch.cat((value[:, :, :2], value[:, :, 2:] - 100), dim=2)
        actual = model.causal_attention(query, changed_key, changed_value)
        torch.testing.assert_close(actual[:, :, :2], expected[:, :, :2], rtol=0, atol=0)

    def test_matches_hugging_face_for_grouped_heads_lengths_and_dtypes(self):
        for kv_heads in (2, 4):
            config = Qwen2Config(hidden_size=16, num_attention_heads=4,
                                 num_key_value_heads=kv_heads, attention_dropout=0.)
            config._attn_implementation = 'eager'
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for tokens in (1, 3):
                    with self.subTest(kv_heads=kv_heads, dtype=dtype, tokens=tokens):
                        with torch.random.fork_rng():
                            torch.manual_seed(23)
                            attention = Qwen2Attention(config, layer_idx=0).eval()
                            query = torch.randn(2, tokens, 4, 4, dtype=dtype).transpose(1, 2)
                            key = torch.randn(2, tokens, kv_heads, 4, dtype=dtype).transpose(1, 2)
                            value = torch.randn(2, tokens, kv_heads, 4, dtype=dtype).transpose(1, 2)
                        expected, _ = eager_attention_forward(
                            attention, query, key, value,
                            reference_mask(tokens, dtype), scaling=4 ** -0.5, dropout=0.)
                        actual = model.causal_attention(query, key, value)
                        self.assertEqual(actual.shape, (2, 4, tokens, 4))
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual, expected.transpose(1, 2))

    def test_rejects_incompatible_shapes_heads_and_dtypes(self):
        query, key, value = torch.zeros(1, 4, 2, 4), torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4)
        invalid = (
            (query.squeeze(0), key, value),
            (query, key, torch.zeros(1, 2, 3, 4)),
            (query, torch.zeros(2, 2, 2, 4), torch.zeros(2, 2, 2, 4)),
            (query, torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),
            (query, torch.zeros(1, 2, 2, 8), torch.zeros(1, 2, 2, 8)),
            (torch.zeros(1, 3, 2, 4), key, value),
            (torch.zeros(1, 0, 2, 4), key, value),
            (query.long(), key.long(), value.long()),
            (query, key.half(), value.half()),
            (query, torch.empty_like(key, device='meta'), torch.empty_like(value, device='meta')),
        )
        for index, tensors in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                model.causal_attention(*tensors)

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_matches_pretrained_qwen_before_output_projection(self):
        reference = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True,
            dtype=torch.float32, attn_implementation='eager',
        ).eval()
        config, layer = reference.config, reference.model.layers[0]
        attention = layer.self_attn
        ids = torch.tensor([[0, 10, 20, 20, 100]])
        positions = torch.arange(ids.shape[1]).unsqueeze(0)
        head_dim = config.hidden_size // config.num_attention_heads
        with torch.inference_mode():
            hidden = model.embed_tokens(ids, reference.get_input_embeddings().weight)
            normalized = model.rms_norm(hidden, layer.input_layernorm.weight, config.rms_norm_eps)
            query, key, value = model.project_qkv(
                normalized, attention.q_proj.weight, attention.k_proj.weight,
                attention.v_proj.weight, attention.q_proj.bias,
                attention.k_proj.bias, attention.v_proj.bias)
            query = model.split_heads(query, config.num_attention_heads, head_dim)
            key = model.split_heads(key, config.num_key_value_heads, head_dim)
            value = model.split_heads(value, config.num_key_value_heads, head_dim)
            cos, sin = model.rope_cos_sin(positions, head_dim, config.rope_theta, dtype=query.dtype)
            query, key = model.apply_rope(query, key, cos, sin)
            actual = model.causal_attention(query, key, value)
            expected_hidden = layer.input_layernorm(reference.get_input_embeddings()(ids))
            position_embeddings = reference.model.rotary_emb(expected_hidden, positions)
            captured = []
            handle = attention.o_proj.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].clone()))
            try:
                reference_output, _ = attention(expected_hidden, position_embeddings=position_embeddings,
                          attention_mask=reference_mask(ids.shape[1], expected_hidden.dtype))
            finally:
                handle.remove()
        merged = model.merge_heads(actual)
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(merged, captured[0], rtol=1e-5, atol=1e-6)
        with torch.inference_mode():
            projected = model.attention_output(actual, attention.o_proj.weight)
        torch.testing.assert_close(projected, reference_output, rtol=1e-5, atol=1e-6)

class AttentionOutputTests(unittest.TestCase):
    def setUp(self):
        for name in ('merge_heads', 'attention_output'):
            self.assertTrue(hasattr(model, name), f'model.py must provide {name}')

    def test_merge_concatenates_heads_for_each_token(self):
        heads = torch.tensor([[[[1., 2.], [3., 4.]],
                               [[5., 6.], [7., 8.]]]])
        original = heads.clone()
        expected = torch.tensor([[[1., 2., 5., 6.], [3., 4., 7., 8.]]])
        actual = model.merge_heads(heads)
        self.assertEqual(actual.shape, (1, 2, 4))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(heads, original, rtol=0, atol=0)

    def test_dense_output_projection_matches_hand_calculated_head_mixing(self):
        heads = torch.tensor([[[[1., 2.], [3., 4.]],
                               [[5., 6.], [7., 8.]]]])
        weight = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., -1.],
                               [1., 2., 3., 4.]])
        original_heads, original_weight = heads.clone(), weight.clone()
        expected = torch.tensor([[[6., -4., 44.], [10., -4., 64.]]])
        actual = model.attention_output(heads, weight)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(heads, original_heads, rtol=0, atol=0)
        torch.testing.assert_close(weight, original_weight, rtol=0, atol=0)

    def test_matches_hugging_face_projection_for_batched_noncontiguous_heads(self):
        config = Qwen2Config(hidden_size=16, num_attention_heads=4,
                             num_key_value_heads=2)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for tokens in (1, 3):
                with self.subTest(dtype=dtype, tokens=tokens), torch.random.fork_rng():
                    torch.manual_seed(29)
                    attention = Qwen2Attention(config, layer_idx=0).to(dtype=dtype).eval()
                    heads = torch.randn(2, tokens, 4, 4, dtype=dtype).transpose(1, 2)
                    if tokens > 1:
                        self.assertFalse(heads.is_contiguous())
                    expected_merged = torch.cat([heads[:, head] for head in range(4)], dim=-1)
                    with torch.inference_mode():
                        expected = attention.o_proj(expected_merged)
                        actual = model.attention_output(heads, attention.o_proj.weight)
                        merged = model.merge_heads(heads)
                    torch.testing.assert_close(merged, expected_merged, rtol=0, atol=0)
                    self.assertEqual(actual.shape, (2, tokens, 16))
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, expected)

if __name__ == '__main__':
    unittest.main()

"""Default Qwen RoPE: independent calculations and Hugging Face comparisons."""
import math
import os
import unittest

import torch
from transformers import AutoModelForCausalLM, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding, apply_rotary_pos_emb,
)

import model


class RoPETests(unittest.TestCase):
    def setUp(self):
        for name in ('rope_cos_sin', 'rotate_half', 'apply_rope'):
            self.assertTrue(hasattr(model, name), f'model.py must provide {name}')

    def test_rotation_uses_split_halves_without_mutating_input(self):
        values = torch.tensor([[[[1., 2., 3., 4.]]]])
        expected = torch.tensor([[[[-3., -4., 1., 2.]]]])
        torch.testing.assert_close(model.rotate_half(values), expected, rtol=0, atol=0)
        self.assertEqual(values.tolist(), [[[[1., 2., 3., 4.]]]])

    def test_frequencies_match_scalar_calculation(self):
        positions = torch.tensor([[0, 3]])
        cos, sin = model.rope_cos_sin(positions, 4, 10000., dtype=torch.float32)
        angles = (3., 0.03, 3., 0.03)
        expected_cos = torch.tensor([[[1.] * 4, [math.cos(x) for x in angles]]])
        expected_sin = torch.tensor([[[0.] * 4, [math.sin(x) for x in angles]]])
        torch.testing.assert_close(cos, expected_cos, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(sin, expected_sin, rtol=1e-6, atol=1e-7)

    def test_quarter_turn_broadcasts_over_unequal_head_counts(self):
        query = torch.arange(1., 17.).reshape(1, 2, 2, 4)
        key = torch.tensor([[[[20., 30., 40., 50.], [60., 70., 80., 90.]]]])
        cos, sin = torch.zeros(1, 2, 4), torch.ones(1, 2, 4)
        actual_q, actual_k = model.apply_rope(query, key, cos, sin)
        expected_q = torch.tensor([[[[-3., -4., 1., 2.], [-7., -8., 5., 6.]],
                                    [[-11., -12., 9., 10.], [-15., -16., 13., 14.]]]])
        expected_k = torch.tensor([[[[-40., -50., 20., 30.],
                                     [-80., -90., 60., 70.]]]])
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)

    def test_zero_position_identity_and_length_preservation(self):
        positions = torch.tensor([[0, 1, 57], [0, 7, 101]])
        with torch.random.fork_rng():
            torch.manual_seed(13)
            query = torch.randn(2, 3, 4, 8).transpose(1, 2)
            key = torch.randn(2, 3, 2, 8).transpose(1, 2)
        original_q, original_k = query.clone(), key.clone()
        cos, sin = model.rope_cos_sin(positions, 8, 1000000., dtype=query.dtype)
        actual_q, actual_k = model.apply_rope(query, key, cos, sin)
        for actual, original in ((actual_q, original_q), (actual_k, original_k)):
            torch.testing.assert_close(actual[:, :, 0], original[:, :, 0], rtol=0, atol=0)
            torch.testing.assert_close(actual.square().sum(-1), original.square().sum(-1))
        torch.testing.assert_close(query, original_q, rtol=0, atol=0)
        torch.testing.assert_close(key, original_k, rtol=0, atol=0)

    def test_matches_hugging_face_for_dtypes_batches_and_position_offsets(self):
        for head_dim in (2, 8):
            config = Qwen2Config(hidden_size=4 * head_dim, num_attention_heads=4,
                                 num_key_value_heads=2, rope_theta=1000000.)
            reference = Qwen2RotaryEmbedding(config)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for values in ([[0, 1, 2], [31, 127, 2047]], [[30000], [32767]]):
                    with self.subTest(head_dim=head_dim, dtype=dtype, positions=values):
                        positions = torch.tensor(values)
                        tokens = positions.shape[1]
                        with torch.random.fork_rng():
                            torch.manual_seed(17)
                            query = torch.randn(2, tokens, 4, head_dim, dtype=dtype).transpose(1, 2)
                            key = torch.randn(2, tokens, 2, head_dim, dtype=dtype).transpose(1, 2)
                        expected_cos, expected_sin = reference(query, positions)
                        cos, sin = model.rope_cos_sin(
                            positions, head_dim, config.rope_theta, dtype=dtype)
                        torch.testing.assert_close(cos, expected_cos, rtol=0, atol=0)
                        torch.testing.assert_close(sin, expected_sin, rtol=0, atol=0)
                        expected = apply_rotary_pos_emb(query, key, expected_cos, expected_sin)
                        actual = model.apply_rope(query, key, cos, sin)
                        for result, target in zip(actual, expected):
                            self.assertEqual(result.dtype, dtype)
                            torch.testing.assert_close(result, target, rtol=0, atol=0)

    def test_rejects_invalid_head_dimensions_and_base(self):
        positions = torch.tensor([[0]])
        for head_dim, theta in ((0, 10000.), (3, 10000.), (4, 0.),
                                (4, -1.), (4, float('nan')), (4, float('inf'))):
            with self.subTest(head_dim=head_dim, theta=theta):
                with self.assertRaises(ValueError):
                    model.rope_cos_sin(positions, head_dim, theta, dtype=torch.float32)
        with self.assertRaises(ValueError):
            model.rotate_half(torch.zeros(1, 3))

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_matches_cached_qwen_projection_through_rope(self):
        reference = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True,
            dtype=torch.float32, attn_implementation='eager',
        ).eval()
        config, layer = reference.config, reference.model.layers[0]
        ids, positions = torch.tensor([[0, 10, 20, 20, 100]]), torch.tensor([[7, 8, 9, 10, 11]])
        head_dim = config.hidden_size // config.num_attention_heads
        attention = layer.self_attn
        with torch.inference_mode():
            hidden = model.embed_tokens(ids, reference.get_input_embeddings().weight)
            normalized = model.rms_norm(hidden, layer.input_layernorm.weight, config.rms_norm_eps)
            query, key, _ = model.project_qkv(
                normalized, attention.q_proj.weight, attention.k_proj.weight,
                attention.v_proj.weight, attention.q_proj.bias,
                attention.k_proj.bias, attention.v_proj.bias)
            query = model.split_heads(query, config.num_attention_heads, head_dim)
            key = model.split_heads(key, config.num_key_value_heads, head_dim)
            cos, sin = model.rope_cos_sin(positions, head_dim, config.rope_theta, dtype=query.dtype)
            actual = model.apply_rope(query, key, cos, sin)
            expected_hidden = layer.input_layernorm(reference.get_input_embeddings()(ids))
            expected_q = attention.q_proj(expected_hidden).view(1, 5, 14, 64).transpose(1, 2)
            expected_k = attention.k_proj(expected_hidden).view(1, 5, 2, 64).transpose(1, 2)
            expected_cos, expected_sin = reference.model.rotary_emb(expected_hidden, positions)
            expected = apply_rotary_pos_emb(expected_q, expected_k, expected_cos, expected_sin)
        for result, target in zip(actual, expected):
            torch.testing.assert_close(result, target, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

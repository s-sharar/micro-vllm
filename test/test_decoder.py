"""One full Qwen pre-norm decoder layer, without caching or padding."""
import math
import os
import unittest

import torch
from transformers import AutoModelForCausalLM, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from micro_vllm import model


def controlled_weights():
    return {
        'input_layernorm.weight': torch.ones(2),
        'post_attention_layernorm.weight': torch.tensor([2., 0.5]),
        'self_attn.q_proj.weight': torch.eye(2),
        'self_attn.k_proj.weight': torch.eye(2),
        'self_attn.v_proj.weight': torch.eye(2),
        'self_attn.q_proj.bias': torch.zeros(2),
        'self_attn.k_proj.bias': torch.zeros(2),
        'self_attn.v_proj.bias': torch.zeros(2),
        'self_attn.o_proj.weight': torch.eye(2),
        'mlp.gate_proj.weight': torch.eye(2),
        'mlp.up_proj.weight': torch.tensor([[1., 1.], [1., -1.]]),
        'mlp.down_proj.weight': torch.eye(2),
    }


def causal_mask(tokens, dtype):
    return torch.full((tokens, tokens), float('-inf'), dtype=dtype).triu(1)[None, None]


class DecoderLayerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(model, 'decoder_layer'),
                        'model.py must provide decoder_layer')

    def test_hand_calculated_two_residuals_and_norm_order_without_mutation(self):
        hidden = torch.tensor([[[3., 4.]]])
        weights = controlled_weights()
        originals = {name: value.clone() for name, value in weights.items()}
        original_hidden = hidden.clone()
        first_denominator = math.sqrt(12.5 + 0.25)
        h0, h1 = 3 + 3 / first_denominator, 4 + 4 / first_denominator
        second_denominator = math.sqrt((h0 * h0 + h1 * h1) / 2 + 0.25)
        g0, g1 = 2 * h0 / second_denominator, 0.5 * h1 / second_denominator
        expected = torch.tensor([[[
            h0 + g0 / (1 + math.exp(-g0)) * (g0 + g1),
            h1 + g1 / (1 + math.exp(-g1)) * (g0 - g1),
        ]]])
        actual = model.decoder_layer(
            hidden, weights, torch.ones(1, 1, 2), torch.zeros(1, 1, 2),
            num_attention_heads=1, num_key_value_heads=1, eps=0.25)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(hidden, original_hidden, rtol=0, atol=0)
        self.assertEqual(weights.keys(), originals.keys())
        for name, original in originals.items():
            torch.testing.assert_close(weights[name], original, rtol=0, atol=0)

    def test_zero_attention_and_mlp_updates_preserve_original_input(self):
        weights = {**controlled_weights(), 'self_attn.o_proj.weight': torch.zeros(2, 2),
                   'mlp.down_proj.weight': torch.zeros(2, 2)}
        hidden = torch.tensor([[[1., 2.], [3., 4.]], [[-1., 1.], [2., -3.]]])
        positions = torch.tensor([[0, 1]])
        cos, sin = model.rope_cos_sin(positions, 2, 1000000., dtype=hidden.dtype)
        actual = model.decoder_layer(hidden, weights, cos, sin,
                                     num_attention_heads=1, num_key_value_heads=1, eps=1e-6)
        torch.testing.assert_close(actual, hidden, rtol=0, atol=0)

    def test_matches_hugging_face_across_dtypes_batches_and_lengths(self):
        config = Qwen2Config(hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.)
        config._attn_implementation = 'eager'
        rotary = Qwen2RotaryEmbedding(config)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for tokens in (1, 3):
                with self.subTest(dtype=dtype, tokens=tokens), torch.random.fork_rng():
                    torch.manual_seed(41)
                    reference = Qwen2DecoderLayer(config, layer_idx=0).to(dtype=dtype).eval()
                    hidden = torch.randn(2, tokens, 16, dtype=dtype)
                    positions = torch.stack((torch.arange(tokens), torch.arange(tokens) + 17))
                    cos, sin = model.rope_cos_sin(positions, 4, config.rope_theta, dtype=dtype)
                    with torch.inference_mode():
                        expected = reference(hidden, attention_mask=causal_mask(tokens, dtype),
                                             position_embeddings=rotary(hidden, positions), use_cache=False)
                        actual = model.decoder_layer(
                            hidden, reference.state_dict(), cos, sin,
                            num_attention_heads=4, num_key_value_heads=2, eps=config.rms_norm_eps)
                    self.assertEqual(actual.shape, hidden.shape)
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, expected)

    def test_future_tokens_cannot_change_prefix_of_complete_layer(self):
        config = Qwen2Config(hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2)
        with torch.random.fork_rng():
            torch.manual_seed(43)
            reference = Qwen2DecoderLayer(config, layer_idx=0).eval()
            hidden = torch.randn(2, 4, 16)
        cos, sin = model.rope_cos_sin(torch.arange(4).unsqueeze(0), 4,
                                      config.rope_theta, dtype=hidden.dtype)
        changed = torch.cat((hidden[:, :2], hidden[:, 2:] + 10), dim=1)
        arguments = dict(num_attention_heads=4, num_key_value_heads=2, eps=config.rms_norm_eps)
        with torch.inference_mode():
            expected = model.decoder_layer(hidden, reference.state_dict(), cos, sin, **arguments)
            actual = model.decoder_layer(changed, reference.state_dict(), cos, sin, **arguments)
        torch.testing.assert_close(actual[:, :2], expected[:, :2])

    def test_rejects_incompatible_heads_and_position_tables(self):
        hidden, weights = torch.ones(1, 1, 2), controlled_weights()
        cos, sin = torch.ones(1, 1, 2), torch.zeros(1, 1, 2)
        invalid = (
            (hidden.squeeze(0), cos, sin, 1, 1),
            (hidden, cos, sin, 0, 1),
            (hidden, cos, sin, 1, 0),
            (hidden, cos, sin, 3, 1),
            (hidden, cos, sin, 1, 2),
            (hidden, torch.ones(1, 2), sin, 1, 1),
            (hidden, cos, torch.zeros(1, 1, 4), 1, 1),
            (hidden, torch.ones(2, 1, 2), torch.zeros(2, 1, 2), 1, 1),
            (hidden, torch.ones(1, 2, 2), torch.zeros(1, 2, 2), 1, 1),
        )
        for index, (values, cosines, sines, heads, kv_heads) in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                model.decoder_layer(values, weights, cosines, sines,
                                    num_attention_heads=heads, num_key_value_heads=kv_heads, eps=1e-6)

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_matches_cached_pretrained_first_and_last_qwen_layers(self):
        reference = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True,
            dtype=torch.float32, attn_implementation='eager',
        ).eval()
        config = reference.config
        ids, positions = torch.tensor([[0, 10, 20, 20, 100]]), torch.arange(5).unsqueeze(0)
        with torch.inference_mode():
            hidden = model.embed_tokens(ids, reference.get_input_embeddings().weight)
            head_dim = config.hidden_size // config.num_attention_heads
            cos, sin = model.rope_cos_sin(positions, head_dim, config.rope_theta, dtype=hidden.dtype)
            expected_positions = reference.model.rotary_emb(hidden, positions)
            for index in (0, config.num_hidden_layers - 1):
                with self.subTest(layer=index):
                    layer = reference.model.layers[index]
                    expected = layer(hidden, attention_mask=causal_mask(5, hidden.dtype),
                                     position_embeddings=expected_positions, use_cache=False)
                    actual = model.decoder_layer(
                        hidden, layer.state_dict(), cos, sin,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads, eps=config.rms_norm_eps)
                    torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

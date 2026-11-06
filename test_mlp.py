"""Qwen gated MLP checks against scalar math and pretrained reference modules."""
import math
import os
import unittest

import torch
from transformers import AutoModelForCausalLM, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

import model


class MLPTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(model, 'mlp'), 'model.py must provide mlp')

    def test_matches_hand_calculated_gated_mlp_without_mutation(self):
        hidden = torch.tensor([[[1., 2.], [-1., 1.]]])
        gate = torch.tensor([[1., 0.], [0., -1.], [1., 1.]])
        up = torch.tensor([[0., 2.], [3., 0.], [-1., 1.]])
        down = torch.tensor([[1., 2., -1.], [-2., 1., 3.]])
        originals = tuple(value.clone() for value in (hidden, gate, up, down))
        s1 = 1 / (1 + math.exp(-1))
        sn2 = -2 / (1 + math.exp(2))
        s3 = 3 / (1 + math.exp(-3))
        sn1 = -1 / (1 + math.exp(1))
        expected = torch.tensor([[[4 * s1 + 6 * sn2 - s3,
                                   -8 * s1 + 3 * sn2 + 3 * s3],
                                  [-4 * sn1, -7 * sn1]]])
        actual = model.mlp(hidden, gate, up, down)
        self.assertEqual(actual.shape, hidden.shape)
        torch.testing.assert_close(actual, expected)
        for current, original in zip((hidden, gate, up, down), originals):
            torch.testing.assert_close(current, original, rtol=0, atol=0)

    def test_zero_input_produces_zero_output(self):
        actual = model.mlp(torch.zeros(2, 3, 4), torch.ones(6, 4),
                           torch.ones(6, 4), torch.ones(4, 6))
        torch.testing.assert_close(actual, torch.zeros(2, 3, 4), rtol=0, atol=0)

    def test_changing_one_token_does_not_change_other_tokens(self):
        with torch.random.fork_rng():
            torch.manual_seed(31)
            hidden = torch.randn(2, 8, 3).transpose(1, 2)
            gate, up, down = torch.randn(12, 8), torch.randn(12, 8), torch.randn(8, 12)
        self.assertFalse(hidden.is_contiguous())
        expected = model.mlp(hidden, gate, up, down)
        changed = torch.cat((hidden[:, :1] + 1, hidden[:, 1:]), dim=1)
        actual = model.mlp(changed, gate, up, down)
        # Different input layouts can select different FP32 matrix kernels.
        torch.testing.assert_close(actual[:, 1:], expected[:, 1:])

    def test_matches_hugging_face_across_dtypes_and_lengths(self):
        config = Qwen2Config(hidden_size=8, intermediate_size=12, hidden_act='silu')
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for tokens in (1, 3):
                with self.subTest(dtype=dtype, tokens=tokens), torch.random.fork_rng():
                    torch.manual_seed(37)
                    reference = Qwen2MLP(config).to(dtype=dtype).eval()
                    hidden = torch.randn(2, tokens, 8, dtype=dtype)
                    with torch.inference_mode():
                        actual = model.mlp(hidden, reference.gate_proj.weight,
                                           reference.up_proj.weight, reference.down_proj.weight)
                        expected = reference(hidden)
                    self.assertEqual(actual.shape, (2, tokens, 8))
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_matches_cached_pretrained_qwen_mlps(self):
        reference = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True,
            dtype=torch.float32, attn_implementation='eager',
        ).eval()
        ids = torch.tensor([[0, 10, 20]])
        with torch.inference_mode():
            embedded = reference.get_input_embeddings()(ids)
            for index in (0, reference.config.num_hidden_layers - 1):
                with self.subTest(layer=index):
                    layer = reference.model.layers[index]
                    hidden = layer.post_attention_layernorm(embedded)
                    actual = model.mlp(hidden, layer.mlp.gate_proj.weight,
                                       layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)
                    expected = layer.mlp(hidden)
                    self.assertEqual(actual.shape, (1, 3, reference.config.hidden_size))
                    torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

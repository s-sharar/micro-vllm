"""Embedding checks against explicit vectors and Hugging Face Qwen.

Set RUN_MODEL_TESTS=1 to include the locally cached pretrained checkpoint.
"""
import importlib.util
import os
import unittest

import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

if importlib.util.find_spec('micro_vllm.model') is not None:
    from micro_vllm import model
else:
    model = None


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(model, 'model.py must provide embed_tokens')

    def test_selects_vectors_in_order_with_repeated_ids_and_batches(self):
        weight = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        original = weight.clone()
        input_ids = torch.tensor([[2, 0, 2], [1, 2, 0]])
        expected = torch.tensor([
            [[5., 6.], [1., 2.], [5., 6.]],
            [[3., 4.], [5., 6.], [1., 2.]],
        ])
        actual = model.embed_tokens(input_ids, weight)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(weight, original, rtol=0, atol=0)

    def test_matches_hugging_face_embedding(self):
        config = Qwen2Config(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        )
        with torch.random.fork_rng():
            torch.manual_seed(7)
            reference = Qwen2ForCausalLM(config).eval()
        embedding = reference.get_input_embeddings()
        for values in ([[0]], [[15, 0, 7, 7]], [[0, 1], [15, 7]]):
            with self.subTest(values=values), torch.inference_mode():
                ids = torch.tensor(values)
                expected = embedding(ids)
                actual = model.embed_tokens(ids, embedding.weight)
                self.assertEqual(actual.shape, (*ids.shape, config.hidden_size))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_negative_and_out_of_vocabulary_ids(self):
        weight = torch.zeros(3, 2)
        for invalid in (-1, 3):
            with self.subTest(invalid=invalid):
                with self.assertRaises(IndexError):
                    model.embed_tokens(torch.tensor([[invalid]]), weight)

    def test_rejects_floating_point_token_ids(self):
        with self.assertRaises(RuntimeError):
            model.embed_tokens(torch.tensor([[1.0]]), torch.zeros(3, 2))

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_matches_cached_pretrained_qwen_embedding(self):
        reference = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True,
            dtype=torch.float32, attn_implementation='eager',
        ).eval()
        embedding = reference.get_input_embeddings()
        ids = torch.tensor([[0, 10, 20, 20, reference.config.vocab_size - 1]])
        with torch.inference_mode():
            actual = model.embed_tokens(ids, embedding.weight)
            expected = embedding(ids)
        self.assertEqual(actual.shape, (1, 5, reference.config.hidden_size))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

class RMSNormTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(model is not None and hasattr(model, 'rms_norm'),
                        'model.py must provide rms_norm')

    def test_matches_manual_calculation_per_token_without_mutation(self):
        hidden = torch.tensor([[[3., 4.], [-3., 4.]],
                               [[4., -3.], [0., 0.]]])
        weight = torch.tensor([2., 0.5])
        original_hidden, original_weight = hidden.clone(), weight.clone()
        eps = 0.25
        denominator = (12.5 + eps) ** 0.5
        expected = torch.tensor([
            [[6. / denominator, 2. / denominator],
             [-6. / denominator, 2. / denominator]],
            [[8. / denominator, -1.5 / denominator], [0., 0.]],
        ])
        actual = model.rms_norm(hidden, weight, eps)
        self.assertEqual(actual.shape, hidden.shape)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(hidden, original_hidden, rtol=0, atol=0)
        torch.testing.assert_close(weight, original_weight, rtol=0, atol=0)

    def test_zero_vectors_remain_finite(self):
        actual = model.rms_norm(torch.zeros(2, 3, 8), torch.ones(8), 1e-6)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, torch.zeros(2, 3, 8), rtol=0, atol=0)

    def test_matches_hugging_face_across_dtypes(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

        values = [[[1000., -2000., 3000., -4000., 500., -600., 700., -800.],
                   [0.125, -0.3, 0.7, -1.2, 2.1, -3.7, 0.01, 0.99]]]
        for input_dtype in (torch.float32, torch.float16, torch.bfloat16):
            for weight_dtype in (input_dtype, torch.float32):
                with self.subTest(input_dtype=input_dtype, weight_dtype=weight_dtype):
                    reference = Qwen2RMSNorm(8, eps=1e-6).to(dtype=weight_dtype)
                    weight = torch.linspace(0.5, 1.5, 8, dtype=weight_dtype)
                    reference.load_state_dict({'weight': weight})
                    hidden = torch.tensor(values, dtype=input_dtype)
                    with torch.inference_mode():
                        expected = reference(hidden)
                        actual = model.rms_norm(hidden, reference.weight, 1e-6)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_matches_cached_pretrained_qwen_norms(self):
        reference = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True,
            dtype=torch.float32, attn_implementation='eager',
        ).eval()
        ids = torch.tensor([[0, 10, 20, 20, reference.config.vocab_size - 1]])
        norms = (reference.model.layers[0].input_layernorm,
                 reference.model.layers[0].post_attention_layernorm,
                 reference.model.norm)
        with torch.inference_mode():
            hidden = model.embed_tokens(ids, reference.get_input_embeddings().weight)
            for norm in norms:
                with self.subTest(norm=norm):
                    expected = norm(hidden)
                    actual = model.rms_norm(hidden, norm.weight,
                                            reference.config.rms_norm_eps)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

class QKVProjectionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(model is not None and hasattr(model, 'project_qkv'),
                        'model.py must provide project_qkv')

    def project(self, hidden, attention):
        return model.project_qkv(
            hidden,
            q_weight=attention.q_proj.weight,
            k_weight=attention.k_proj.weight,
            v_weight=attention.v_proj.weight,
            q_bias=attention.q_proj.bias,
            k_bias=attention.k_proj.bias,
            v_bias=attention.v_proj.bias,
        )

    def test_matches_hand_calculated_projections_without_mutation(self):
        hidden = torch.tensor([[[1., 2.], [3., 4.]],
                               [[-1., 1.], [0., 0.]]])
        weights = (
            torch.tensor([[1., 0.], [0., 2.], [1., -1.], [-1., -1.]]),
            torch.tensor([[2., 1.], [-1., 1.]]),
            torch.tensor([[1., -2.], [3., 1.]]),
        )
        biases = (torch.tensor([0.5, -0.5, 1., 2.]),
                  torch.tensor([-1., 2.]), torch.tensor([0.25, -0.25]))
        inputs = (hidden, *weights, *biases)
        originals = tuple(value.clone() for value in inputs)
        expected = (
            torch.tensor([[[1.5, 3.5, 0., -1.], [3.5, 7.5, 0., -5.]],
                          [[-0.5, 1.5, -1., 2.], [0.5, -0.5, 1., 2.]]]),
            torch.tensor([[[3., 3.], [9., 3.]], [[-2., 4.], [-1., 2.]]]),
            torch.tensor([[[-2.75, 4.75], [-4.75, 12.75]],
                          [[-2.75, -2.25], [0.25, -0.25]]]),
        )
        actual = model.project_qkv(*inputs)
        self.assertEqual(len(actual), 3)
        for result, target in zip(actual, expected):
            torch.testing.assert_close(result, target, rtol=0, atol=0)
        for original, current in zip(originals, inputs):
            torch.testing.assert_close(current, original, rtol=0, atol=0)

    def test_matches_hugging_face_projections_across_dtypes(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

        config = Qwen2Config(hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.random.fork_rng():
                torch.manual_seed(11)
                attention = Qwen2Attention(config, layer_idx=0).to(dtype=dtype).eval()
                hidden = torch.randn(2, 3, 16, dtype=dtype)
                with torch.inference_mode():
                    actual = self.project(hidden, attention)
                    expected = (attention.q_proj(hidden), attention.k_proj(hidden),
                                attention.v_proj(hidden))
                self.assertEqual([tuple(value.shape) for value in actual],
                                 [(2, 3, 16), (2, 3, 8), (2, 3, 8)])
                for result, target in zip(actual, expected):
                    torch.testing.assert_close(result, target, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_matches_cached_qwen_embedding_norm_projection_pipeline(self):
        reference = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True,
            dtype=torch.float32, attn_implementation='eager',
        ).eval()
        layer = reference.model.layers[0]
        ids = torch.tensor([[0, 10, 20, 20, reference.config.vocab_size - 1]])
        with torch.inference_mode():
            hidden = model.embed_tokens(ids, reference.get_input_embeddings().weight)
            normalized = model.rms_norm(hidden, layer.input_layernorm.weight,
                                        reference.config.rms_norm_eps)
            actual = self.project(normalized, layer.self_attn)
            reference_hidden = layer.input_layernorm(reference.get_input_embeddings()(ids))
            expected = (layer.self_attn.q_proj(reference_hidden),
                        layer.self_attn.k_proj(reference_hidden),
                        layer.self_attn.v_proj(reference_hidden))
        self.assertEqual([tuple(value.shape) for value in actual],
                         [(1, 5, 896), (1, 5, 128), (1, 5, 128)])
        for result, target in zip(actual, expected):
            torch.testing.assert_close(result, target, rtol=0, atol=0)

        heads = (reference.config.num_attention_heads,
                 reference.config.num_key_value_heads,
                 reference.config.num_key_value_heads)
        head_dim = reference.config.hidden_size // reference.config.num_attention_heads
        for projected, target, count in zip(actual, expected, heads):
            headed = model.split_heads(projected, count, head_dim)
            expected_heads = torch.stack(target.split(head_dim, dim=-1), dim=1)
            self.assertEqual(headed.shape, (1, count, 5, head_dim))
            torch.testing.assert_close(headed, expected_heads, rtol=0, atol=0)

class SplitHeadsTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(model is not None and hasattr(model, 'split_heads'),
                        'model.py must provide split_heads')

    def test_splits_each_token_vector_into_heads(self):
        projected = torch.tensor([[[0., 1., 2., 3., 4., 5.],
                                   [10., 11., 12., 13., 14., 15.]]])
        original = projected.clone()
        expected = torch.tensor([[[[0., 1., 2.], [10., 11., 12.]],
                                  [[3., 4., 5.], [13., 14., 15.]]]])
        actual = model.split_heads(projected, num_heads=2, head_dim=3)
        self.assertEqual(actual.shape, (1, 2, 2, 3))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(projected, original, rtol=0, atol=0)

    def test_preserves_axes_for_noncontiguous_batched_input(self):
        projected = torch.arange(120, dtype=torch.float32).reshape(3, 2, 20).transpose(0, 1)
        self.assertFalse(projected.is_contiguous())
        actual = model.split_heads(projected, num_heads=4, head_dim=5)
        self.assertEqual(actual.shape, (2, 4, 3, 5))
        self.assertEqual(actual.dtype, projected.dtype)
        self.assertEqual(actual.device, projected.device)
        for batch in range(2):
            for head in range(4):
                for token in range(3):
                    expected = projected[batch, token, head * 5:(head + 1) * 5]
                    torch.testing.assert_close(actual[batch, head, token], expected,
                                               rtol=0, atol=0)

    def test_rejects_incompatible_projection_width(self):
        with self.assertRaises(RuntimeError):
            model.split_heads(torch.zeros(2, 3, 19), num_heads=4, head_dim=5)

if __name__ == '__main__':
    unittest.main()

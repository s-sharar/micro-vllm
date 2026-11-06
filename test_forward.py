"""Full uncached Qwen logits without calling a Hugging Face forward internally."""
import math
import os
from types import SimpleNamespace
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import model


def tiny_config(**updates):
    settings = dict(vocab_size=16, hidden_size=8, intermediate_size=12,
                    num_hidden_layers=2, num_attention_heads=2,
                    num_key_value_heads=1, max_position_embeddings=64,
                    attention_dropout=0., tie_word_embeddings=False)
    config = Qwen2Config(**{**settings, **updates})
    config._attn_implementation = 'eager'
    return config


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(model, 'forward'), 'model.py must provide forward')

    def test_matches_hugging_face_for_layers_dtypes_and_tied_or_untied_heads(self):
        for tied in (False, True):
            config = tiny_config(tie_word_embeddings=tied, num_hidden_layers=3)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(tied=tied, dtype=dtype), torch.random.fork_rng():
                    torch.manual_seed(47)
                    reference = Qwen2ForCausalLM(config).to(dtype=dtype).eval()
                    weights = reference.state_dict()
                    if tied:
                        weights = {name: value for name, value in weights.items()
                                   if name != 'lm_head.weight'}
                    for values in ([[0]], [[0, 1, 2], [3, 4, 5]]):
                        ids = torch.tensor(values)
                        with torch.inference_mode():
                            expected = reference(input_ids=ids, use_cache=False).logits
                            actual = model.forward(ids, weights, config)
                        self.assertEqual(actual.shape, (*ids.shape, config.vocab_size))
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual, expected)

    def test_matches_explicit_position_ids_and_preserves_inputs_and_weights(self):
        config = tiny_config()
        with torch.random.fork_rng():
            torch.manual_seed(53)
            reference = Qwen2ForCausalLM(config).eval()
        weights = reference.state_dict()
        originals = {name: value.clone() for name, value in weights.items()}
        ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        original_ids = ids.clone()
        for positions in (torch.tensor([[7, 8, 9]]), torch.tensor([[0, 1, 2], [17, 18, 19]])):
            with self.subTest(positions=positions.tolist()), torch.inference_mode():
                original_positions = positions.clone()
                actual = model.forward(ids, weights, config, position_ids=positions)
                expected = reference(input_ids=ids, position_ids=positions, use_cache=False).logits
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(positions, original_positions, rtol=0, atol=0)
        torch.testing.assert_close(ids, original_ids, rtol=0, atol=0)
        self.assertEqual(weights.keys(), originals.keys())
        for name, original in originals.items():
            torch.testing.assert_close(weights[name], original, rtol=0, atol=0)

    def test_zero_layer_updates_leave_final_norm_and_vocab_projection(self):
        config = tiny_config(vocab_size=3, hidden_size=2, intermediate_size=2,
                             num_attention_heads=1, num_key_value_heads=1)
        reference = Qwen2ForCausalLM(config).eval()
        weights = {
            **reference.state_dict(),
            'model.embed_tokens.weight': torch.tensor([[3., 4.], [1., 2.], [-1., 1.]]),
            'model.norm.weight': torch.tensor([2., 0.5]),
            'lm_head.weight': torch.tensor([[1., 0.], [0., 1.], [1., -1.]]),
            **{f'model.layers.{index}.{name}': torch.zeros(2, 2)
               for index in range(config.num_hidden_layers)
               for name in ('self_attn.o_proj.weight', 'mlp.down_proj.weight')},
        }
        denominator = math.sqrt(12.5 + config.rms_norm_eps)
        expected = torch.tensor([[[6 / denominator, 2 / denominator, 4 / denominator]]])
        actual = model.forward(torch.tensor([[0]]), weights, config)
        torch.testing.assert_close(actual, expected)

    def test_distinct_later_layer_weights_affect_logits(self):
        config = tiny_config(num_hidden_layers=3)
        with torch.random.fork_rng():
            torch.manual_seed(59)
            reference = Qwen2ForCausalLM(config).eval()
        weights = reference.state_dict()
        key = 'model.layers.2.mlp.down_proj.weight'
        changed = {**weights, key: weights[key] * 20}
        ids = torch.tensor([[1, 2, 3]])
        actual = model.forward(ids, weights, config)
        modified = model.forward(ids, changed, config)
        self.assertFalse(torch.allclose(actual, modified))

    def test_changing_future_ids_does_not_change_prefix_logits(self):
        config = tiny_config()
        with torch.random.fork_rng():
            torch.manual_seed(61)
            reference = Qwen2ForCausalLM(config).eval()
        weights = reference.state_dict()
        original = model.forward(torch.tensor([[1, 2, 3, 4]]), weights, config)
        changed = model.forward(torch.tensor([[1, 2, 7, 8]]), weights, config)
        torch.testing.assert_close(changed[:, :2], original[:, :2])

    def test_rejects_invalid_ids_positions_and_weight_shapes(self):
        config = tiny_config()
        weights = Qwen2ForCausalLM(config).state_dict()
        ids = torch.tensor([[1, 2]])
        for invalid in (torch.tensor([1, 2]), torch.empty(1, 0, dtype=torch.long),
                        torch.tensor([[1., 2.]]), torch.empty(1, 65, dtype=torch.long),
                        torch.empty(1, 2, dtype=torch.long, device='meta')):
            with self.subTest(ids_shape=invalid.shape), self.assertRaises(ValueError):
                model.forward(invalid, weights, config)
        for positions in (torch.tensor([0, 1]), torch.tensor([[0]]),
                          torch.zeros(2, 2, dtype=torch.long), torch.tensor([[0., 1.]]),
                          torch.tensor([[-1, 0]]), torch.tensor([[63, 64]]),
                          torch.empty(1, 2, dtype=torch.long, device='meta')):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                model.forward(ids, weights, config, position_ids=positions)
        for name, shape in (('model.embed_tokens.weight', (16, 4)),
                            ('model.norm.weight', (1,)), ('lm_head.weight', (15, 8))):
            with self.subTest(weight=name), self.assertRaises(ValueError):
                model.forward(ids, {**weights, name: torch.zeros(shape)}, config)
        for invalid in (-1, config.vocab_size):
            with self.subTest(id=invalid), self.assertRaises(IndexError):
                model.forward(torch.tensor([[invalid]]), weights, config)

    def test_rejects_unsupported_architectures_and_configurations(self):
        config = tiny_config()
        weights = Qwen2ForCausalLM(config).state_dict()
        updates = (
            dict(model_type='llama'), dict(hidden_act='relu'),
            dict(rope_scaling={'rope_type': 'linear', 'factor': 2.}),
            dict(use_sliding_window=True), dict(layer_types=['sliding_attention'] * 2),
            dict(partial_rotary_factor=0.5), dict(head_dim=8),
            dict(num_attention_heads=0), dict(num_key_value_heads=0),
            dict(num_attention_heads=3), dict(num_key_value_heads=3),
            dict(num_hidden_layers=0), dict(hidden_size=0),
        )
        for update in updates:
            modified = SimpleNamespace(**{**config.to_dict(), **update})
            with self.subTest(update=update), self.assertRaises(ValueError):
                model.forward(torch.tensor([[1]]), weights, modified)

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_matches_full_cached_pretrained_qwen_logits(self):
        name = 'Qwen/Qwen2.5-0.5B-Instruct'
        reference = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=torch.float32, attn_implementation='eager',
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        weights, config = reference.state_dict(), reference.config
        for prompt in ('Hello.', 'Explain a KV cache in one sentence.'):
            with self.subTest(prompt=prompt), torch.inference_mode():
                inputs = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}], tokenize=True,
                    add_generation_prompt=True, return_tensors='pt')
                actual = model.forward(inputs, weights, config)
                expected = reference(input_ids=inputs, use_cache=False).logits
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                self.assertTrue(torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1)))
                print(f'Full Qwen logits {tuple(actual.shape)}: max error {(actual - expected).abs().max().item():.8f}',
                      flush=True)


if __name__ == '__main__':
    unittest.main()

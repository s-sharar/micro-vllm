"""Cached model execution, shared work, and prepared weights."""
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from micro_vllm import model
from micro_vllm.kv_cache import ContiguousKVCache


def cache_for(config, weights, **changes):
    embedding = weights['model.embed_tokens.weight']
    arguments = dict(num_layers=config.num_hidden_layers, batch_size=2,
                     num_kv_heads=config.num_key_value_heads, capacity=12,
                     head_dim=config.hidden_size // config.num_attention_heads,
                     dtype=embedding.dtype, device=embedding.device)
    return ContiguousKVCache.allocate(**{**arguments, **changes})


class GuardedWeights(dict):
    blocked = False

    def __getitem__(self, key):
        if self.blocked:
            raise AssertionError('prepared execution queried the source weight map again')
        return super().__getitem__(key)

    def items(self):
        if self.blocked:
            raise AssertionError('prepared execution rescanned the weight map')
        return super().items()


class CachedForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                max_position_embeddings=64, attention_dropout=0.,
                                eos_token_id=None, tie_word_embeddings=True)
        cls.config._attn_implementation = 'eager'
        with torch.random.fork_rng():
            torch.manual_seed(73)
            cls.reference = Qwen2ForCausalLM(cls.config).eval()
        cls.weights = cls.reference.state_dict()
        cls.ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])

    def setUp(self):
        self.assertTrue(hasattr(model, 'prepare_model'), 'model.py must provide prepare_model')

    def test_prepared_logits_match_reference_without_rereading_weight_mapping(self):
        weights = GuardedWeights(self.weights)
        prepared = model.prepare_model(weights, self.config)
        weights.blocked = True
        with torch.inference_mode():
            expected = self.reference(input_ids=self.ids, use_cache=False).logits
        actual = model.forward(self.ids, prepared)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(model.forward(self.ids[:, :3], prepared), expected[:, :3])

    def test_prefill_single_decode_and_multi_token_chunks_match_full_logits(self):
        prepared = model.prepare_model(self.weights, self.config)
        cache = cache_for(self.config, self.weights)
        addresses = tuple(t.data_ptr() for t in (*cache._keys, *cache._values))
        expected = model.forward(self.ids, prepared)
        pieces = []
        offset = 0
        first_prefix = None
        for count in (2, 1, 3, 1):
            actual = model.forward(self.ids[:, offset:offset + count], prepared, cache=cache)
            torch.testing.assert_close(actual, expected[:, offset:offset + count])
            pieces.append(actual)
            offset += count
            self.assertEqual(cache.sequence_length, offset)
            if first_prefix is None:
                first_prefix = tuple(t.clone() for t in cache.read(0))
            else:
                for current, original in zip(cache.read(0), first_prefix):
                    torch.testing.assert_close(current[:, :, :2], original, rtol=0, atol=0)
            self.assertEqual(tuple(t.data_ptr() for t in (*cache._keys, *cache._values)), addresses)
        torch.testing.assert_close(torch.cat(pieces, dim=1), expected)

    def test_cache_contains_rotated_keys_and_unrotated_values_with_kv_heads(self):
        prepared = model.prepare_model(self.weights, self.config)
        cache = cache_for(self.config, self.weights)
        ids = self.ids[:, :3]
        model.forward(ids, prepared, cache=cache)
        layer = self.reference.model.layers[0]
        with torch.inference_mode():
            hidden = layer.input_layernorm(self.reference.get_input_embeddings()(ids))
            q = model.split_heads(layer.self_attn.q_proj(hidden), 4, 4)
            raw_k = model.split_heads(layer.self_attn.k_proj(hidden), 2, 4)
            raw_v = model.split_heads(layer.self_attn.v_proj(hidden), 2, 4)
            cos, sin = model.rope_cos_sin(torch.arange(3).unsqueeze(0), 4,
                                         self.config.rope_theta, dtype=hidden.dtype)
            _, rotated_k = model.apply_rope(q, raw_k, cos, sin)
        stored_k, stored_v = cache.read(0)
        self.assertEqual(stored_k.shape, (2, 2, 3, 4))
        torch.testing.assert_close(stored_k, rotated_k)
        torch.testing.assert_close(stored_v, raw_v)
        self.assertFalse(torch.equal(stored_k[:, :, 1:], raw_k[:, :, 1:]))

    def test_mask_is_constructed_once_and_shared_across_all_layers(self):
        from micro_vllm import runtime

        prepared = model.prepare_model(self.weights, self.config)
        cache = cache_for(self.config, self.weights)
        model.forward(self.ids[:, :2], prepared, cache=cache)
        with patch('micro_vllm.runtime.build_causal_mask', wraps=runtime.build_causal_mask) as build:
            with patch('micro_vllm.model.decoder_layer', wraps=model.decoder_layer) as layers:
                model.forward(self.ids[:, 2:5], prepared, cache=cache)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(layers.call_count, self.config.num_hidden_layers)
        masks = [call.kwargs['attention_mask'] for call in layers.call_args_list]
        self.assertTrue(all(mask is masks[0] for mask in masks))
        self.assertEqual(masks[0].shape, (3, 5))
        expected = torch.tensor([[False, False, False, True, True],
                                 [False, False, False, False, True],
                                 [False, False, False, False, False]])
        self.assertTrue(torch.equal(masks[0], expected))

    def test_last_position_only_projects_requested_logits(self):
        prepared = model.prepare_model(self.weights, self.config)
        all_logits = model.forward(self.ids, prepared)
        with patch('torch.nn.functional.linear', wraps=F.linear) as linear:
            last = model.forward(self.ids, prepared, logits_to_keep=1)
        self.assertEqual(linear.call_args.args[0].shape, (2, 1, 16))
        torch.testing.assert_close(last, all_logits[:, -1:])
        torch.testing.assert_close(model.forward(self.ids, prepared, logits_to_keep=2), all_logits[:, -2:])
        torch.testing.assert_close(model.forward(self.ids, prepared, logits_to_keep=100), all_logits)
        for invalid in (-1, True, 1.5):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                model.forward(self.ids, prepared, logits_to_keep=invalid)

    def test_reset_allows_new_shorter_prompt_without_old_context(self):
        prepared = model.prepare_model(self.weights, self.config)
        cache = cache_for(self.config, self.weights)
        model.forward(self.ids, prepared, cache=cache)
        cache.reset()
        new_ids = self.ids[:, -2:]
        actual = model.forward(new_ids, prepared, cache=cache)
        torch.testing.assert_close(actual, model.forward(new_ids, prepared))
        self.assertEqual(cache.sequence_length, 2)

    def test_rejected_cache_requests_do_not_write_partial_layers(self):
        prepared = model.prepare_model(self.weights, self.config)
        invalid = [cache_for(self.config, self.weights, **change) for change in (
            dict(batch_size=1), dict(num_layers=2), dict(num_kv_heads=1),
            dict(head_dim=8), dict(dtype=torch.float16), dict(device='meta'), dict(capacity=1))]
        misaligned = cache_for(self.config, self.weights)
        misaligned.append(0, torch.ones(2, 2, 1, 4), torch.ones(2, 2, 1, 4))
        invalid.append(misaligned)
        for cache in invalid:
            lengths = tuple(cache.length(index) for index in range(cache.num_layers))
            with self.subTest(lengths=lengths, device=cache.device), self.assertRaises(ValueError):
                model.forward(self.ids[:, :2], prepared, cache=cache)
            self.assertEqual(tuple(cache.length(i) for i in range(cache.num_layers)), lengths)
        cache = cache_for(self.config, self.weights)
        with self.assertRaises(ValueError):
            model.forward(self.ids[:, :2], prepared, cache=cache,
                          position_ids=torch.tensor([[0, 1]]))
        self.assertEqual(cache.sequence_length, 0)
        for invalid_cache in ({}, object()):
            with self.assertRaises(ValueError):
                model.forward(self.ids[:, :2], prepared, cache=invalid_cache)

    def test_preparation_rejects_bad_later_weights_before_any_execution(self):
        for name, tensor in (
            ('model.layers.2.mlp.down_proj.weight', torch.zeros(16, 25)),
            ('model.layers.1.input_layernorm.weight', torch.zeros(1)),
            ('model.layers.0.self_attn.k_proj.weight', torch.zeros(8, 16, dtype=torch.float16)),
        ):
            with self.subTest(weight=name), self.assertRaises(ValueError):
                model.prepare_model({**self.weights, name: tensor}, self.config)

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for actual Qwen checkpoint')
    def test_real_qwen_chunked_cache_matches_full_cpu_and_cuda_logits(self):
        devices = ['cpu']
        if os.environ.get('RUN_GPU_TESTS') == '1' and torch.cuda.is_available():
            devices.append('cuda')
        for device in devices:
            dtype = torch.float32 if device == 'cpu' else torch.float16
            with self.subTest(device=device):
                reference = AutoModelForCausalLM.from_pretrained(
                    'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True,
                    dtype=dtype, attn_implementation='eager').to(device).eval()
                ids = torch.tensor([[10, 20, 30, 40, 50, 60]], device=device)
                config, weights = reference.config, reference.state_dict()
                # Preserve the tight eager-operation reference; accelerated paths have separate tests.
                prepared = model.prepare_model(weights, config, attention_backend='torch')
                cache = cache_for(config, weights, batch_size=1, capacity=6)
                with torch.inference_mode():
                    expected = reference(input_ids=ids, use_cache=False).logits
                chunks, hf_chunks, offset, hf_cache = [], [], 0, None
                with torch.inference_mode():
                    for count in (2, 1, 3):
                        chunk = ids[:, offset:offset + count]
                        chunks.append(model.forward(chunk, prepared, cache=cache))
                        output = reference(input_ids=chunk, past_key_values=hf_cache, use_cache=True)
                        hf_chunks.append(output.logits)
                        hf_cache = output.past_key_values
                        offset += count
                actual, cached_reference = torch.cat(chunks, dim=1), torch.cat(hf_chunks, dim=1)
                # Compare identical chunk shapes tightly. HF's own cached/full paths
                # differ due to matrix-kernel rounding, especially in FP16.
                cached_tolerance = dict(rtol=1e-4, atol=1e-4) if device == 'cpu' else dict(rtol=1e-3, atol=1e-3)
                full_tolerance = dict(rtol=1e-4, atol=2e-4) if device == 'cpu' else dict(rtol=5e-3, atol=0.15)
                torch.testing.assert_close(actual, cached_reference, **cached_tolerance)
                torch.testing.assert_close(actual, expected, **full_tolerance)
                self.assertTrue(torch.equal(actual.argmax(-1), cached_reference.argmax(-1)))
                self.assertTrue(torch.equal(actual.argmax(-1), expected.argmax(-1)))


if __name__ == '__main__':
    unittest.main()

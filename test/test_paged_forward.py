"""Paged caches inside full Qwen execution: prefill handoff and direct decode."""
import os
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from micro_vllm import cuda_attention, cuda_paged_cache, generate, model, reference
from micro_vllm.paged_kv_cache import PagedKVCache
from test.test_accelerated_generation import make_qwen, cache_for

GPU = os.environ.get('RUN_GPU_TESTS') == '1' and torch.cuda.is_available()


def paged_for(checkpoint, *, page_size=2, mapping=(4, 1, 5, 0)):
    cfg, weight = checkpoint.config, checkpoint.get_input_embeddings().weight
    return PagedKVCache.allocate(
        num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
        num_pages=max(mapping)+2, page_size=page_size,
        head_dim=cfg.hidden_size//cfg.num_attention_heads, dtype=weight.dtype,
        device=weight.device, page_table=mapping)


def poison(cache):
    for tensor in cache._keys + cache._values:
        tensor.fill_(float('nan'))


class PagedForwardCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = make_qwen()
        cls.config, cls.weights = cls.checkpoint.config, cls.checkpoint.state_dict()
        cls.ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

    def test_initial_prefill_uses_local_kv_without_gather(self):
        prepared = model.prepare_model(self.weights, self.config, attention_backend='torch')
        for count in (1, 3):
            cache = paged_for(self.checkpoint)
            with patch.object(cache, 'gather', side_effect=AssertionError('no prefill gather')):
                try:
                    actual = prepared.forward(self.ids[:, :count], cache=cache)
                except ValueError as error:
                    self.fail(f'paged cache must be accepted by model prefill: {error}')
            expected = prepared.forward(self.ids[:, :count])
            torch.testing.assert_close(actual, expected)
            self.assertEqual(cache.sequence_length, count)

    def test_chunked_reference_logits_and_stored_kv_match_contiguous(self):
        prepared = model.prepare_model(self.weights, self.config, attention_backend='torch')
        paged, dense = paged_for(self.checkpoint), cache_for(self.checkpoint, capacity=8)
        poison(paged)
        offset, hf_cache = 0, None
        for count in (3, 1, 3):
            chunk = self.ids[:, offset:offset+count]
            actual = prepared.forward(chunk, cache=paged)
            expected = prepared.forward(chunk, cache=dense)
            with torch.inference_mode():
                hf = self.checkpoint(input_ids=chunk, past_key_values=hf_cache, use_cache=True)
                hf_cache = hf.past_key_values
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual, hf.logits, rtol=1e-4, atol=1e-4)
            for layer in range(self.config.num_hidden_layers):
                for left, right in zip(paged.gather(layer), dense.read(layer)):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
            offset += count
        self.assertEqual(paged.sequence_length, 7)

    def test_invalid_forward_preserves_pages_and_lengths(self):
        prepared = model.prepare_model(self.weights, self.config)
        cache = paged_for(self.checkpoint)
        poison(cache)
        prepared.forward(self.ids[:, :3], cache=cache)
        saved = [p.clone() for p in cache._keys + cache._values]
        calls = [
            dict(input_ids=self.ids[:, :6]),
            dict(input_ids=self.ids[:, 3:4], attention_backend='bad'),
            dict(input_ids=self.ids[:, 3:4], attention_backend='cuda'),
            dict(input_ids=self.ids[:, 3:4], position_ids=torch.tensor([[3]])),
            dict(input_ids=self.ids[:, 3:4].expand(2, -1)),
        ]
        for kwargs in calls:
            with self.assertRaises(ValueError):
                prepared.forward(cache=cache, **kwargs)
            self.assertEqual(cache.sequence_length, 3)
            for actual, expected in zip(cache._keys + cache._values, saved):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_rounded_capacity_does_not_extend_model_context(self):
        checkpoint = make_qwen()
        checkpoint.config.max_position_embeddings = 5
        prepared = model.prepare_model(checkpoint.state_dict(), checkpoint.config)
        cache = paged_for(checkpoint, page_size=4, mapping=(2, 0))
        self.assertEqual(cache.capacity, 8)
        prepared.forward(self.ids[:, :5], cache=cache)
        with self.assertRaisesRegex(ValueError, 'context'):
            prepared.forward(self.ids[:, 5:6], cache=cache)
        self.assertEqual(cache.sequence_length, 5)

    def test_explicit_blocked_mask_preserves_reference_semantics(self):
        prepared = model.prepare_model(self.weights, self.config, attention_backend='torch')
        paged, dense = paged_for(self.checkpoint), cache_for(self.checkpoint, capacity=8)
        prepared.forward(self.ids[:, :3], cache=paged)
        prepared.forward(self.ids[:, :3], cache=dense)
        hidden = torch.ones(1, 1, self.config.hidden_size)
        cos, sin = model.rope_cos_sin(torch.tensor([[3]]), prepared.head_dim,
                                      prepared.theta, dtype=hidden.dtype)
        mask = torch.tensor([[True, False, False, False]])
        kwargs = dict(num_attention_heads=prepared.num_attention_heads,
                      num_key_value_heads=prepared.num_key_value_heads, eps=prepared.eps,
                      layer_idx=0, attention_mask=mask, attention_backend='cuda')
        left = model.decoder_layer(hidden, prepared.layers[0], cos, sin, cache=paged, **kwargs)
        right = model.decoder_layer(hidden, prepared.layers[0], cos, sin, cache=dense, **kwargs)
        torch.testing.assert_close(left, right)


@unittest.skipUnless(GPU, 'set RUN_GPU_TESTS=1 for full paged GPU execution')
class PagedForwardGPUTests(unittest.TestCase):
    def test_full_optimized_path_has_no_gather_and_uses_paged_operators(self):
        for dtype in (torch.float32, torch.float16):
            checkpoint = make_qwen(accelerated=True, device='cuda', dtype=dtype)
            cfg, weights = checkpoint.config, checkpoint.state_dict()
            ids = torch.tensor([[10, 20, 30, 40, 50, 60]], device='cuda')
            for mode in ('auto', 'cuda'):
                prepared = model.prepare_model(weights, cfg, attention_backend=mode)
                paged, dense = paged_for(checkpoint, page_size=3), cache_for(checkpoint, capacity=12)
                poison(paged)
                outputs = []
                with patch.object(paged, 'gather', side_effect=AssertionError('must not gather')), \
                        patch.object(model, 'decode_attention_paged',
                                     wraps=cuda_attention.decode_attention_paged) as decode, \
                        patch.object(cuda_paged_cache, '_write_paged_kv',
                                     wraps=cuda_paged_cache._write_paged_kv) as write, \
                        patch.object(model, 'attention_forward', wraps=model.attention_forward) as attention:
                    for start, end in ((0, 3), (3, 4), (4, 5), (5, 6)):
                        outputs.append(prepared.forward(ids[:, start:end], cache=paged))
                    self.assertEqual(decode.call_count, 3 * cfg.num_hidden_layers)
                    self.assertEqual(write.call_count, 4 * cfg.num_hidden_layers)
                    self.assertEqual(attention.call_count, cfg.num_hidden_layers)
                    expected_route = 'flash' if mode == 'auto' and dtype == torch.float16 else 'torch'
                    self.assertTrue(all(call.kwargs['implementation'] == expected_route
                                        for call in attention.call_args_list))
                expected = [prepared.forward(ids[:, a:b], cache=dense)
                            for a, b in ((0, 3), (3, 4), (4, 5), (5, 6))]
                torch.testing.assert_close(torch.cat(outputs, 1), torch.cat(expected, 1), rtol=0, atol=0)
                self.assertEqual(paged.sequence_length, 6)
                for layer in range(cfg.num_hidden_layers):
                    for left, right in zip(paged.gather(layer), dense.read(layer)):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_cached_multi_token_chunk_keeps_correct_reference_fallback(self):
        checkpoint = make_qwen(accelerated=True, device='cuda', dtype=torch.float16)
        prepared = model.prepare_model(checkpoint.state_dict(), checkpoint.config)
        paged, dense = paged_for(checkpoint), cache_for(checkpoint, capacity=8)
        ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70]], device='cuda')
        offset = 0
        for count in (2, 1, 3, 1):
            chunk = ids[:, offset:offset+count]
            left = prepared.forward(chunk, cache=paged)
            right = prepared.forward(chunk, cache=dense)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            offset += count


@unittest.skipUnless(GPU, 'set RUN_GPU_TESTS=1 for offline real-Qwen paged generation')
class RealQwenPagedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name = 'Qwen/Qwen2.5-0.5B-Instruct'
        cls.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        cls.checkpoint = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=torch.float16,
            attn_implementation='eager').to('cuda').eval()
        cls.weights, cls.config = cls.checkpoint.state_dict(), cls.checkpoint.config

    @classmethod
    def tearDownClass(cls):
        del cls.weights, cls.config, cls.checkpoint, cls.tokenizer

    def test_generated_ids_match_hf_and_contiguous_without_paged_gather(self):
        for prompt in ('What is 2 + 2?', 'Explain a KV cache in one sentence.'):
            expected = reference.generate_tokens(self.checkpoint, self.tokenizer, prompt, 8)
            for mode in ('auto', 'cuda'):
                dense = generate.generate_tokens(self.weights, self.config, self.tokenizer, prompt, 8,
                    eos_token_id=self.checkpoint.generation_config.eos_token_id, attention_backend=mode)
                self.assertEqual(dense.tolist(), expected.tolist())
                for size in (3, 16):
                    with patch.object(PagedKVCache, 'gather', side_effect=AssertionError('must not gather')):
                        actual = generate.generate_tokens(self.weights, self.config, self.tokenizer,
                            prompt, 8, eos_token_id=self.checkpoint.generation_config.eos_token_id,
                            attention_backend=mode, kv_cache='paged', page_size=size)
                    self.assertEqual(actual.tolist(), expected.tolist())

    def test_real_prefill_and_decode_logits_match_contiguous_and_hf(self):
        ids = torch.tensor([[10, 20, 30, 40, 50, 60]], device='cuda')
        for mode in ('torch', 'cuda', 'auto'):
            prepared = model.prepare_model(self.weights, self.config, attention_backend=mode)
            paged, dense = paged_for(self.checkpoint), cache_for(self.checkpoint, capacity=8)
            hf_cache, offset = None, 0
            for count in (3, 1, 1, 1):
                chunk = ids[:, offset:offset+count]
                actual = prepared.forward(chunk, cache=paged)
                expected = prepared.forward(chunk, cache=dense)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with torch.inference_mode():
                    hf = self.checkpoint(input_ids=chunk, past_key_values=hf_cache, use_cache=True)
                    hf_cache = hf.past_key_values
                # Grouped/Flash/native FP16 vs HF eager rounding depends on
                # chunk geometry; the dense path has the identical difference.
                # Exact paged/contiguous parity above isolates paging changes.
                torch.testing.assert_close(actual, hf.logits, rtol=5e-3, atol=.125)
                self.assertTrue(torch.equal(actual.argmax(-1), hf.logits.argmax(-1)))
                offset += count


if __name__ == '__main__':
    unittest.main()

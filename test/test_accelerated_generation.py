"""Attention backend integration with RoPE, fixed-capacity caches and generation."""
import os
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from micro_vllm import attention_backend, cuda_attention, generate, model, reference
from micro_vllm.kv_cache import ContiguousKVCache

GPU_TESTS = os.environ.get('RUN_GPU_TESTS') == '1' and torch.cuda.is_available()


def make_qwen(*, accelerated=False, device='cpu', dtype=torch.float32):
    heads, dimension = (14, 64) if accelerated else (4, 4)
    config = Qwen2Config(vocab_size=128 if accelerated else 8,
                        hidden_size=heads * dimension, intermediate_size=128 if accelerated else 24,
                        num_hidden_layers=2, num_attention_heads=heads, num_key_value_heads=2,
                        max_position_embeddings=128, attention_dropout=0.,
                        eos_token_id=None, bos_token_id=None, pad_token_id=0,
                        tie_word_embeddings=True)
    config._attn_implementation = 'eager'
    with torch.random.fork_rng():
        torch.manual_seed(123)
        checkpoint = Qwen2ForCausalLM(config).eval().to(device=device, dtype=dtype)
    return checkpoint


def cache_for(checkpoint, capacity=17, batch_size=1):
    config, embedding = checkpoint.config, checkpoint.get_input_embeddings().weight
    return ContiguousKVCache.allocate(
        num_layers=config.num_hidden_layers, batch_size=batch_size,
        num_kv_heads=config.num_key_value_heads, capacity=capacity,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=embedding.dtype, device=embedding.device)


def make_tokenizer():
    vocabulary = {'<pad>': 0, '<unk>': 1, 'user': 2, 'assistant': 3,
                  'hello': 4, 'world': 5, 'answer': 6, 'yes': 7}
    raw = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='<unk>',
                                       pad_token='<pad>', model_input_names=['input_ids', 'attention_mask'])
    tokenizer.chat_template = (
        '{% for message in messages %}{{ message["role"] + " " + message["content"] + " " }}{% endfor %}'
        '{% if add_generation_prompt %}assistant {% endif %}')
    return tokenizer


class BackendIntegrationCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = make_qwen()
        cls.weights, cls.config = cls.checkpoint.state_dict(), cls.checkpoint.config
        cls.ids = torch.tensor([[1, 2, 3, 4, 5]])
        cls.tokenizer = make_tokenizer()

    def test_auto_and_torch_preserve_cpu_cached_logits(self):
        with torch.inference_mode():
            expected = self.checkpoint(input_ids=self.ids, use_cache=False).logits
        for mode in ('torch', 'auto'):
            prepared = model.prepare_model(self.weights, self.config, attention_backend=mode)
            cache = cache_for(self.checkpoint)
            pieces = []
            for start, end in ((0, 2), (2, 3), (3, 5)):
                pieces.append(prepared.forward(self.ids[:, start:end], cache=cache))
            torch.testing.assert_close(torch.cat(pieces, dim=1), expected)
            self.assertEqual(cache.sequence_length, 5)
            torch.testing.assert_close(model.forward(self.ids, self.weights, self.config,
                                                     attention_backend=mode), expected)

    def test_invalid_backend_override_and_forced_cuda_leave_cache_untouched(self):
        prepared = model.prepare_model(self.weights, self.config, attention_backend='torch')
        cache = cache_for(self.checkpoint)
        model.forward(self.ids[:, :2], prepared, cache=cache)
        before = [tensor.clone() for pair in zip(cache._keys, cache._values) for tensor in pair]
        for mode in ('bad', 'flash', 'cuda'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                model.forward(self.ids[:, 2:3], prepared, cache=cache, attention_backend=mode)
            self.assertEqual(cache.sequence_length, 2)
            for pair, saved_pair in zip(zip(cache._keys, cache._values), zip(before[::2], before[1::2])):
                for current, saved in zip(pair, saved_pair):
                    torch.testing.assert_close(current, saved, rtol=0, atol=0, equal_nan=True)
        with self.assertRaises(ValueError):
            model.prepare_model(self.weights, self.config, attention_backend='flash')

    def test_cpu_generation_selects_backend_and_matches_hf(self):
        for prompt in ('hello', 'hello world answer'):
            expected = reference.generate_tokens(self.checkpoint, self.tokenizer, prompt, 5)
            for mode in ('torch', 'auto'):
                for use_cache in (True, False):
                    with self.subTest(prompt=prompt, mode=mode, cache=use_cache):
                        actual = generate.generate_tokens(
                            self.weights, self.config, self.tokenizer, prompt, 5,
                            eos_token_id=None, use_cache=use_cache, attention_backend=mode)
                        self.assertTrue(torch.equal(actual, expected))
        ids = torch.tensor([[2, 4, 3]])
        for mode in ('invalid', 'cuda'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                generate._greedy_decode(ids, self.weights, self.config, 2,
                                        eos_token_id=None, attention_backend=mode)

    def test_explicit_forward_override_wins_over_prepared_default(self):
        prepared = model.prepare_model(self.weights, self.config, attention_backend='cuda')
        actual = prepared.forward(self.ids, attention_backend='torch')
        expected = model.forward(self.ids, self.weights, self.config, attention_backend='torch')
        torch.testing.assert_close(actual, expected)


@unittest.skipUnless(GPU_TESTS, 'set RUN_GPU_TESTS=1 for CUDA model integration')
class AcceleratedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = make_qwen(accelerated=True, device='cuda', dtype=torch.float16)
        cls.weights, cls.config = cls.checkpoint.state_dict(), cls.checkpoint.config
        cls.ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70]], device='cuda')

    def test_logits_and_cache_match_reference_for_all_backends_and_chunks(self):
        reference_cache, hf_chunks, offset = None, [], 0
        with torch.inference_mode():
            for count in (3, 1, 2, 1):
                output = self.checkpoint(input_ids=self.ids[:, offset:offset + count],
                                         past_key_values=reference_cache, use_cache=True)
                hf_chunks.append(output.logits)
                reference_cache = output.past_key_values
                offset += count
        expected = torch.cat(hf_chunks, dim=1)
        torch_cache = None
        for mode in ('torch', 'cuda', 'auto'):
            prepared = model.prepare_model(self.weights, self.config, attention_backend=mode)
            cache, pieces, offset = cache_for(self.checkpoint), [], 0
            for count in (3, 1, 2, 1):
                pieces.append(model.forward(self.ids[:, offset:offset + count], prepared, cache=cache))
                offset += count
                self.assertEqual(cache.sequence_length, offset)
            actual = torch.cat(pieces, dim=1)
            with self.subTest(mode=mode):
                # FP16 rounding measured below 0.002 for this seeded two-layer model.
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-3)
                self.assertTrue(torch.equal(actual.argmax(-1), expected.argmax(-1)))
                if torch_cache is None:
                    torch_cache = tuple(t.clone() for i in range(cache.num_layers) for t in cache.read(i))
                else:
                    for current, expected_cache in zip(
                            (t for i in range(cache.num_layers) for t in cache.read(i)), torch_cache):
                        torch.testing.assert_close(current, expected_cache, rtol=3e-3, atol=2e-3)

    def test_dispatch_receives_post_rope_q_and_original_cache_prefix_per_layer(self):
        if (not torch.backends.cuda.is_flash_attention_available()
                or torch.cuda.get_device_capability()[0] < 8):
            self.skipTest('FlashAttention requires SM80+')
        prepared = model.prepare_model(self.weights, self.config, attention_backend='auto')
        cache = cache_for(self.checkpoint, capacity=19)
        original_rope, rotated = model.apply_rope, []

        def record_rope(*args, **kwargs):
            result = original_rope(*args, **kwargs)
            rotated.append(result)
            return result

        with patch('micro_vllm.model.apply_rope', side_effect=record_rope), \
                patch('micro_vllm.cuda_attention.decode_attention_gqa',
                      wraps=cuda_attention.decode_attention_gqa) as native, \
                patch('micro_vllm.attention_backend.flash_attention',
                      wraps=attention_backend.flash_attention) as flash:
            model.forward(self.ids[:, :3], prepared, cache=cache)
            self.assertEqual(flash.call_count, 2)
            native.assert_not_called()
            for index, call in enumerate(flash.call_args_list):
                self.assertIs(call.args[0], rotated[index][0])
                self.assertEqual(call.args[1].data_ptr(), cache._keys[index].data_ptr())
                self.assertEqual(call.args[1].stride(1), 19 * 64)
                torch.testing.assert_close(cache.read(index)[0], rotated[index][1], rtol=0, atol=0)
            rotated.clear()
            model.forward(self.ids[:, 3:4], prepared, cache=cache)
            self.assertEqual(native.call_count, 2)
            self.assertEqual(flash.call_count, 2)
            for index, call in enumerate(native.call_args_list):
                query, key, value = call.args
                self.assertIs(query, rotated[index][0])
                self.assertEqual(key.shape, (1, 2, 4, 64))
                self.assertEqual(key.data_ptr(), cache._keys[index].data_ptr())
                self.assertEqual(value.data_ptr(), cache._values[index].data_ptr())
                self.assertEqual(key.stride(1), 19 * 64)
                self.assertEqual(value.stride(1), 19 * 64)
                self.assertFalse(key.is_contiguous())
                torch.testing.assert_close(key[:, :, -1:], rotated[index][1], rtol=0, atol=0)
            rotated.clear()
            model.forward(self.ids[:, 4:6], prepared, cache=cache)
            self.assertEqual(native.call_count, 2)
            self.assertEqual(flash.call_count, 4)
            self.assertEqual(cache.sequence_length, 6)

    def test_forced_cuda_rejects_batch_and_bfloat16_before_cache_write(self):
        for dtype, batch in ((torch.bfloat16, 1), (torch.float16, 2)):
            with self.subTest(dtype=dtype, batch=batch):
                checkpoint = make_qwen(accelerated=True, device='cuda', dtype=dtype)
                weights = checkpoint.state_dict()
                cache = cache_for(checkpoint, batch_size=batch)
                ids = self.ids[:, :2].expand(batch, -1)
                for tensor in (*cache._keys, *cache._values):
                    tensor.fill_(17.)
                with self.assertRaises(ValueError):
                    model.forward(ids, weights, self.config, cache=cache, attention_backend='cuda')
                self.assertEqual(cache.sequence_length, 0)
                for tensor in (*cache._keys, *cache._values):
                    self.assertTrue((tensor == 17.).all())


@unittest.skipUnless(GPU_TESTS, 'set RUN_GPU_TESTS=1 for cached offline Qwen generation')
class RealQwenAcceleratedGenerationTests(unittest.TestCase):
    def test_real_qwen_token_ids_match_torch_cuda_auto_and_huggingface(self):
        name = 'Qwen/Qwen2.5-0.5B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        checkpoint = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=torch.float16,
            attn_implementation='eager').to('cuda').eval()
        weights = checkpoint.state_dict()
        for prompt in ('What is 2 + 2?', 'Explain a KV cache in one sentence.'):
            expected = reference.generate_tokens(checkpoint, tokenizer, prompt, 8)
            for mode in ('torch', 'cuda', 'auto'):
                with self.subTest(prompt=prompt, mode=mode):
                    actual = generate.generate_tokens(
                        weights, checkpoint.config, tokenizer, prompt, 8,
                        eos_token_id=checkpoint.generation_config.eos_token_id,
                        attention_backend=mode)
                    self.assertEqual(actual.tolist(), expected.tolist())


    def test_real_qwen_chunked_logits_have_bounded_fp16_error_and_identical_argmax(self):
        checkpoint = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True,
            dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
        ids = torch.tensor([[10, 20, 30, 40, 50, 60]], device='cuda')
        expected_chunks, hf_cache, offset = [], None, 0
        with torch.inference_mode():
            for count in (2, 1, 3):
                result = checkpoint(input_ids=ids[:, offset:offset + count],
                                    past_key_values=hf_cache, use_cache=True)
                expected_chunks.append(result.logits)
                hf_cache = result.past_key_values
                offset += count
        expected = torch.cat(expected_chunks, dim=1)
        for mode in ('torch', 'cuda', 'auto'):
            with self.subTest(mode=mode):
                prepared = model.prepare_model(checkpoint.state_dict(), checkpoint.config,
                                               attention_backend=mode)
                cache, chunks, offset = cache_for(checkpoint, capacity=6), [], 0
                for count in (2, 1, 3):
                    chunks.append(model.forward(ids[:, offset:offset + count], prepared, cache=cache))
                    offset += count
                actual = torch.cat(chunks, dim=1)
                if mode == 'torch':
                    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
                else:
                    # Across 24 layers the seeded IDs measured max <= 0.098 and mean <= 0.011.
                    difference = (actual.float() - expected.float()).abs()
                    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=.125)
                    self.assertLessEqual(difference.max().item(), .125)
                    self.assertLessEqual(difference.mean().item(), .015)
                self.assertTrue(torch.equal(actual.argmax(-1), expected.argmax(-1)))
                self.assertEqual(cache.sequence_length, 6)


class DirectLayerPreflightTests(unittest.TestCase):
    def layer_request(self, checkpoint, *, batch=1, tokens=1):
        config = checkpoint.config
        embedding = checkpoint.get_input_embeddings().weight
        ids = torch.ones((batch, tokens), dtype=torch.long, device=embedding.device)
        hidden = model.embed_tokens(ids, embedding)
        positions = torch.arange(1, tokens + 1, device=embedding.device).unsqueeze(0)
        cos, sin = model.rope_cos_sin(
            positions, config.hidden_size // config.num_attention_heads,
            config.rope_theta, dtype=embedding.dtype)
        prefix = 'model.layers.0.'
        weights = {name[len(prefix):]: tensor for name, tensor in checkpoint.state_dict().items()
                   if name.startswith(prefix)}
        cache = cache_for(checkpoint, batch_size=batch)
        for storage in (*cache._keys, *cache._values):
            storage.fill_(17.)
        first = torch.ones((batch, config.num_key_value_heads, 1, cache.head_dim),
                           dtype=embedding.dtype, device=embedding.device)
        for index in range(cache.num_layers):
            cache.append(index, first, first)
        arguments = dict(num_attention_heads=config.num_attention_heads,
                         num_key_value_heads=config.num_key_value_heads,
                         eps=config.rms_norm_eps, cache=cache, layer_idx=0)
        return (hidden, weights, cos, sin), arguments, cache

    def assert_native_rejected_before_write(self, checkpoint, *, batch=1, tokens=1):
        with torch.inference_mode():
            args, kwargs, cache = self.layer_request(checkpoint, batch=batch, tokens=tokens)
            buffers = (*cache._keys, *cache._values)
            before = tuple(tensor.clone() for tensor in buffers)
            error = None
            try:
                model.decoder_layer(*args, **kwargs, attention_backend='cuda')
            except Exception as caught:
                error = caught
            self.assertEqual([cache.length(i) for i in range(cache.num_layers)], [1, 1])
            for current, original in zip(buffers, before):
                torch.testing.assert_close(current, original, rtol=0, atol=0)
            self.assertIsInstance(error, ValueError)

    def test_direct_native_cpu_request_preserves_cache_on_rejection(self):
        self.assert_native_rejected_before_write(make_qwen())

    @unittest.skipUnless(GPU_TESTS, 'set RUN_GPU_TESTS=1 for native layer preflight')
    def test_direct_native_invalid_dtype_batch_and_chunk_preserve_cache(self):
        for dtype, batch, tokens in ((torch.bfloat16, 1, 1),
                                     (torch.float16, 2, 1),
                                     (torch.float16, 1, 2)):
            with self.subTest(dtype=dtype, batch=batch, tokens=tokens):
                checkpoint = make_qwen(device='cuda', dtype=dtype)
                self.assert_native_rejected_before_write(checkpoint, batch=batch, tokens=tokens)

    def test_direct_native_label_with_explicit_mask_uses_reference_on_cpu(self):
        # An explicit blocked-position mask intentionally forces reference attention.
        checkpoint = make_qwen()
        with torch.inference_mode():
            args, kwargs, cache = self.layer_request(checkpoint, tokens=2)
            mask = torch.tensor([[False, True, True], [False, True, False]])
            actual = model.decoder_layer(*args, **kwargs, attention_mask=mask,
                                         attention_backend='cuda')
            args, kwargs, reference_cache = self.layer_request(checkpoint, tokens=2)
            expected = model.decoder_layer(*args, **kwargs, attention_mask=mask,
                                           attention_backend='torch')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(cache.length(0), 3)
        for actual, expected in zip(cache.read(0), reference_cache.read(0)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

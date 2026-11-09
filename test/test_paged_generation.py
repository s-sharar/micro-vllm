"""Paged generation uses the same greedy model path with request-local fixed pages."""
import contextlib
import io
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from micro_vllm import generate, reference
from micro_vllm.kv_cache import ContiguousKVCache
from micro_vllm.paged_kv_cache import PagedKVCache
from test import test_generate as fixtures


class PagedGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.GenerationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        fixtures.GenerationTests.tearDownClass.__func__(cls)

    def generate(self, prompt='hello world', budget=4, eos=None, **kwargs):
        return generate.generate_tokens(
            self.weights, self.model.config, self.tokenizer, prompt, budget,
            eos_token_id=eos, kv_cache='paged', **kwargs)

    def test_paged_tokens_match_contiguous_and_hf_across_page_boundaries(self):
        for page_size in (1, 3, 16):
            for prompt, budget in (('hello', 1), ('hello world', 2),
                                   ('hello world answer yes', 6)):
                with self.subTest(page_size=page_size, prompt=prompt, budget=budget):
                    expected = reference.generate_tokens(self.model, self.tokenizer, prompt, budget)
                    contiguous = generate.generate_tokens(
                        self.weights, self.model.config, self.tokenizer, prompt, budget,
                        eos_token_id=None)
                    paged = self.generate(prompt, budget, page_size=page_size)
                    self.assertTrue(torch.equal(paged, expected))
                    self.assertTrue(torch.equal(paged, contiguous))
                    self.assertEqual(paged.shape, (budget,))
                    self.assertFalse(paged.requires_grad)

    def test_paged_allocation_rounds_capacity_and_tracks_each_forward(self):
        for page_size in (1, 3, 16):
            records = []
            real_forward = generate.forward

            def capture(ids, prepared, **kwargs):
                cache = kwargs['cache']
                before = cache.sequence_length
                result = real_forward(ids, prepared, **kwargs)
                records.append((ids.shape[1], cache, before, cache.sequence_length))
                return result

            with self.subTest(page_size=page_size), patch(
                    'micro_vllm.generate.forward', side_effect=capture):
                self.generate(budget=5, page_size=page_size)
            prompt_length = records[0][0]
            capacity = prompt_length + 4
            cache = records[0][1]
            self.assertIsInstance(cache, PagedKVCache)
            self.assertEqual(cache.num_pages, (capacity + page_size - 1) // page_size)
            self.assertEqual(cache.capacity, cache.num_pages * page_size)
            self.assertEqual(cache.page_size, page_size)
            self.assertEqual(cache.page_table, tuple(range(cache.num_pages)))
            self.assertEqual(cache.num_layers, self.model.config.num_hidden_layers)
            self.assertEqual(cache.num_kv_heads, self.model.config.num_key_value_heads)
            self.assertEqual(cache.head_dim, self.model.config.hidden_size
                             // self.model.config.num_attention_heads)
            self.assertEqual(cache.dtype, self.weights['model.embed_tokens.weight'].dtype)
            self.assertEqual(cache.device, self.weights['model.embed_tokens.weight'].device)
            for step, (tokens, seen_cache, before, after) in enumerate(records):
                self.assertIs(seen_cache, cache)
                self.assertEqual(tokens, prompt_length if step == 0 else 1)
                self.assertEqual(before, 0 if step == 0 else prompt_length + step - 1)
                self.assertEqual(after, prompt_length + step)
            self.assertTrue(all(cache.length(i) == capacity for i in range(cache.num_layers)))

    def test_paged_stops_at_first_and_later_eos(self):
        sequence = self.generate(budget=8, page_size=3).tolist()
        self.assertNotIn(sequence[2], sequence[:2])
        prompt_length = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'hello world'}], tokenize=True,
            add_generation_prompt=True, return_tensors='pt').shape[1]
        for eos, count in ((sequence[0], 1), ([sequence[0]], 1), (sequence[2], 3)):
            with self.subTest(eos=eos), patch(
                    'micro_vllm.generate.forward', wraps=generate.forward) as calls:
                actual = self.generate(budget=8, eos=eos, page_size=3)
                self.assertEqual(actual.tolist(), sequence[:count])
                self.assertEqual(calls.call_count, count)
                cache = calls.call_args.kwargs['cache']
                self.assertEqual(cache.sequence_length, prompt_length + count - 1)
                with patch.object(self.model.generation_config, 'eos_token_id', eos):
                    expected = reference.generate_tokens(
                        self.model, self.tokenizer, 'hello world', 8)
                self.assertTrue(torch.equal(actual, expected))

    def test_requests_use_fresh_pages_and_contiguous_remains_default(self):
        caches = []
        real_forward = generate.forward

        def capture(ids, prepared, **kwargs):
            cache = kwargs['cache']
            if cache.sequence_length == 0:
                caches.append(cache)
            return real_forward(ids, prepared, **kwargs)

        with patch('micro_vllm.generate.forward', side_effect=capture):
            first = self.generate(page_size=3)
            second = self.generate(page_size=3)
            contiguous = generate.generate_tokens(
                self.weights, self.model.config, self.tokenizer, 'hello world', 4,
                eos_token_id=None)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, contiguous))
        self.assertEqual(len(caches), 3)
        self.assertIsNot(caches[0], caches[1])
        self.assertNotEqual(caches[0]._keys[0].data_ptr(), caches[1]._keys[0].data_ptr())
        self.assertIsInstance(caches[2], ContiguousKVCache)

    def test_invalid_cache_options_fail_before_prepare_allocation_or_forward(self):
        ids = torch.tensor([[2, 4, 3]])
        cases = [({'kv_cache': 'other'}, 'kv_cache'),
                 ({'kv_cache': None}, 'kv_cache'),
                 ({'kv_cache': 'paged', 'use_cache': False}, 'use_cache')]
        cases.extend(({'kv_cache': mode, 'page_size': value}, 'page_size')
                     for mode in ('paged', 'contiguous')
                     for value in (0, -1, True, 1.5, None, '3'))
        for options, message in cases:
            with (self.subTest(options=options),
                  patch('micro_vllm.generate.prepare_model') as prepare,
                  patch('micro_vllm.generate.forward') as forward,
                  patch.object(PagedKVCache, 'allocate') as paged,
                  patch.object(ContiguousKVCache, 'allocate') as contiguous):
                with self.assertRaisesRegex(ValueError, message):
                    generate._greedy_decode(
                        ids, self.weights, self.model.config, 2,
                        eos_token_id=None, **options)
                prepare.assert_not_called()
                forward.assert_not_called()
                paged.assert_not_called()
                contiguous.assert_not_called()

    def test_public_api_rejects_invalid_cache_options_before_tokenization(self):
        for options in ({'kv_cache': 'other'}, {'page_size': True},
                        {'kv_cache': 'paged', 'use_cache': False}):
            with (self.subTest(options=options),
                  patch.object(self.tokenizer, 'apply_chat_template') as tokenize):
                with self.assertRaises(ValueError):
                    generate.generate_tokens(
                        self.weights, self.model.config, self.tokenizer, 'hello', 2,
                        eos_token_id=None, **options)
                tokenize.assert_not_called()

    def test_parser_cache_options_are_opt_in_and_have_compatible_defaults(self):
        original = reference.parse_args([])
        self.assertFalse(hasattr(original, 'kv_cache'))
        self.assertFalse(hasattr(original, 'page_size'))
        for argv in (['--kv-cache', 'paged'], ['--page-size', '3']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                reference.parse_args(argv)
        defaults = reference.parse_args([], with_kv_cache=True)
        self.assertEqual(defaults.kv_cache, 'contiguous')
        self.assertEqual(defaults.page_size, 16)
        selected = reference.parse_args(
            ['--kv-cache', 'paged', '--page-size', '3'], with_kv_cache=True)
        self.assertEqual((selected.kv_cache, selected.page_size), ('paged', 3))

    def test_invalid_cli_cache_options_fail_before_loading_model(self):
        for argv in (['--kv-cache', 'other'], ['--page-size', '0'],
                     ['--page-size', '-1'], ['--page-size', '1.5']):
            with (self.subTest(argv=argv),
                  patch('micro_vllm.generate.AutoModelForCausalLM.from_pretrained') as model,
                  patch('micro_vllm.generate.AutoTokenizer.from_pretrained') as tokenizer,
                  contextlib.redirect_stderr(io.StringIO()),
                  self.assertRaises(SystemExit) as error):
                generate.main(argv)
            self.assertEqual(error.exception.code, 2)
            model.assert_not_called()
            tokenizer.assert_not_called()

    def test_offline_paged_cli_matches_hf_on_cpu_reference_fallback(self):
        expected = reference.generate_tokens(self.model, self.tokenizer, 'hello', 5).tolist()
        for backend in ('auto', 'torch'):
            with self.subTest(backend=backend):
                result = subprocess.run(
                    [sys.executable, '-m', 'micro_vllm.generate', '--model', str(self.checkpoint),
                     '--device', 'cpu', '--prompt', 'hello', '--max-new-tokens', '5',
                     '--kv-cache', 'paged', '--page-size', '3', '--attention-backend', backend],
                    text=True, capture_output=True, timeout=60,
                    env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'Token IDs: {expected}', result.stdout)
                self.assertIn('Response:', result.stdout)


if __name__ == '__main__':
    unittest.main()

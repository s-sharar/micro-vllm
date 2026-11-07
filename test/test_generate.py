"""Our cached forward drives greedy generation; checkpoint tests stay offline."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from micro_vllm import reference

if importlib.util.find_spec('micro_vllm.generate') is not None:
    from micro_vllm import generate
else:
    generate = None

ROOT = Path(__file__).resolve().parents[1] / 'src' / 'micro_vllm'


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend = Tokenizer(WordLevel(
            {'<pad>': 0, '<unk>': 1, 'user': 2, 'assistant': 3,
             'hello': 4, 'world': 5, 'answer': 6, 'yes': 7}, unk_token='<unk>'))
        backend.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token='<unk>', pad_token='<pad>',
            model_input_names=['input_ids', 'attention_mask'])
        cls.tokenizer.chat_template = (
            '{% for message in messages %}{{ message[\'role\'] + \' \' + '
            'message[\'content\'] + \' \' }}{% endfor %}'
            '{% if add_generation_prompt %}assistant {% endif %}')
        config = Qwen2Config(vocab_size=8, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             max_position_embeddings=128, eos_token_id=None,
                             bos_token_id=None, pad_token_id=0, attention_dropout=0.)
        config._attn_implementation = 'eager'
        with torch.random.fork_rng():
            torch.manual_seed(2)
            cls.model = Qwen2ForCausalLM(config).eval()
        cls.weights = cls.model.state_dict()
        cls.temp = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.temp.name)
        cls.model.save_pretrained(cls.checkpoint)
        cls.tokenizer.save_pretrained(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.assertIsNotNone(generate, 'generate.py must drive our own forward pass')

    def run_generation(self, prompt='hello world', budget=4, eos=None, *, use_cache=True):
        options = {} if use_cache else {'use_cache': False}
        return generate.generate_tokens(self.weights, self.model.config, self.tokenizer,
                                        prompt, budget, eos_token_id=eos, **options)

    def test_tokens_match_reference_across_prompts_and_budgets(self):
        for prompt in ('hello', 'hello world', 'hello world answer yes'):
            for budget in (1, 2, 6):
                with self.subTest(prompt=prompt, budget=budget):
                    expected = reference.generate_tokens(self.model, self.tokenizer, prompt, budget)
                    actual = self.run_generation(prompt, budget)
                    uncached = self.run_generation(prompt, budget, use_cache=False)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(actual, uncached))
                    self.assertEqual(actual.shape, (budget,))
                    self.assertEqual(actual.dtype, torch.long)
                    self.assertFalse(actual.requires_grad)

    def test_default_cached_forward_prefills_then_only_processes_new_tokens(self):
        records = []
        actual_forward = generate.forward

        def capture(ids, weights, config=None, **kwargs):
            cache = kwargs.get('cache')
            before = None if cache is None else cache.sequence_length
            output = actual_forward(ids, weights, config, **kwargs)
            after = None if cache is None else cache.sequence_length
            records.append((ids.clone(), cache, before, after,
                            kwargs.get('logits_to_keep'), torch.is_grad_enabled()))
            return output

        with patch('micro_vllm.generate.forward', side_effect=capture):
            tokens = self.run_generation(budget=4)
        prompt_ids = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'hello world'}], tokenize=True,
            add_generation_prompt=True, return_tensors='pt')
        prompt_length = prompt_ids.shape[1]
        self.assertEqual([record[0].shape[1] for record in records], [prompt_length, 1, 1, 1])
        cache = records[0][1]
        self.assertIsNotNone(cache)
        self.assertEqual(cache.capacity, prompt_length + 3)
        self.assertEqual(cache.dtype, self.weights['model.embed_tokens.weight'].dtype)
        self.assertEqual(cache.device, self.weights['model.embed_tokens.weight'].device)
        for step, (ids, seen_cache, before, after, kept, grad_enabled) in enumerate(records):
            self.assertIs(seen_cache, cache)
            self.assertEqual(before, 0 if step == 0 else prompt_length + step - 1)
            self.assertEqual(after, prompt_length + step)
            self.assertEqual(kept, 1)
            self.assertFalse(grad_enabled)
            expected = prompt_ids if step == 0 else tokens[step - 1:step].unsqueeze(0)
            self.assertTrue(torch.equal(ids, expected))

    def test_cached_path_never_concatenates_growing_token_prefixes(self):
        token_concatenations = []
        actual_cat = torch.cat

        def capture(tensors, *args, **kwargs):
            if tensors[0].dtype == torch.long:
                token_concatenations.append([tuple(tensor.shape) for tensor in tensors])
            return actual_cat(tensors, *args, **kwargs)

        with patch('micro_vllm.generate.torch.cat', side_effect=capture):
            actual = self.run_generation(budget=4)
        self.assertEqual(actual.shape, (4,))
        self.assertEqual(token_concatenations, [[(1, 1)] * 4])

    def test_one_token_budget_can_forward_at_exact_context_limit(self):
        ids = torch.ones((1, self.model.config.max_position_embeddings), dtype=torch.long)
        with patch('micro_vllm.generate.forward', wraps=generate.forward) as call:
            tokens = generate._greedy_decode(
                ids, self.weights, self.model.config, 1, eos_token_id=None)
        self.assertEqual(tokens.shape, (1,))
        self.assertEqual(call.call_count, 1)
        cache = call.call_args.kwargs['cache']
        self.assertEqual(cache.capacity, ids.shape[1])
        self.assertEqual(cache.sequence_length, ids.shape[1])

    def test_model_prepared_once_and_cache_fresh_for_each_request(self):
        caches = []
        actual_forward = generate.forward

        def capture(ids, weights, config=None, **kwargs):
            cache = kwargs.get('cache')
            if cache is not None and cache.sequence_length == 0:
                caches.append(cache)
            return actual_forward(ids, weights, config, **kwargs)

        with patch('micro_vllm.generate.prepare_model', wraps=generate.prepare_model) as prepare:
            with patch('micro_vllm.generate.forward', side_effect=capture):
                first = self.run_generation(budget=3)
                second = self.run_generation(budget=3)
                uncached = self.run_generation(budget=3, use_cache=False)
            self.assertEqual(prepare.call_count, 3)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, uncached))
        self.assertEqual(len(caches), 2)
        self.assertIsNot(caches[0], caches[1])

    def test_uncached_forward_receives_full_growing_context_under_inference_mode(self):
        records = []
        actual_forward = generate.forward

        def capture(ids, weights, config=None, **kwargs):
            records.append((ids.clone(), torch.is_grad_enabled()))
            self.assertIsNone(kwargs.get('cache'))
            self.assertEqual(kwargs.get('logits_to_keep'), 1)
            return actual_forward(ids, weights, config, **kwargs)

        with patch('micro_vllm.generate.forward', side_effect=capture):
            tokens = self.run_generation(budget=3, use_cache=False)
        prompt_ids = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'hello world'}], tokenize=True,
            add_generation_prompt=True, return_tensors='pt')
        self.assertEqual(len(records), 3)
        for step, (ids, grad_enabled) in enumerate(records):
            expected = torch.cat((prompt_ids, tokens[:step].unsqueeze(0)), dim=1)
            self.assertTrue(torch.equal(ids, expected))
            self.assertFalse(grad_enabled)

    def test_integer_list_and_later_eos_are_included_and_stop_generation(self):
        first = self.run_generation(budget=1).item()
        for eos in (first, [(first + 1) % 8, first], (first,)):
            with self.subTest(eos=eos), patch('micro_vllm.generate.forward', wraps=generate.forward) as calls:
                actual = self.run_generation(budget=6, eos=eos)
                self.assertEqual(actual.tolist(), [first])
                self.assertEqual(calls.call_count, 1)
        sequence = self.run_generation(budget=8).tolist()
        self.assertNotIn(sequence[2], sequence[:2], 'fixture must first reach this EOS at step three')
        with patch('micro_vllm.generate.forward', wraps=generate.forward) as calls:
            actual = self.run_generation(budget=8, eos=sequence[2])
            self.assertEqual(actual.tolist(), sequence[:3])
            self.assertEqual(calls.call_count, 3)

    def test_calls_and_input_tensors_remain_independent(self):
        original = self.run_generation('hello', 4)
        self.run_generation('world answer yes', 5)
        self.assertTrue(torch.equal(self.run_generation('hello', 4), original))
        ids = torch.tensor([[2, 4, 3]])
        snapshot = ids.clone()
        generate._greedy_decode(ids, self.weights, self.model.config, 2, eos_token_id=None)
        self.assertTrue(torch.equal(ids, snapshot))

    def test_rejects_invalid_prompts_budgets_eos_and_context_before_forward(self):
        for prompt, budget in ((' ', 1), ('hello', 0), ('hello', -1), ('hello', 1.5), ('hello', True), (None, 1)):
            with (self.subTest(prompt=prompt, budget=budget),
                  patch('micro_vllm.generate.forward') as call,
                  patch('micro_vllm.generate.prepare_model') as prepare):
                with self.assertRaises(ValueError):
                    self.run_generation(prompt, budget)
                call.assert_not_called()
                prepare.assert_not_called()
        for eos in (-1, 8, [1.5], 'stop'):
            with (self.subTest(eos=eos),
                  patch('micro_vllm.generate.forward') as call,
                  patch('micro_vllm.generate.prepare_model') as prepare):
                with self.assertRaises(ValueError):
                    self.run_generation(eos=eos)
                call.assert_not_called()
                prepare.assert_not_called()
        for ids, budget in ((torch.ones(2, 2, dtype=torch.long), 1),
                            (torch.ones(1, 0, dtype=torch.long), 1),
                            (torch.ones(1, 128, dtype=torch.long), 2)):
            with (self.subTest(shape=ids.shape),
                  patch('micro_vllm.generate.forward') as call,
                  patch('micro_vllm.generate.prepare_model') as prepare):
                with self.assertRaises(ValueError):
                    generate._greedy_decode(ids, self.weights, self.model.config, budget, eos_token_id=None)
                call.assert_not_called()
                prepare.assert_not_called()

    def test_main_and_offline_cli_match_reference(self):
        expected = reference.generate_tokens(self.model, self.tokenizer, 'hello', 3).tolist()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(
                Qwen2ForCausalLM, 'forward', side_effect=AssertionError('HF forward called')):
            generate.main(['--model', str(self.checkpoint), '--device', 'cpu',
                           '--prompt', 'hello', '--max-new-tokens', '3'])
        self.assertIn(f'Token IDs: {expected}', output.getvalue())
        result = subprocess.run(
            [sys.executable, '-m', 'micro_vllm.generate', '--model', str(self.checkpoint),
             '--device', 'cpu', '--prompt', 'hello', '--max-new-tokens', '3'],
            text=True, capture_output=True, timeout=60,
            env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'Token IDs: {expected}', result.stdout)
        self.assertIn('Response:', result.stdout)

    def test_help_and_missing_cuda_do_not_load_weights(self):
        with patch('micro_vllm.generate.AutoModelForCausalLM.from_pretrained') as load:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                generate.main(['--help'])
            self.assertEqual(error.exception.code, 0)
            with patch('micro_vllm.generate.torch.cuda.is_available', return_value=False):
                with self.assertRaisesRegex(SystemExit, '--device cpu'):
                    generate.main(['--device', 'cuda'])
            load.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_half_cached_uncached_and_reference_tokens_match(self):
        model = Qwen2ForCausalLM(self.model.config).to(device='cuda', dtype=torch.float16).eval()
        model.load_state_dict(self.weights)
        for prompt, budget in (('hello', 1), ('hello world', 6)):
            with self.subTest(prompt=prompt, budget=budget):
                expected = reference.generate_tokens(model, self.tokenizer, prompt, budget)
                actual = generate.generate_tokens(
                    model.state_dict(), model.config, self.tokenizer, prompt, budget,
                    eos_token_id=None)
                uncached = generate.generate_tokens(
                    model.state_dict(), model.config, self.tokenizer, prompt, budget,
                    eos_token_id=None, use_cache=False)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(actual, uncached))

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_cached_real_qwen_generates_same_tokens_as_reference(self):
        self._check_real_generation('cpu', torch.float32)

    @unittest.skipUnless(os.environ.get('RUN_GPU_TESTS') == '1',
                         'set RUN_GPU_TESTS=1 for cached pretrained GPU weights')
    def test_cuda_real_qwen_cached_uncached_and_reference_tokens_match(self):
        self.assertTrue(torch.cuda.is_available())
        self._check_real_generation('cuda', torch.float16)

    def _check_real_generation(self, device, dtype):
        name = 'Qwen/Qwen2.5-0.5B-Instruct'
        model = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=dtype,
            attn_implementation='eager').to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        weights, config = model.state_dict(), model.config
        eos = model.generation_config.eos_token_id
        for prompt, budget in (('What is the capital of France?', 12),
                               ('Count from 1 to 3.', 12)):
            with self.subTest(prompt=prompt):
                expected = reference.generate_tokens(model, tokenizer, prompt, budget)
                actual = generate.generate_tokens(weights, config, tokenizer, prompt,
                                                   budget, eos_token_id=eos)
                uncached = generate.generate_tokens(
                    weights, config, tokenizer, prompt, budget, eos_token_id=eos,
                    use_cache=False)
                self.assertTrue(torch.equal(actual, expected),
                                f'ours={actual.tolist()} reference={expected.tolist()}')
                self.assertTrue(torch.equal(actual, uncached))
                print(f'Own forward response: {tokenizer.decode(actual, skip_special_tokens=True)}', flush=True)


if __name__ == '__main__':
    unittest.main()

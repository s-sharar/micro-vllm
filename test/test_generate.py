"""Our uncached forward drives greedy generation; checkpoint tests stay offline."""
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

    def run_generation(self, prompt='hello world', budget=4, eos=None):
        return generate.generate_tokens(self.weights, self.model.config, self.tokenizer,
                                        prompt, budget, eos_token_id=eos)

    def test_tokens_match_reference_across_prompts_and_budgets(self):
        for prompt in ('hello', 'hello world', 'hello world answer yes'):
            for budget in (1, 2, 6):
                with self.subTest(prompt=prompt, budget=budget):
                    expected = reference.generate_tokens(self.model, self.tokenizer, prompt, budget)
                    actual = self.run_generation(prompt, budget)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.shape, (budget,))
                    self.assertEqual(actual.dtype, torch.long)
                    self.assertFalse(actual.requires_grad)

    def test_forward_receives_full_growing_context_under_inference_mode(self):
        records = []
        actual_forward = generate.forward

        def capture(ids, weights, config):
            records.append((ids.clone(), torch.is_grad_enabled()))
            return actual_forward(ids, weights, config)

        with patch('micro_vllm.generate.forward', side_effect=capture):
            tokens = self.run_generation(budget=3)
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
            with self.subTest(prompt=prompt, budget=budget), patch('micro_vllm.generate.forward') as call:
                with self.assertRaises(ValueError):
                    self.run_generation(prompt, budget)
                call.assert_not_called()
        for eos in (-1, 8, [1.5], 'stop'):
            with self.subTest(eos=eos), patch('micro_vllm.generate.forward') as call:
                with self.assertRaises(ValueError):
                    self.run_generation(eos=eos)
                call.assert_not_called()
        for ids, budget in ((torch.ones(2, 2, dtype=torch.long), 1),
                            (torch.ones(1, 0, dtype=torch.long), 1),
                            (torch.ones(1, 128, dtype=torch.long), 2)):
            with self.subTest(shape=ids.shape), patch('micro_vllm.generate.forward') as call:
                with self.assertRaises(ValueError):
                    generate._greedy_decode(ids, self.weights, self.model.config, budget, eos_token_id=None)
                call.assert_not_called()

    def test_main_and_offline_cli_match_reference(self):
        expected = reference.generate_tokens(self.model, self.tokenizer, 'hello', 3).tolist()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
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

    @unittest.skipUnless(os.environ.get('RUN_MODEL_TESTS') == '1',
                         'set RUN_MODEL_TESTS=1 for cached pretrained weights')
    def test_cached_real_qwen_generates_same_tokens_as_reference(self):
        name = 'Qwen/Qwen2.5-0.5B-Instruct'
        model = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        weights, config = model.state_dict(), model.config
        eos = model.generation_config.eos_token_id
        for prompt, budget in (('What is the capital of France?', 12),
                               ('Count from 1 to 3.', 12)):
            with self.subTest(prompt=prompt):
                expected = reference.generate_tokens(model, tokenizer, prompt, budget)
                actual = generate.generate_tokens(weights, config, tokenizer, prompt,
                                                   budget, eos_token_id=eos)
                self.assertTrue(torch.equal(actual, expected),
                                f'ours={actual.tolist()} reference={expected.tolist()}')
                print(f'Own forward response: {tokenizer.decode(actual, skip_special_tokens=True)}', flush=True)


if __name__ == '__main__':
    unittest.main()

"""Offline manual-decode tests. Set RUN_GPU_TESTS=1 for cached Qwen GPU checks."""
import contextlib
import copy
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
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
    PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
)

import reference

ROOT = Path(__file__).resolve().parent
if importlib.util.find_spec("decode") is not None:
    import decode


class DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2)
        backend = Tokenizer(WordLevel(
            {"<pad>": 0, "<unk>": 1, "user": 2, "assistant": 3,
             "hello": 4, "world": 5, "answer": 6, "yes": 7},
            unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
            model_input_names=["input_ids", "attention_mask"])
        cls.tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] + ' ' + "
            "message['content'] + ' ' }}{% endfor %}"
            "{% if add_generation_prompt %}assistant {% endif %}")
        config = Qwen2Config(
            vocab_size=8, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=128, eos_token_id=None,
            bos_token_id=None, pad_token_id=0, attention_dropout=0.0)
        config._attn_implementation = "eager"
        cls.model = Qwen2ForCausalLM(config).eval()
        cls.temp = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.temp.name)
        cls.model.save_pretrained(cls.checkpoint)
        cls.tokenizer.save_pretrained(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.assertTrue((ROOT / "decode.py").exists(),
                        "decode.py must implement manual cached generation")

    def inputs(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt")

    def test_tokens_match_reference_for_varied_lengths_and_budgets(self):
        for prompt in ("hello", "hello world", "hello world answer yes"):
            for budget in (1, 2, 6):
                with self.subTest(prompt=prompt, budget=budget):
                    expected = reference.generate_tokens(
                        self.model, self.tokenizer, prompt, budget)
                    actual = decode.generate_tokens(
                        self.model, self.tokenizer, prompt, budget)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.shape, (budget,))
                    self.assertEqual(actual.dtype, torch.long)
                    self.assertFalse(actual.requires_grad)

    def test_prefill_then_single_token_calls_and_cached_logits(self):
        prompt = "hello world"
        inputs = self.inputs(prompt)
        prompt_length = inputs["input_ids"].shape[1]
        records = []

        def capture(module, args, kwargs, output):
            cache = kwargs.get("past_key_values")
            records.append({
                "ids": kwargs["input_ids"].clone(),
                "mask": kwargs["attention_mask"].clone(),
                "positions": kwargs["cache_position"].clone(),
                "cache": cache,
                "cache_length": output.past_key_values.get_seq_length(),
                "logits": output.logits[:, -1, :].clone(),
                "grad_enabled": torch.is_grad_enabled(),
            })

        handle = self.model.register_forward_hook(capture, with_kwargs=True)
        try:
            tokens = decode.generate_tokens(
                self.model, self.tokenizer, prompt, 5)
        finally:
            handle.remove()

        self.assertEqual(len(records), 5)
        self.assertEqual([r["ids"].shape[1] for r in records],
                         [prompt_length, 1, 1, 1, 1])
        for step, record in enumerate(records):
            with self.subTest(step=step):
                length = prompt_length + step
                self.assertEqual(record["mask"].shape, (1, length))
                self.assertTrue(record["mask"].eq(1).all())
                positions = list(range(prompt_length)) if step == 0 else [length - 1]
                self.assertEqual(record["positions"].tolist(), positions)
                self.assertEqual(record["cache_length"], length)
                self.assertFalse(record["grad_enabled"])
                if step == 0:
                    self.assertIsNone(record["cache"])
                else:
                    self.assertIsNotNone(record["cache"])
                    self.assertEqual(record["ids"].item(), tokens[step - 1].item())
                full_ids = torch.cat(
                    (inputs["input_ids"], tokens[:step].unsqueeze(0)), dim=1)
                with torch.inference_mode():
                    uncached = self.model(
                        input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                        use_cache=False, return_dict=True).logits[:, -1, :]
                torch.testing.assert_close(
                    record["logits"], uncached, rtol=1e-5, atol=1e-6)

    def test_stops_on_integer_or_any_list_eos_and_includes_it(self):
        first = reference.generate_tokens(
            self.model, self.tokenizer, "hello", 1).item()
        for eos in (first, [(first + 1) % 8, first]):
            with self.subTest(eos=eos):
                model = copy.deepcopy(self.model)
                model.generation_config = GenerationConfig(eos_token_id=eos)
                actual = decode.generate_tokens(model, self.tokenizer, "hello", 6)
                expected = reference.generate_tokens(model, self.tokenizer, "hello", 6)
                self.assertEqual(actual.tolist(), [first])
                self.assertTrue(torch.equal(actual, expected))

    def test_stops_on_later_eos(self):
        prompt = "hello world"
        sequence = reference.generate_tokens(
            self.model, self.tokenizer, prompt, 8).tolist()
        eos = sequence[2]
        self.assertNotIn(eos, sequence[:2],
                         "fixture must first produce its EOS at step three")
        model = copy.deepcopy(self.model)
        model.generation_config = GenerationConfig(eos_token_id=eos)
        actual = decode.generate_tokens(model, self.tokenizer, prompt, 8)
        expected = reference.generate_tokens(model, self.tokenizer, prompt, 8)
        self.assertEqual(actual.numel(), 3)
        self.assertTrue(torch.equal(actual, expected))

    def test_calls_do_not_share_cache(self):
        first = decode.generate_tokens(self.model, self.tokenizer, "hello", 4)
        decode.generate_tokens(self.model, self.tokenizer, "world answer yes", 6)
        repeated = decode.generate_tokens(self.model, self.tokenizer, "hello", 4)
        self.assertTrue(torch.equal(first, repeated))

    def test_never_calls_generate_and_ignores_sampling_defaults(self):
        model = copy.deepcopy(self.model)
        model.generation_config = GenerationConfig(
            do_sample=True, repetition_penalty=1.5,
            return_dict_in_generate=True, pad_token_id=0)
        expected = reference.generate_tokens(model, self.tokenizer, "hello", 5)
        with patch.object(model, "generate", side_effect=AssertionError(
                "manual decoder must call forward directly")):
            actual = decode.generate_tokens(model, self.tokenizer, "hello", 5)
        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_invalid_function_inputs_before_forward(self):
        for prompt, budget in (("  ", 1), ("hello", 0), ("hello", -1)):
            with self.subTest(prompt=prompt, budget=budget):
                with self.assertRaises(ValueError):
                    decode.generate_tokens(self.model, self.tokenizer, prompt, budget)

    def test_main_loads_local_checkpoint(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            decode.main(["--model", str(self.checkpoint), "--device", "cpu",
                         "--prompt", "hello", "--max-new-tokens", "2"])
        self.assertIn("Generated tokens: 2", output.getvalue())
        self.assertIn("Token IDs:", output.getvalue())
        self.assertIn("Response:", output.getvalue())

    def test_cli_end_to_end_offline(self):
        command = [sys.executable, str(ROOT / "decode.py"), "--model",
                   str(self.checkpoint), "--device", "cpu", "--prompt", "hello",
                   "--max-new-tokens", "3"]
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = reference.generate_tokens(
            self.model, self.tokenizer, "hello", 3).tolist()
        self.assertIn(f"Token IDs: {expected}", result.stdout)

    def test_help_describes_manual_generation(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as error:
                decode.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("Manual greedy", output.getvalue())

    def test_missing_cuda_exits_before_loading_model(self):
        with patch("decode.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(SystemExit, "--device cpu"):
                decode.main(["--device", "cuda"])

    @unittest.skipUnless(os.environ.get("RUN_GPU_TESTS") == "1",
                         "set RUN_GPU_TESTS=1 to use the cached real Qwen checkpoint")
    def test_real_qwen_gpu_matches_reference(self):
        self.assertTrue(torch.cuda.is_available())
        name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, attn_implementation="eager",
            local_files_only=True).to("cuda").eval()
        for prompt, budget in (
            ("What is the capital of France? Answer in one sentence.", 32),
            ("Count from 1 to 10.", 48),
            ("Explain a KV cache in one sentence.", 1),
        ):
            with self.subTest(prompt=prompt, budget=budget):
                actual = decode.generate_tokens(model, tokenizer, prompt, budget)
                expected = reference.generate_tokens(model, tokenizer, prompt, budget)
                self.assertTrue(torch.equal(actual, expected),
                                f"manual={actual.tolist()} reference={expected.tolist()}")


if __name__ == "__main__":
    unittest.main()

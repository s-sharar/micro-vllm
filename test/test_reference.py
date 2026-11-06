"""Offline unit, integration, and CLI tests; no model downloads required."""
import contextlib
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
from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1] / 'src' / 'micro_vllm'


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "reference.py").exists():
            return
        from micro_vllm import reference
        cls.reference = reference
        torch.manual_seed(7)
        backend = Tokenizer(WordLevel(
            {"<pad>": 0, "<unk>": 1, "user": 2, "assistant": 3, "hello": 4,
             "world": 5, "answer": 6, "yes": 7}, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
            model_input_names=["input_ids", "attention_mask"])
        cls.tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] + ' ' + "
            "message['content'] + ' ' }}{% endfor %}"
            "{% if add_generation_prompt %}assistant {% endif %}")
        cls.model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=8, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=128, eos_token_id=None,
            bos_token_id=None, pad_token_id=0, attention_dropout=0.0)).eval()
        cls.temp = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.temp.name)
        cls.model.save_pretrained(cls.checkpoint)
        cls.tokenizer.save_pretrained(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    def setUp(self):
        self.assertTrue((ROOT / "reference.py").exists(),
                        "reference.py must implement the baseline")

    def test_cli_defaults(self):
        args = self.reference.parse_args([])
        self.assertEqual(args.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.max_new_tokens, 64)

    def test_cli_rejects_invalid_input(self):
        for arguments in (["--prompt", "  "], ["--max-new-tokens", "0"],
                          ["--max-new-tokens", "-1"], ["--device", "tpu"]):
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        self.reference.parse_args(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_generation_returns_only_new_tokens_and_is_repeatable(self):
        first = self.reference.generate_tokens(
            self.model, self.tokenizer, "hello world", 3)
        second = self.reference.generate_tokens(
            self.model, self.tokenizer, "hello world", 3)
        self.assertEqual(first.shape, (3,))
        self.assertEqual(first.dtype, torch.long)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(first.requires_grad)

    def test_generation_stops_at_eos(self):
        first = self.reference.generate_tokens(
            self.model, self.tokenizer, "hello world", 1)
        stopping_model = Qwen2ForCausalLM(self.model.config).eval()
        stopping_model.load_state_dict(self.model.state_dict())
        stopping_model.generation_config.eos_token_id = first.item()
        tokens = self.reference.generate_tokens(
            stopping_model, self.tokenizer, "hello world", 5)
        self.assertEqual(tokens.tolist(), first.tolist())

    def test_main_runs_real_local_checkpoint(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.reference.main(["--model", str(self.checkpoint),
                                 "--device", "cpu", "--prompt", "hello",
                                 "--max-new-tokens", "2"])
        self.assertIn("Generated tokens: 2", output.getvalue())
        self.assertIn("Token IDs:", output.getvalue())
        self.assertIn("Response:", output.getvalue())

    def test_cuda_unavailable_has_actionable_error(self):
        with patch("micro_vllm.reference.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(SystemExit, "--device cpu"):
                self.reference.main(["--device", "cuda"])

    def test_cli_end_to_end_offline(self):
        result = subprocess.run(
            [sys.executable, '-m', 'micro_vllm.reference', "--model",
             str(self.checkpoint), "--device", "cpu", "--prompt", "hello",
             "--max-new-tokens", "2"],
            text=True, capture_output=True, timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1",
                 "TRANSFORMERS_OFFLINE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Generated tokens: 2", result.stdout)

    def test_checkpoint_settings_cannot_override_greedy_baseline(self):
        configured = Qwen2ForCausalLM(self.model.config).eval()
        configured.load_state_dict(self.model.state_dict())
        configured.generation_config = GenerationConfig(
            do_sample=True, repetition_penalty=1.2,
            return_dict_in_generate=True, pad_token_id=0)
        checkpoint = self.checkpoint / "sampling-defaults"
        configured.save_pretrained(checkpoint)
        self.tokenizer.save_pretrained(checkpoint)
        result = subprocess.run(
            [sys.executable, '-m', 'micro_vllm.reference', "--model",
             str(checkpoint), "--device", "cpu", "--prompt", "hello world",
             "--max-new-tokens", "3"],
            text=True, capture_output=True, timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = self.reference.generate_tokens(
            self.model, self.tokenizer, "hello world", 3).tolist()
        self.assertIn(f"Token IDs: {expected}", result.stdout)

    def test_help_does_not_load_weights(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as error:
                self.reference.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--max-new-tokens", output.getvalue())


if __name__ == "__main__":
    unittest.main()

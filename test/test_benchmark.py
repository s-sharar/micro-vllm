"""Benchmark behavior: greedy feedback, fixed work, and honest timing summaries."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from micro_vllm.kv_cache import ContiguousKVCache
from micro_vllm.model import prepare_model

try:
    from benchmarks import generation as benchmark
except ImportError:
    benchmark = None


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(benchmark, 'the reproducible benchmark module must exist')

    def test_report_defaults_to_separate_benchmark_results_folder(self):
        self.assertEqual(benchmark.parse_args([]).output.parent,
                         Path(benchmark.__file__).resolve().parent / 'results')

    def test_flash_mode_changes_only_cached_decode_and_restores_selector(self):
        from micro_vllm import runtime
        original = runtime.select_attention_backend
        prepared = SimpleNamespace(
            embedding_weight=SimpleNamespace(device=torch.device('cuda'), dtype=torch.float16),
            attention_backend='auto', head_dim=64)
        prefix = torch.tensor([[1, 2, 3]])
        with patch('torch.backends.cuda.is_flash_attention_available', return_value=True), \
                patch('torch.cuda.get_device_capability', return_value=(8, 9)):
            self.assertEqual(benchmark.resolved_routes(prepared, prefix),
                             dict(prefill='flash', decode='cuda'))
            with benchmark.benchmark_mode('flash') as policy:
                self.assertEqual(policy, 'auto')
                self.assertEqual(benchmark.resolved_routes(prepared, prefix),
                                 dict(prefill='flash', decode='flash'))
            self.assertIs(runtime.select_attention_backend, original)
            self.assertEqual(benchmark.resolved_routes(prepared, prefix),
                             dict(prefill='flash', decode='cuda'))

    def test_flash_selector_is_restored_after_failure(self):
        from micro_vllm import runtime
        original = runtime.select_attention_backend
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with benchmark.benchmark_mode('flash'):
                raise RuntimeError('deliberate')
        self.assertIs(runtime.select_attention_backend, original)
        with self.assertRaises(ValueError):
            with benchmark.benchmark_mode('unknown'):
                pass

    def test_cli_rejects_nonpositive_work(self):
        for argv in (['--contexts', '0'], ['--decode-steps', '0'], ['--repeats', '-1']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                benchmark.parse_args(argv)

    def test_prefix_is_exact_and_shared_across_context_lengths(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                self.text = text
                return [2, 7, 4]
        tokenizer = Tokenizer()
        short = benchmark.make_prefix(tokenizer, 4, 'cpu')
        long = benchmark.make_prefix(tokenizer, 11, 'cpu')
        self.assertEqual(short.tolist(), [[2, 7, 4, 2]])
        self.assertTrue(torch.equal(short, long[:, :4]))
        self.assertGreater(len(tokenizer.text.split()), 10)

    def test_timed_run_matches_uncached_greedy_and_uses_fixed_decode_work(self):
        config = Qwen2Config(vocab_size=13, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, max_position_embeddings=32,
                             eos_token_id=0)
        with torch.random.fork_rng():
            torch.manual_seed(31)
            model = Qwen2ForCausalLM(config).eval()
        prepared = prepare_model(model.state_dict(), config)
        prefix = torch.tensor([[1, 5, 8]])
        cache = ContiguousKVCache.allocate(
            num_layers=1, batch_size=1, num_kv_heads=1, capacity=7,
            head_dim=8, dtype=torch.float32, device='cpu')
        records = []

        class RecordingModel:
            def forward(self, ids, **kwargs):
                records.append((ids.clone(), kwargs['cache'].sequence_length,
                                kwargs['logits_to_keep']))
                return prepared.forward(ids, **kwargs)

        result = benchmark.timed_generation(RecordingModel(), prefix, cache, 4)
        expected, full = [], prefix.clone()
        with torch.inference_mode():
            for _ in range(5):
                token = prepared.forward(full, logits_to_keep=1)[:, -1:].argmax(-1)
                expected.append(token.item())
                full = torch.cat((full, token), dim=1)
        self.assertEqual(result['token_ids'], expected)
        self.assertEqual([r[0].shape[1] for r in records], [3, 1, 1, 1, 1])
        self.assertEqual([r[1] for r in records], [0, 3, 4, 5, 6])
        self.assertEqual(cache.sequence_length, 7)
        for index in range(1, 5):
            self.assertEqual(records[index][0].item(), expected[index - 1])
        self.assertTrue(all(record[2] == 1 for record in records))
        self.assertIsNone(result['prefill_event_ms'])
        self.assertGreaterEqual(result['total_wall_ms'], result['decode_wall_ms'])
        summary = benchmark.benchmark_context(prepared, prefix, config, 4, 1, torch.float32)
        self.assertEqual(summary['resolved_attention'], dict(prefill='torch', decode='torch'))

    def test_route_report_reflects_flash_eligibility_and_cached_decode(self):
        prepared = SimpleNamespace(
            embedding_weight=SimpleNamespace(device=torch.device('cuda'), dtype=torch.float16),
            attention_backend='auto', head_dim=4)
        prefix = torch.tensor([[1, 2, 3]])
        with patch('torch.backends.cuda.is_flash_attention_available', return_value=True), \
                patch('torch.cuda.get_device_capability', return_value=(9, 0)):
            fallback = benchmark.resolved_routes(prepared, prefix)
            self.assertEqual(fallback, dict(prefill='torch', decode='cuda'))
            prepared.head_dim = 64
            self.assertEqual(benchmark.resolved_routes(prepared, prefix),
                             dict(prefill='flash', decode='cuda'))
            prepared.attention_backend = 'torch'
            self.assertEqual(benchmark.resolved_routes(prepared, prefix),
                             dict(prefill='torch', decode='torch'))

    def test_comparison_rotates_order_and_resets_every_request_cache(self):
        config = Qwen2Config(vocab_size=13, hidden_size=16, intermediate_size=24,
                            num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=1, max_position_embeddings=16)
        with torch.random.fork_rng():
            torch.manual_seed(31)
            checkpoint = Qwen2ForCausalLM(config).eval()
        models = {name: prepare_model(checkpoint.state_dict(), config,
                                     attention_backend='torch' if name == 'torch' else 'auto')
                  for name in ('torch', 'auto', 'flash')}
        prefix = torch.tensor([[1, 5, 8]])
        modes, orders = benchmark.compare_context(models, prefix, config, 2, 3, torch.float32)
        self.assertEqual(orders, [['torch', 'auto', 'flash'],
                                  ['auto', 'flash', 'torch'], ['flash', 'torch', 'auto']])
        expected = modes['torch']['runs'][0]['token_ids']
        for mode in modes.values():
            self.assertEqual(mode['resolved_attention'], dict(prefill='torch', decode='torch'))
            self.assertEqual(len(mode['runs']), 3)
            for run in mode['runs']:
                self.assertEqual(run['token_ids'], expected)

    def test_summary_uses_medians_and_retains_divergent_tokens(self):
        runs = [
            dict(prefill_wall_ms=2., decode_wall_ms=8., total_wall_ms=10.,
                 prefill_event_ms=None, decode_event_ms=None, token_ids=[1, 2]),
            dict(prefill_wall_ms=8., decode_wall_ms=16., total_wall_ms=24.,
                 prefill_event_ms=None, decode_event_ms=None, token_ids=[1, 9]),
            dict(prefill_wall_ms=4., decode_wall_ms=12., total_wall_ms=16.,
                 prefill_event_ms=None, decode_event_ms=None, token_ids=[1, 2]),
        ]
        summary = benchmark.summarize_runs(runs, 4)
        self.assertEqual(summary['median']['prefill_wall_ms'], 4.)
        self.assertEqual(summary['median']['decode_wall_ms_per_token'], 3.)
        self.assertFalse(summary['repeat_token_parity'])
        self.assertEqual(summary['runs'][1]['token_ids'], [1, 9])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_event_smoke(self):
        class StepModel:
            def forward(self, ids, **kwargs):
                logits = torch.zeros((1, 1, 4), device=ids.device)
                logits[..., 2] = 1
                return logits
        result = benchmark.timed_generation(
            StepModel(), torch.tensor([[1, 2]], device='cuda'), None, 2)
        self.assertEqual(result['token_ids'], [2, 2, 2])
        self.assertGreater(result['prefill_event_ms'], 0)
        self.assertGreater(result['decode_event_ms'], 0)


if __name__ == '__main__':
    unittest.main()

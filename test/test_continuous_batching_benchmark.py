"""Continuous batching benchmark: shared timing boundaries, parity records and honest summaries."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from micro_vllm import generate
from micro_vllm.engine import Engine
from test.test_accelerated_generation import make_qwen, make_tokenizer

try:
    from benchmarks import continuous_batching as benchmark
except ImportError:
    benchmark = None


@unittest.skipIf(benchmark is None, 'benchmarks/ is gitignored and absent from this checkout')
class ContinuousBatchingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = make_qwen()
        cls.weights, cls.config = checkpoint.state_dict(), checkpoint.config
        cls.tokenizer = make_tokenizer()

    def test_cli_defaults_and_validation(self):
        args = benchmark.parse_args([])
        self.assertEqual(args.output.parent, Path(benchmark.__file__).resolve().parent / 'results')
        self.assertEqual((args.concurrency, args.attention_backend, args.page_size),
                         ([1, 2, 4, 8], 'auto', 16))
        for argv in (['--concurrency', '0'], ['--concurrency', '9'], ['--concurrency', '2', '2'],
                     ['--max-new-tokens', '0'], ['--repeats', '0'], ['--page-size', '-1'],
                     ['--attention-backend', 'flash']):
            with (self.subTest(argv=argv), self.assertRaises(SystemExit),
                  redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO())):
                benchmark.parse_args(argv)

    def test_pretokenized_ids_drive_generate_tokens_unchanged(self):
        texts = ('hello world', 'answer yes')
        pretokenized = benchmark.Pretokenized(self.tokenizer, texts)
        for text in texts:
            expected = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True)
            self.assertEqual(pretokenized.ids(text), expected)
            options = dict(eos_token_id=None, kv_cache='paged', page_size=2)
            self.assertTrue(torch.equal(
                generate.generate_tokens(self.weights, self.config, pretokenized, text, 4, **options),
                generate.generate_tokens(self.weights, self.config, self.tokenizer, text, 4,
                                         **options)))

    def test_pool_holds_every_request_at_once(self):
        self.assertEqual(benchmark.pages_needed([[1] * 5, [1] * 2], 4, 3), 3 + 2)

    def test_concurrency_run_compares_paths_with_matching_ids(self):
        texts = benchmark.PROMPTS[:3]
        with redirect_stdout(io.StringIO()):
            result = benchmark.benchmark_concurrency(
                self.weights, self.config, self.tokenizer, texts, max_new_tokens=3, repeats=2,
                attention_backend='auto', page_size=2)
        self.assertEqual(result['concurrency'], 3)
        self.assertTrue(result['token_ids_match'])
        self.assertEqual(result['measurement_orders'],
                         [['engine', 'sequential'], ['sequential', 'engine']])
        self.assertEqual(result['preemptions'], [[0, 0, 0]] * 2)
        self.assertEqual(result['first_divergent_token'],
                         {'engine': [[None] * 3] * 2, 'sequential': [[None] * 3] * 2})
        for name in ('engine', 'sequential'):
            summary = result[name]
            self.assertGreater(summary['tokens_per_s'], 0)
            self.assertEqual(len(summary['per_request_completion_ms_per_token']), 3)
            for run in result['runs'][name]:
                self.assertEqual([len(ids) for ids in run['token_ids']], [3] * 3)
                self.assertLessEqual(max(run['completion_ms']), run['wall_ms'])
        self.assertEqual(result['engine']['ticks'], 3)
        for run in result['runs']['sequential']:
            self.assertEqual(run['completion_ms'], sorted(run['completion_ms']))
            for service, completion in zip(run['service_ms'], run['completion_ms']):
                self.assertLessEqual(service, completion)
        expected = result['engine']['tokens_per_s'] / result['sequential']['tokens_per_s']
        self.assertAlmostEqual(result['throughput_speedup'], expected)
        self.assertEqual(result['divergence_margins'], [])

    def test_divergence_margins_compare_engine_and_standalone_logits(self):
        prompts = [[2, 4, 5, 3], [2, 6, 3]]
        engine = Engine(self.weights, self.config, num_pages=8, page_size=2)
        requests = [engine.submit(prompt, 4, None) for prompt in prompts]
        while engine.has_work:
            engine.step_batch()
        ids = [list(engine.result(request).tokens) for request in requests]
        altered = [list(tokens) for tokens in ids]
        altered[1][2] = (ids[1][2] + 1) % self.config.vocab_size
        (entry,) = benchmark.divergence_margins(
            self.weights, self.config, prompts, 4, [(1, 2)], altered, ids,
            attention_backend='auto', page_size=2)
        self.assertEqual((entry['request'], entry['position']), (1, 2))
        self.assertEqual((entry['engine_token'], entry['standalone_token']), (ids[1][2], altered[1][2]))
        self.assertTrue(entry['rerun_matches_timed_engine'])
        self.assertLess(entry['max_abs_logit_difference'], 1e-4)
        self.assertAlmostEqual(entry['engine_top2_margin'], entry['standalone_top2_margin'], places=4)
        self.assertGreater(entry['engine_top2_margin'], 0)

    def test_summary_uses_medians_over_repeats(self):
        runs = [dict(wall_ms=100., completion_ms=[50., 100.], ticks=4),
                dict(wall_ms=300., completion_ms=[150., 300.], ticks=6),
                dict(wall_ms=200., completion_ms=[100., 200.], ticks=5)]
        summary = benchmark.summarize(runs, concurrency=2, max_new_tokens=10)
        self.assertEqual(summary['wall_ms'], 200.)
        self.assertEqual(summary['tokens_per_s'], 100.)
        self.assertEqual(summary['mean_completion_ms_per_token'], 15.)
        self.assertEqual(summary['max_completion_ms_per_token'], 20.)
        self.assertEqual(summary['per_request_completion_ms_per_token'], [10., 20.])
        self.assertEqual(summary['ticks'], 5)
        self.assertNotIn('mean_service_ms_per_token', summary)

    def test_first_divergence_reports_index_or_none(self):
        self.assertIsNone(benchmark._first_divergence([1, 2], [1, 2]))
        self.assertEqual(benchmark._first_divergence([1, 2, 3], [1, 4, 3]), 1)
        self.assertEqual(benchmark._first_divergence([1, 2], [1, 2, 3]), 2)

    def test_engine_timing_rejects_incomplete_requests(self):
        engine = Engine(self.weights, self.config, num_pages=4, page_size=2,
                        attention_backend='cuda')
        with self.assertRaisesRegex(RuntimeError, 'complete'):
            benchmark.time_engine(engine, [[2, 3]], 2, torch.device('cpu'))

    def test_run_stops_at_the_first_concurrency_that_does_not_fit(self):
        args = benchmark.parse_args(['--concurrency', '4', '1', '8', '2'])
        attempted = []

        def run(weights, config, tokenizer, texts, **kwargs):
            attempted.append(len(texts))
            if len(texts) == 4:
                raise torch.cuda.OutOfMemoryError('no room')
            return dict(concurrency=len(texts))
        with (patch.object(benchmark, 'benchmark_concurrency', side_effect=run),
              patch.object(benchmark, '_metadata', return_value={})):
            report = benchmark.run_benchmark(args, self.weights, self.config, self.tokenizer)
        self.assertEqual(attempted, [1, 2, 4])
        self.assertEqual([result['concurrency'] for result in report['results']], [1, 2])
        self.assertEqual([entry['concurrency'] for entry in report['skipped']], [4])

    def test_main_requires_cuda_and_writes_the_report(self):
        with patch('torch.cuda.is_available', return_value=False), \
                self.assertRaisesRegex(SystemExit, 'CUDA'):
            benchmark.main([])
        checkpoint = SimpleNamespace(state_dict=lambda: self.weights, config=self.config)
        checkpoint.to = lambda device: checkpoint
        checkpoint.eval = lambda: checkpoint
        report = dict(results=[dict(concurrency=1, token_ids_match=True,
                                    engine=dict(tokens_per_s=2.), sequential=dict(tokens_per_s=1.))])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'nested' / 'report.json'
            with (patch('torch.cuda.is_available', return_value=True),
                  patch.object(benchmark.AutoTokenizer, 'from_pretrained',
                               return_value=self.tokenizer),
                  patch.object(benchmark.AutoModelForCausalLM, 'from_pretrained',
                               return_value=checkpoint),
                  patch.object(benchmark, 'run_benchmark', return_value=report) as run,
                  redirect_stdout(io.StringIO()) as printed):
                benchmark.main(['--output', str(output)])
            self.assertEqual(json.loads(output.read_text()), report)
        self.assertIs(run.call_args.args[1], self.weights)
        self.assertIn('engine 2.0 tok/s vs sequential 1.0 tok/s', printed.getvalue())


if __name__ == '__main__':
    unittest.main()

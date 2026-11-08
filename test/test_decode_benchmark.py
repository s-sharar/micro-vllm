"""Isolated decode benchmark safety, measurements, and optional GPU integration."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

MODULE = Path(__file__).resolve().parents[1] / 'benchmarks' / 'decode_attention.py'


class DecodeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), 'provide the isolated decode benchmark')
        self.bench = importlib.import_module('benchmarks.decode_attention')

    def inputs(self):
        return self.bench.make_inputs(7, query_heads=14, kv_heads=2, head_dim=8,
                                      spare_capacity=5, seed=37, device='cpu')

    def test_seeded_prefix_views_keep_original_kv_and_unused_nan_storage(self):
        inputs = self.inputs()
        other = self.inputs()
        self.assertEqual(tuple(inputs.query.shape), (1, 14, 1, 8))
        self.assertEqual(tuple(inputs.key.shape), (1, 2, 7, 8))
        self.assertEqual(inputs.key.stride(1), 12 * 8)
        self.assertFalse(inputs.key.is_contiguous())
        for name in ('query', 'key', 'value'):
            tensor = getattr(inputs, name)
            self.assertEqual(tensor.dtype, torch.float16)
            self.assertTrue(torch.isfinite(tensor).all())
            torch.testing.assert_close(tensor, getattr(other, name), rtol=0, atol=0)
        self.assertTrue(torch.isnan(inputs.key._base[:, :, 7:]).all())
        self.assertTrue(torch.isnan(inputs.value._base[:, :, 7:]).all())

    def test_backend_closures_receive_identical_objects_and_lower_right_offset(self):
        inputs = self.inputs()
        with patch.object(self.bench, 'decode_attention_gqa') as custom, \
                patch.object(self.bench, 'flash_attention') as flash, \
                patch.object(self.bench, 'causal_attention') as reference:
            functions = self.bench.make_backends(inputs, include_torch=True)
            for function in functions.values():
                function()
        for function in (custom, flash, reference):
            args = function.call_args.args
            for actual, expected in zip(args, (inputs.query, inputs.key, inputs.value)):
                self.assertIs(actual, expected)
        self.assertEqual(flash.call_args.kwargs, {'past_length': 6})
        self.assertEqual(reference.call_args.kwargs, {'past_length': 6})

    def test_incorrect_or_nonfinite_results_stop_before_profiling_and_timing(self):
        args = self.bench.parse_args(['--contexts', '7', '--head-dim', '8'])
        for result in (torch.ones(1, 14, 1, 8), torch.full((1, 14, 1, 8), float('nan'))):
            with patch.object(self.bench, 'make_inputs', return_value=self.inputs()), \
                    patch.object(self.bench, 'make_backends', return_value={'custom': lambda: result}), \
                    patch.object(self.bench, 'profile_backends') as profile, \
                    patch.object(self.bench, 'measure_all') as measure:
                with self.assertRaisesRegex(RuntimeError, 'correctness'):
                    self.bench.benchmark_case(args, 7, device='cpu')
                profile.assert_not_called()
                measure.assert_not_called()

    def test_correctness_records_max_abs_error(self):
        expected = torch.ones(1, 2, 1, 8, dtype=torch.float16)
        actual = expected + .001
        result = self.bench.check_outputs({'custom': lambda: actual}, expected)
        self.assertAlmostEqual(result['custom'], .0009765625)

    def test_flash_verification_requires_fused_operator_and_cuda_kernel(self):
        valid = {'operators': ['aten::_scaled_dot_product_flash_attention'],
                 'cuda_kernels': ['pytorch_flash::flash_fwd_kernel']}
        self.bench.verify_profile('flash', valid)
        for invalid in (
            {'operators': ['aten::_scaled_dot_product_attention_math'], 'cuda_kernels': ['gemm']},
            {'operators': valid['operators'], 'cuda_kernels': []},
            {'operators': valid['operators'] + ['aten::_scaled_dot_product_attention_math'],
             'cuda_kernels': valid['cuda_kernels']},
        ):
            with self.assertRaisesRegex(RuntimeError, 'Flash'):
                self.bench.verify_profile('flash', invalid)

    def test_custom_verification_requires_its_cuda_kernel(self):
        self.bench.verify_profile('custom', {'operators': [], 'cuda_kernels': ['decode_attention_online_kernel<c10::Half>']})
        with self.assertRaises(RuntimeError):
            self.bench.verify_profile('custom', {'operators': [], 'cuda_kernels': ['unrelated']})

    def test_cli_rejects_nonpositive_and_incompatible_geometry(self):
        invalid = [
            ['--contexts', '0'], ['--repeats', '-1'], ['--calls-per-sample', '0'],
            ['--graph-calls', '0'], ['--graph-replays', '-2'], ['--warmup', '0'],
            ['--spare-capacity', '0'], ['--query-heads', '15', '--kv-heads', '2'],
            ['--head-dim', '7'], ['--head-dim', '264'],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), patch('sys.stderr'), self.assertRaises(SystemExit):
                self.bench.parse_args(argv)
        args = self.bench.parse_args([])
        self.assertEqual(args.contexts, [32, 128, 512, 1024])
        self.assertEqual((args.query_heads, args.kv_heads, args.head_dim), (14, 2, 64))
        self.assertIn('benchmarks/results', str(args.output))

    def test_rotated_order_and_summary_retain_raw_timings(self):
        names = ['custom', 'flash', 'torch']
        self.assertEqual(self.bench.rotated_names(names, 1), ['flash', 'torch', 'custom'])
        samples = []
        for repeat in range(3):
            for backend, latency in [('custom', 2 + repeat), ('flash', 4 + 2 * repeat)]:
                samples.append(dict(repeat=repeat, backend=backend,
                                    wrapper_wall_us=latency, cuda_event_us=latency + 1,
                                    graph_replay_wall_us=latency / 2,
                                    wrapper_wall_ms=latency / 10,
                                    cuda_event_ms=(latency + 1) / 10,
                                    graph_replay_wall_ms=latency / 20))
        report = self.bench.summarize_samples(samples)
        self.assertIs(report['samples'], samples)
        self.assertEqual(report['timings']['custom']['wrapper_wall_us']['median'], 3)
        self.assertEqual(report['timings']['custom']['cuda_event_us']['samples'], [3, 4, 5])
        self.assertEqual(report['custom_speedup_vs_flash']['wrapper_wall_us'], 2)

    def test_cli_writes_complete_reviewable_report_with_raw_samples(self):
        metric = {'samples': [1.0, 3.0], 'median': 2.0}
        raw = [{'repeat': 0, 'backend': 'custom', 'wrapper_wall_ms': .2,
                'cuda_event_ms': .22, 'graph_replay_wall_ms': .1}]
        case = {'samples': raw, 'timings': {
            'custom': {'wrapper_wall_us': metric}, 'flash': {'wrapper_wall_us': metric}}}
        metadata = {'gpu': 'test GPU', 'dtype': 'torch.float16'}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results' / 'decode.json'
            with patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(self.bench, 'environment', return_value=metadata), \
                    patch.object(self.bench, 'benchmark_case', return_value=case) as run, \
                    patch('builtins.print'):
                report = self.bench.main(['--contexts', '32', '128', '--output', str(output)])
            saved = json.loads(output.read_text())
        self.assertEqual([call.args[1] for call in run.call_args_list], [32, 128])
        self.assertEqual(saved, report)
        self.assertEqual(saved['schema_version'], 1)
        self.assertEqual(saved['environment'], metadata)
        self.assertEqual(saved['config']['seed'], 932)
        self.assertEqual(saved['config']['output'], str(output))
        self.assertEqual(saved['cases'][0]['samples'], raw)
        self.assertEqual(saved['cases'][0]['timings']['custom']['wrapper_wall_us'], metric)
        self.assertIn('full attention operation', saved['measurement_notes']['graph_replay_wall_us'])

    def test_failed_benchmark_does_not_publish_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.json'
            with patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(self.bench, 'environment', return_value={}), \
                    patch.object(self.bench, 'benchmark_case',
                                 side_effect=RuntimeError('correctness check failed')), \
                    self.assertRaisesRegex(RuntimeError, 'correctness'):
                self.bench.main(['--output', str(output)])
            self.assertFalse(output.exists())

    def test_cli_without_cuda_fails_before_environment_queries(self):
        with patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(self.bench, 'environment') as metadata, \
                self.assertRaisesRegex(RuntimeError, 'requires a CUDA GPU'):
            self.bench.main([])
        metadata.assert_not_called()

    def test_environment_records_versions_and_handles_missing_driver_tool(self):
        with patch.object(self.bench.subprocess, 'run', side_effect=FileNotFoundError), \
                patch.object(torch.cuda, 'get_device_name', return_value='test GPU'), \
                patch.object(torch.cuda, 'get_device_capability', return_value=(8, 9)), \
                patch.object(torch.cuda, 'get_device_properties',
                             return_value=SimpleNamespace(total_memory=123456)):
            metadata = self.bench.environment()
        self.assertEqual(metadata['gpu'], 'test GPU')
        self.assertEqual(metadata['gpu_capability'], [8, 9])
        self.assertEqual(metadata['gpu_total_memory'], 123456)
        self.assertEqual(metadata['pytorch'], torch.__version__)
        self.assertEqual(metadata['cuda_build'], torch.version.cuda)
        self.assertIsNone(metadata['driver'])

    @unittest.skipUnless(os.environ.get('RUN_GPU_TESTS') == '1' and torch.cuda.is_available(),
                         'set RUN_GPU_TESTS=1 for the isolated GPU smoke')
    def test_gpu_smoke_profiles_fused_flash_and_measures_all_paths(self):
        # CUPTI can lose CUDA activities after earlier in-process GPU tests.
        # Exercise the actual standalone CLI in a clean profiler process.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'decode-smoke.json'
            command = [
                sys.executable, '-m', 'benchmarks.decode_attention',
                '--contexts', '32', '--repeats', '2', '--calls-per-sample', '2',
                '--warmup', '2', '--graph-calls', '2', '--graph-replays', '2',
                '--include-torch', '--output', str(output),
            ]
            result = subprocess.run(command, cwd=MODULE.parents[1], capture_output=True,
                                    text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
        self.assertEqual(report['environment']['dtype'], 'torch.float16')
        case = report['cases'][0]
        self.bench.verify_profile('custom', case['profiles']['custom'])
        self.bench.verify_profile('flash', case['profiles']['flash'])
        self.assertEqual(len(case['samples']), 6)
        for name in ('custom', 'flash', 'torch'):
            self.assertLess(case['max_abs_error'][name], .003)
            self.assertTrue(case['profiles'][name]['cuda_kernels'])
            for metric in ('wrapper_wall_us', 'cuda_event_us', 'graph_replay_wall_us'):
                self.assertGreater(case['timings'][name][metric]['median'], 0)
        self.assertEqual(case['samples'][0]['order'], ['custom', 'flash', 'torch'])
        self.assertEqual(case['samples'][3]['order'], ['flash', 'torch', 'custom'])


if __name__ == '__main__':
    unittest.main()

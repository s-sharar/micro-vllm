"""GPU reduction checks: tails, repeated per-thread loads, precision, and streams."""
import importlib
import os
import subprocess
import sys
import tempfile
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class ReduceSumTests(unittest.TestCase):
    def setUp(self):
        self.ops = importlib.import_module('micro_vllm.cuda_ops')
        self.assertTrue(hasattr(self.ops, 'reduce_sum'),
                        'provide the public CUDA reduce_sum operation')

    def test_exact_sums_across_warp_block_and_load_boundaries(self):
        with torch.inference_mode():
            for size in (0, 1, 31, 32, 33, 255, 256, 257, 511, 512, 513, 100003):
                with self.subTest(size=size):
                    values = torch.ones(size, device='cuda', dtype=torch.float32)
                    actual = self.ops.reduce_sum(values)
                    self.assertEqual(actual.shape, torch.Size([]))
                    self.assertEqual(actual.device, values.device)
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, values.sum(), rtol=0, atol=0)

    def test_random_sums_match_fp64_reference_without_mutating_input(self):
        generator = torch.Generator(device='cuda').manual_seed(53)
        with torch.inference_mode():
            for size in (17, 257, 4099, 100003):
                with self.subTest(size=size):
                    values = torch.randn(size, device='cuda', generator=generator)
                    original = values.clone()
                    actual = self.ops.reduce_sum(values)
                    reference = values.sum(dtype=torch.float64).float()
                    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-3)
                    torch.testing.assert_close(values, original, rtol=0, atol=0)
                    self.assertNotEqual(actual.data_ptr(), values.data_ptr())

    def test_cancellation_uses_absolute_tolerance_near_zero(self):
        generator = torch.Generator(device='cuda').manual_seed(59)
        values = torch.randn(4097, device='cuda', generator=generator)
        inputs = torch.cat((values, -values.roll(17)))
        reference = inputs.sum(dtype=torch.float64).float()
        torch.testing.assert_close(self.ops.reduce_sum(inputs), reference, rtol=0, atol=1e-3)

    def test_offsets_signed_values_and_repeated_outputs(self):
        with torch.inference_mode():
            values = torch.arange(-300, 300, device='cuda', dtype=torch.float32)[13:528]
            self.assertTrue(values.is_contiguous())
            first = self.ops.reduce_sum(values)
            before = first.clone()
            torch.testing.assert_close(first, values.sum(), rtol=0, atol=0)
            for value in (0., 2., -3.):
                current = torch.full((257,), value, device='cuda')
                actual = self.ops.reduce_sum(current)
                torch.testing.assert_close(actual, current.sum(), rtol=0, atol=0)
                self.assertNotEqual(actual.data_ptr(), first.data_ptr())
            torch.testing.assert_close(first, before, rtol=0, atol=0)

    def test_nonfinite_values_propagate(self):
        for values in ([1., float('nan')], [1., float('inf')],
                       [float('inf'), float('-inf')]):
            with self.subTest(values=values):
                inputs = torch.tensor(values, device='cuda')
                torch.testing.assert_close(self.ops.reduce_sum(inputs), inputs.sum(),
                                           rtol=0, atol=0, equal_nan=True)

    def test_current_stream_orders_writes_reduction_and_consumption(self):
        values = torch.zeros(513, device='cuda')
        self.ops.reduce_sum(values)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.inference_mode():
            torch.cuda._sleep(10000000)
            values.fill_(2)
            actual = self.ops.reduce_sum(values) * 3
        stream.synchronize()
        torch.testing.assert_close(actual, torch.tensor(3078., device='cuda'), rtol=0, atol=0)

    def test_rejects_unsupported_inputs(self):
        inputs = torch.ones(4, device='cuda')
        cases = [
            (inputs.cpu(), 'CUDA'),
            (inputs.half(), 'float32'),
            (inputs.double(), 'float32'),
            (inputs.long(), 'float32'),
            (inputs.reshape(2, 2), 'one-dimensional'),
            (inputs[:1].reshape(()), 'one-dimensional'),
            (torch.ones(8, device='cuda')[::2], 'contiguous'),
            (inputs.to_sparse(), 'strided'),
            (torch._neg_view(inputs), 'negative views'),
            (inputs.detach().requires_grad_(), 'gradients'),
        ]
        for value, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.ops.reduce_sum(value)

    def test_public_api_runs_in_an_independent_process(self):
        script = (
            'import torch; from micro_vllm.cuda_ops import reduce_sum, vector_add; '
            'a=torch.tensor([1., -2., 4.], device="cuda"); '
            'torch.testing.assert_close(reduce_sum(a), a.sum(), rtol=0, atol=0); '
            'torch.testing.assert_close(vector_add(a,a), a+a, rtol=0, atol=0); '
            'print("REDUCTION_API_OK")'
        )
        with tempfile.TemporaryDirectory(prefix='micro-vllm-reduction-') as directory:
            result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                    text=True, env=os.environ.copy(), cwd=directory, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('REDUCTION_API_OK', result.stdout)

    @unittest.skipUnless(torch.cuda.device_count() > 1, 'two CUDA GPUs required')
    def test_device_guard_restores_original_device(self):
        with torch.cuda.device(0):
            values = torch.ones(257, device='cuda:1')
            torch.testing.assert_close(self.ops.reduce_sum(values), values.sum(), rtol=0, atol=0)
            self.assertEqual(torch.cuda.current_device(), 0)


if __name__ == '__main__':
    unittest.main()

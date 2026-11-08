"""CUDA dot products: attention-sized vectors, tails, precision, and stream ordering."""
import importlib
import os
import subprocess
import sys
import tempfile
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class DotProductTests(unittest.TestCase):
    def setUp(self):
        self.ops = importlib.import_module('micro_vllm.cuda_ops')
        self.assertTrue(hasattr(self.ops, 'dot_product'),
                        'provide the public CUDA dot_product operation')

    def test_known_products_and_head_dimension_scaling(self):
        a = torch.tensor([1., 2., 3.], device='cuda')
        b = torch.tensor([4., 5., 6.], device='cuda')
        actual = self.ops.dot_product(a, b)
        self.assertEqual(actual.shape, torch.Size([]))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device, a.device)
        torch.testing.assert_close(actual, torch.tensor(32., device='cuda'), rtol=0, atol=0)
        head = torch.ones(64, device='cuda')
        scaled = self.ops.dot_product(head, head) / (64 ** 0.5)
        torch.testing.assert_close(scaled, torch.tensor(8., device='cuda'), rtol=0, atol=0)

    def test_exact_products_across_warp_head_block_and_load_boundaries(self):
        for size in (0, 1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 513, 100003):
            with self.subTest(size=size):
                a = torch.ones(size, device='cuda')
                b = torch.full_like(a, 2)
                torch.testing.assert_close(self.ops.dot_product(a, b), torch.dot(a, b),
                                           rtol=0, atol=0)

    def test_random_products_match_fp64_reference_and_preserve_inputs(self):
        generator = torch.Generator(device='cuda').manual_seed(61)
        for size in (17, 64, 257, 4099, 100003):
            with self.subTest(size=size):
                a = torch.randn(size, device='cuda', generator=generator)
                b = torch.randn(size, device='cuda', generator=generator)
                before_a, before_b = a.clone(), b.clone()
                actual = self.ops.dot_product(a, b)
                reference = (a.double() * b.double()).sum().float()
                torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-3)
                torch.testing.assert_close(a, before_a, rtol=0, atol=0)
                torch.testing.assert_close(b, before_b, rtol=0, atol=0)
                self.assertNotEqual(actual.data_ptr(), a.data_ptr())
                self.assertNotEqual(actual.data_ptr(), b.data_ptr())

    def test_storage_offsets_aliasing_and_output_independence(self):
        a = torch.arange(-300, 300, device='cuda', dtype=torch.float32)[13:528]
        b = torch.arange(700, device='cuda', dtype=torch.float32)[17:532]
        first = self.ops.dot_product(a, b)
        torch.testing.assert_close(first, torch.dot(a, b), rtol=1e-5, atol=1e-3)
        before = first.clone()
        second = self.ops.dot_product(a, a)
        torch.testing.assert_close(second, torch.dot(a, a), rtol=1e-5, atol=1e-3)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, before, rtol=0, atol=0)

    def test_cancellation_and_nonfinite_products(self):
        generator = torch.Generator(device='cuda').manual_seed(67)
        values = torch.randn(4097, device='cuda', generator=generator)
        a = torch.cat((values, -values.roll(17)))
        b = torch.full_like(a, 3)
        reference = (a.double() * b.double()).sum().float()
        torch.testing.assert_close(self.ops.dot_product(a, b), reference, rtol=0, atol=1e-3)
        for left, right in [([1., float('nan')], [2., 3.]),
                            ([1., float('inf')], [2., 3.]),
                            ([float('inf')], [0.])]:
            with self.subTest(left=left, right=right):
                a, b = torch.tensor(left, device='cuda'), torch.tensor(right, device='cuda')
                torch.testing.assert_close(self.ops.dot_product(a, b), torch.dot(a, b),
                                           rtol=0, atol=0, equal_nan=True)

    def test_current_stream_orders_producers_dot_and_consumer(self):
        a, b = torch.zeros(513, device='cuda'), torch.zeros(513, device='cuda')
        self.ops.dot_product(a, b)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10000000)
            a.fill_(2)
            b.fill_(3)
            actual = self.ops.dot_product(a, b) * 2
        stream.synchronize()
        torch.testing.assert_close(actual, torch.tensor(6156., device='cuda'), rtol=0, atol=0)

    def test_rejects_unsupported_inputs_in_each_argument(self):
        valid = torch.ones(4, device='cuda')
        cases = [(valid.cpu(), 'CUDA'), (valid.half(), 'float32'),
                 (valid.double(), 'float32'), (valid.long(), 'float32'),
                 (valid.reshape(2, 2), 'one-dimensional'),
                 (valid[:1].reshape(()), 'one-dimensional'),
                 (valid[:3], 'same shape'),
                 (torch.ones(8, device='cuda')[::2], 'contiguous'),
                 (valid.to_sparse(), 'strided'), (torch._neg_view(valid), 'negative views'),
                 (valid.detach().requires_grad_(), 'gradients')]
        for invalid, message in cases:
            for left, right in ((invalid, valid), (valid, invalid)):
                with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                    self.ops.dot_product(left, right)

    def test_public_apis_coexist_in_an_independent_process(self):
        script = (
            'import torch; from micro_vllm.cuda_ops import dot_product, reduce_sum, vector_add; '
            'a=torch.tensor([1., -2., 4.], device="cuda"); '
            'torch.testing.assert_close(dot_product(a,a), torch.dot(a,a), rtol=0, atol=0); '
            'torch.testing.assert_close(reduce_sum(a), a.sum(), rtol=0, atol=0); '
            'torch.testing.assert_close(vector_add(a,a), a+a, rtol=0, atol=0); '
            'print("DOT_PRODUCT_API_OK")'
        )
        with tempfile.TemporaryDirectory(prefix='micro-vllm-dot-') as directory:
            result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                    text=True, env=os.environ.copy(), cwd=directory, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('DOT_PRODUCT_API_OK', result.stdout)

    @unittest.skipUnless(torch.cuda.device_count() > 1, 'two CUDA GPUs required')
    def test_device_guard_and_mismatched_devices(self):
        with torch.cuda.device(0):
            a = torch.ones(64, device='cuda:1')
            torch.testing.assert_close(self.ops.dot_product(a, a), torch.dot(a, a), rtol=0, atol=0)
            self.assertEqual(torch.cuda.current_device(), 0)
            with self.assertRaisesRegex(RuntimeError, 'same device'):
                self.ops.dot_product(a, a.to('cuda:0'))


if __name__ == '__main__':
    unittest.main()

"""Public-API checks for the introductory Python/C++/CUDA vector addition."""
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import torch


class CudaImportTests(unittest.TestCase):
    def test_import_does_not_build_or_require_a_cuda_toolkit(self):
        script = (
            'import sys; import micro_vllm.cuda_ops; '
            'assert "_micro_vllm_cuda" not in sys.modules; '
            'print("CUDA_IMPORT_OK")'
        )
        with tempfile.TemporaryDirectory(prefix='micro-vllm-import-') as directory:
            cache = os.path.join(directory, 'extension-cache')
            env = dict(os.environ, CUDA_HOME=os.path.join(directory, 'no-toolkit'),
                       TORCH_EXTENSIONS_DIR=cache)
            result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                    text=True, env=env, cwd=directory, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('CUDA_IMPORT_OK', result.stdout)
            self.assertFalse(os.path.exists(cache), 'import must not start a native build')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class VectorAddTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('micro_vllm.cuda_ops'),
                             'provide the public CUDA vector_add operation')
        self.ops = importlib.import_module('micro_vllm.cuda_ops')

    def test_add_matches_pytorch_at_block_boundaries_without_mutating_inputs(self):
        with torch.inference_mode():
            for size in (0, 1, 255, 256, 257, 100003):
                with self.subTest(size=size):
                    a = torch.randn(size, device='cuda', dtype=torch.float32)
                    b = torch.randn_like(a)
                    before_a, before_b = a.clone(), b.clone()
                    actual = self.ops.vector_add(a, b)
                    self.assertEqual(actual.shape, a.shape)
                    self.assertEqual(actual.device, a.device)
                    self.assertEqual(actual.dtype, a.dtype)
                    torch.testing.assert_close(actual, a + b, rtol=0, atol=0)
                    torch.testing.assert_close(a, before_a, rtol=0, atol=0)
                    torch.testing.assert_close(b, before_b, rtol=0, atol=0)
                    if size:
                        self.assertNotEqual(actual.data_ptr(), a.data_ptr())
                        self.assertNotEqual(actual.data_ptr(), b.data_ptr())

    def test_contiguous_slices_with_storage_offsets_and_repeated_calls(self):
        with torch.inference_mode():
            a = torch.arange(300, device='cuda', dtype=torch.float32)[3:260]
            b = torch.arange(400, device='cuda', dtype=torch.float32)[11:268]
            self.assertTrue(a.is_contiguous() and b.is_contiguous())
            for _ in range(3):
                torch.testing.assert_close(self.ops.vector_add(a, b), a + b,
                                           rtol=0, atol=0)

    def test_launch_respects_current_cuda_stream(self):
        a, b = torch.zeros(257, device='cuda'), torch.zeros(257, device='cuda')
        self.ops.vector_add(a, b)  # Finish compilation before testing ordering.
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.inference_mode():
            torch.cuda._sleep(10000000)
            a.fill_(2)
            b.fill_(3)
            actual = self.ops.vector_add(a, b)
            consumed = actual * 2
        stream.synchronize()
        torch.testing.assert_close(consumed, torch.full_like(a, 10), rtol=0, atol=0)

    def test_rejects_unsupported_inputs_before_launch(self):
        a = torch.ones(4, device='cuda')
        cases = [
            (a.cpu(), a.cpu(), 'CUDA'),
            (a, a.cpu(), 'CUDA'),
            (a.half(), a.half(), 'float32'),
            (a, a.half(), 'float32'),
            (a.half(), a, 'float32'),
            (a.double(), a.double(), 'float32'),
            (a.long(), a.long(), 'float32'),
            (a.reshape(2, 2), a.reshape(2, 2), 'one-dimensional'),
            (a, a[:3], 'same shape'),
            (torch.ones(8, device='cuda')[::2], a, 'contiguous'),
            (a.to_sparse(), a.to_sparse(), 'strided'),
            (a.to_sparse(), a, 'strided'),
            (a, a.to_sparse(), 'strided'),
            (torch._neg_view(a), a, 'negative views'),
            (a, torch._neg_view(a), 'negative views'),
            (a.detach().requires_grad_(), a, 'gradients'),
            (a, a.detach().requires_grad_(), 'gradients'),
        ]
        for left, right, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.ops.vector_add(left, right)

    def test_python_api_runs_in_a_fresh_process(self):
        script = (
            'import torch; from micro_vllm.cuda_ops import vector_add; '
            'a=torch.tensor([1., -2., 3.], device="cuda"); '
            'b=torch.tensor([4., 5., -6.], device="cuda"); '
            'torch.testing.assert_close(vector_add(a,b), a+b, rtol=0, atol=0); '
            'print("CUDA_API_OK")'
        )
        with tempfile.TemporaryDirectory(prefix='micro-vllm-api-') as directory:
            result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                    text=True, env=os.environ.copy(), cwd=directory, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('CUDA_API_OK', result.stdout)

    @unittest.skipUnless(torch.cuda.device_count() > 1, 'two CUDA GPUs required')
    def test_device_guard_and_mismatched_devices(self):
        with torch.cuda.device(0):
            a = torch.ones(4, device='cuda:1')
            b = torch.ones_like(a)
            torch.testing.assert_close(self.ops.vector_add(a, b), a + b, rtol=0, atol=0)
            self.assertEqual(torch.cuda.current_device(), 0)
            with self.assertRaisesRegex(RuntimeError, 'same device'):
                self.ops.vector_add(a, b.to('cuda:0'))


if __name__ == '__main__':
    unittest.main()

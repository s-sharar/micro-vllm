"""Online attention: existing contract plus chunk-state, stability, and memory checks."""
import importlib
import os
import subprocess
import sys
import tempfile

import torch

from test import test_cuda_attention as baseline


class OnlineAttentionTests(baseline.SingleHeadAttentionTests):
    def setUp(self):
        super().setUp()
        self.ops = importlib.import_module('micro_vllm.cuda_attention')
        self.assertTrue(hasattr(self.ops, 'decode_attention_online'),
                        'provide the online-softmax CUDA attention operation')
        self.attention = self.ops.decode_attention_online

    def test_chunk_boundaries_match_baseline_and_fp64(self):
        generator = torch.Generator(device='cuda').manual_seed(79)
        for length in (31, 32, 33, 63, 64, 65, 95, 96, 97, 1023, 1024, 1025):
            for dimension in (33, 64, 257):
                with self.subTest(length=length, dimension=dimension):
                    query = torch.randn(dimension, device='cuda', generator=generator)
                    key = torch.randn(length, dimension, device='cuda', generator=generator)
                    value = torch.randn(length, dimension, device='cuda', generator=generator)
                    actual = self.attention(query, key, value)
                    reference = baseline.reference_attention(query, key, value)
                    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
                    torch.testing.assert_close(actual, self.ops.decode_attention(query, key, value),
                                               rtol=2e-4, atol=2e-5)

    def test_maximum_changes_across_chunks_and_underflow_rescaling(self):
        generator = torch.Generator(device='cuda').manual_seed(83)
        query = torch.zeros(64, device='cuda')
        query[0] = 8  # q dot K / sqrt(64) equals K's first feature.
        value = torch.randn(97, 64, device='cuda', generator=generator)
        cases = {
            'rising': torch.arange(97, device='cuda', dtype=torch.float32) * 2,
            'falling': -torch.arange(97, device='cuda', dtype=torch.float32) * 2,
            'alternating': torch.tensor([0.] * 32 + [80.] * 32 + [-80.] * 32 + [81.], device='cuda'),
            'last_dominates': torch.tensor([0.] * 96 + [1000.], device='cuda'),
            'first_dominates': torch.tensor([1000.] + [0.] * 96, device='cuda'),
            'negative_scores': torch.arange(97, device='cuda', dtype=torch.float32) - 10000,
        }
        for name, scores in cases.items():
            with self.subTest(case=name):
                key = torch.zeros_like(value)
                key[:, 0] = scores
                actual = self.attention(query, key, value)
                torch.testing.assert_close(actual, baseline.reference_attention(query, key, value),
                                           rtol=2e-4, atol=2e-5)
                self.assertTrue(torch.isfinite(actual).all())

    def test_partial_chunk_has_its_actual_probability_mass(self):
        query = torch.zeros(64, device='cuda')
        key, value = torch.zeros(65, 64, device='cuda'), torch.ones(65, 64, device='cuda')
        value[32:64].fill_(3)
        value[-1].fill_(100)
        expected = torch.full_like(query, 228 / 65)
        torch.testing.assert_close(self.attention(query, key, value), expected, rtol=1e-6, atol=1e-6)

    def test_large_constant_values_remain_finite_across_chunks(self):
        query = torch.zeros(1, device='cuda')
        for length in (33, 65, 129):
            key = torch.zeros(length, 1, device='cuda')
            for sign in (1., -1.):
                with self.subTest(length=length, sign=sign):
                    value = torch.full_like(key, sign * 2e38)
                    actual = self.attention(query, key, value)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, baseline.reference_attention(query, key, value),
                                               rtol=5e-6, atol=0)
        query = torch.ones(64, device='cuda')
        key = torch.arange(65, device='cuda', dtype=torch.float32)[:, None].expand(65, 64).contiguous() / 64
        value = torch.full_like(key, 2e38)
        actual = self.attention(query, key, value)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, baseline.reference_attention(query, key, value), rtol=5e-6, atol=0)

    def test_no_context_length_score_allocation(self):
        query = torch.zeros(64, device='cuda')
        for length in (8193, 16385):
            key = torch.zeros(length, 64, device='cuda')
            value = torch.ones_like(key)
            self.attention(query, key, value)  # Warm up compilation and allocator first.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            initial = torch.cuda.memory_allocated()
            actual = self.attention(query, key, value)
            torch.cuda.synchronize()
            extra_peak = torch.cuda.max_memory_allocated() - initial
            self.assertLess(extra_peak, length * query.element_size(),
                            'online attention must not allocate an L-element score buffer')
            torch.testing.assert_close(actual, torch.ones_like(query), rtol=2e-5, atol=2e-5)

    def test_public_api_in_an_independent_process(self):
        script = (
            'import sys,torch; from micro_vllm.cuda_attention import decode_attention_online,decode_attention; '
            'q=torch.zeros(64,device="cuda"); '
            'k=torch.ones(65,64,device="cuda"); v=torch.ones_like(k)*2; '
            'actual=decode_attention_online(q,k,v); '
            'torch.testing.assert_close(actual, torch.ones_like(q)*2); '
            'torch.testing.assert_close(actual, decode_attention(q,k,v)); '
            'assert "micro_vllm.cuda_ops" not in sys.modules; '
            'assert "_micro_vllm_cuda" not in sys.modules; '
            'print("ONLINE_ATTENTION_OK")'
        )
        with tempfile.TemporaryDirectory(prefix='micro-vllm-online-') as directory:
            result = subprocess.run([sys.executable, '-c', script], cwd=directory,
                                    env=os.environ.copy(), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ONLINE_ATTENTION_OK', result.stdout)

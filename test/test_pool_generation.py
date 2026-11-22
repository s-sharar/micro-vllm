"""Generation over a caller-owned shared KV pool with dynamic per-request page tables."""
import os
import unittest
from unittest.mock import patch

import torch

from micro_vllm import cuda_attention, generate, model, reference
from micro_vllm.kv_cache import ContiguousKVCache
from micro_vllm.kv_pool import KVPool, KVPoolExhausted, PageState
from micro_vllm.paged_kv_cache import PagedKVCache
from test import test_generate as fixtures
from test.test_accelerated_generation import make_qwen, make_tokenizer

GPU_TESTS = os.environ.get('RUN_GPU_TESTS') == '1' and torch.cuda.is_available()


def pool_for(weights, config, *, num_pages, page_size, **overrides):
    embedding = weights['model.embed_tokens.weight']
    options = dict(num_layers=config.num_hidden_layers,
                   num_kv_heads=config.num_key_value_heads,
                   head_dim=config.hidden_size // config.num_attention_heads,
                   dtype=embedding.dtype, device=embedding.device,
                   num_pages=num_pages, page_size=page_size)
    options.update(overrides)
    return KVPool.allocate(**options)


class PoolGenerationMixin:
    def generate(self, prompt='hello world', budget=4, **kwargs):
        kwargs.setdefault('kv_cache', 'paged')
        return generate.generate_tokens(self.weights, self.config, self.tokenizer, prompt,
                                        budget, eos_token_id=None, **kwargs)

    def tracked(self, records):
        real_forward = generate.forward

        def capture(ids, prepared, **kwargs):
            cache = kwargs['cache']
            records.append((cache, None))
            result = real_forward(ids, prepared, **kwargs)
            records[-1] = (cache, cache.page_table)
            return result
        return patch('micro_vllm.generate.forward', side_effect=capture)

    def states(self, pool):
        return [pool.state(page) for page in range(pool.num_pages)]

    def settle(self, pool):
        if pool.device.type == 'cuda':
            torch.cuda.synchronize(pool.device)
        pool.reclaim()

    def assert_sequential_requests_reuse_poisoned_pages(self, pool, page_size, **options):
        expected = self.generate(kv_cache='contiguous', **options)
        first, second = [], []
        with self.tracked(first):
            self.assertTrue(torch.equal(self.generate(kv_pool=pool, page_size=page_size,
                                                      **options), expected))
        first_pages = first[-1][1]
        self.assertEqual(len(first_pages), pool.num_pages)
        self.assertEqual(first[-1][0].page_table, ())
        self.settle(pool)
        self.assertEqual(self.states(pool), [PageState.FREE] * pool.num_pages)
        for buffer in pool._keys + pool._values:
            buffer[list(first_pages)] = float('nan')
        with self.tracked(second):
            actual = self.generate(kv_pool=pool, page_size=page_size, **options)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(second[-1][1], first_pages)
        self.assertIsNot(second[-1][0], first[-1][0])
        self.assertIs(second[-1][0]._keys, pool._keys)
        self.settle(pool)
        self.assertEqual(self.states(pool), [PageState.FREE] * pool.num_pages)

    def assert_mid_generation_exhaustion_releases_request(self, page_size, num_pages):
        pool = pool_for(self.weights, self.config, num_pages=num_pages, page_size=page_size)
        blocker = PagedKVCache.for_pool(pool, max_tokens=page_size).reserve(1)
        records = []
        with (self.tracked(records),
              patch.object(PagedKVCache, 'append', autospec=True,
                           side_effect=PagedKVCache.append) as append,
              patch.object(PagedKVCache, 'release', autospec=True,
                           side_effect=PagedKVCache.release) as release,
              self.assertRaises(KVPoolExhausted)):
            self.generate(kv_pool=pool, page_size=page_size)
        request = records[-1][0]
        self.assertEqual(len(records), 4)
        self.assertIsNone(records[-1][1])
        self.assertEqual(append.call_count, 3 * self.config.num_hidden_layers)
        release.assert_called_once_with(request)
        self.assertEqual(request.page_table, ())
        self.settle(pool)
        self.assertEqual(self.states(pool),
                         [PageState.OWNED] + [PageState.FREE] * (num_pages - 1))
        self.assertEqual(blocker.page_table, (0,))

    def assert_injected_failure_releases_pages(self, pool, page_size):
        real_layer, calls = model.decoder_layer, []

        def fail_second_forward(*args, **kwargs):
            calls.append(kwargs['layer_idx'])
            if len(calls) == self.config.num_hidden_layers + 1:
                raise RuntimeError('injected layer failure')
            return real_layer(*args, **kwargs)

        with (patch('micro_vllm.model.decoder_layer', side_effect=fail_second_forward),
              self.assertRaisesRegex(RuntimeError, 'injected')):
            self.generate(kv_pool=pool, page_size=page_size)
        self.settle(pool)
        self.assertEqual(self.states(pool), [PageState.FREE] * pool.num_pages)
        expected = self.generate(kv_cache='contiguous')
        self.assertTrue(torch.equal(self.generate(kv_pool=pool, page_size=page_size), expected))


class PoolGenerationCPUTests(PoolGenerationMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.GenerationTests.setUpClass.__func__(cls)
        cls.config = cls.model.config

    @classmethod
    def tearDownClass(cls):
        fixtures.GenerationTests.tearDownClass.__func__(cls)

    def test_supplied_pool_matches_reference_and_contiguous_tokens(self):
        pool = pool_for(self.weights, self.config, num_pages=12, page_size=3)
        for prompt, budget in (('hello', 1), ('hello world', 2), ('hello world answer yes', 6)):
            with self.subTest(prompt=prompt, budget=budget):
                expected = reference.generate_tokens(self.model, self.tokenizer, prompt, budget)
                contiguous = self.generate(prompt, budget, kv_cache='contiguous')
                actual = self.generate(prompt, budget, kv_pool=pool, page_size=3)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(actual, contiguous))
                self.assertEqual(self.states(pool), [PageState.FREE] * 12)

    def test_sequential_requests_reuse_poisoned_pages(self):
        pool = pool_for(self.weights, self.config, num_pages=3, page_size=3)
        self.assert_sequential_requests_reuse_poisoned_pages(pool, 3)

    def test_omitted_pool_is_sized_like_the_previous_fixed_allocation(self):
        records = []
        with self.tracked(records), patch.object(
                KVPool, 'allocate', wraps=KVPool.allocate) as allocate:
            self.generate(budget=4, page_size=3)
        self.assertEqual(allocate.call_args.kwargs['num_pages'], 3)
        self.assertEqual(records[-1][1], (0, 1, 2))
        self.assertEqual([table for _, table in records], [(0, 1), (0, 1), (0, 1), (0, 1, 2)])

    def test_request_larger_than_pool_is_rejected_before_running(self):
        expected = self.generate(kv_cache='contiguous')
        for page_size, num_pages, fits in ((2, 3, False), (1, 6, False), (1, 7, True),
                                           (2, 4, True)):
            pool = pool_for(self.weights, self.config, num_pages=num_pages, page_size=page_size)
            with (self.subTest(page_size=page_size, num_pages=num_pages),
                  patch('micro_vllm.generate.prepare_model',
                        side_effect=generate.prepare_model) as prepare,
                  patch('micro_vllm.generate.forward', side_effect=generate.forward) as forward):
                if fits:
                    actual = self.generate(kv_pool=pool, page_size=page_size)
                    self.assertTrue(torch.equal(actual, expected))
                else:
                    with self.assertRaisesRegex(ValueError, 'pool'):
                        self.generate(kv_pool=pool, page_size=page_size)
                    prepare.assert_not_called()
                    forward.assert_not_called()
                self.assertEqual(self.states(pool), [PageState.FREE] * num_pages)

    def test_mid_generation_exhaustion_releases_pages_and_raises(self):
        self.assert_mid_generation_exhaustion_releases_request(2, 4)

    def test_injected_failure_releases_pages_through_cleanup(self):
        pool = pool_for(self.weights, self.config, num_pages=4, page_size=2)
        self.assert_injected_failure_releases_pages(pool, 2)

    def test_invalid_pool_fails_before_tokenization(self):
        cases = [({'kv_pool': object()}, 'kv_pool'),
                 ({'kv_pool': pool_for(self.weights, self.config, num_pages=4, page_size=16),
                   'kv_cache': 'contiguous'}, 'paged')]
        for overrides in ({'num_layers': 3}, {'num_kv_heads': 2}, {'head_dim': 4},
                          {'dtype': torch.bfloat16}, {'page_size': 8}):
            pool = pool_for(self.weights, self.config, num_pages=4,
                            **{'page_size': 16, **overrides})
            cases.append(({'kv_pool': pool}, 'kv_pool'))
        if torch.cuda.is_available():
            pool = pool_for(self.weights, self.config, num_pages=4, page_size=16, device='cuda')
            cases.append(({'kv_pool': pool}, 'kv_pool'))
        for options, message in cases:
            with (self.subTest(options=options),
                  patch.object(self.tokenizer, 'apply_chat_template') as tokenize,
                  self.assertRaisesRegex(ValueError, message)):
                self.generate(**options)
            tokenize.assert_not_called()

    def test_private_decode_validates_pool_before_preparation(self):
        pool = pool_for(self.weights, self.config, num_pages=4, page_size=16, num_layers=3)
        with (patch('micro_vllm.generate.prepare_model') as prepare,
              self.assertRaisesRegex(ValueError, 'kv_pool')):
            generate._greedy_decode(torch.tensor([[2, 4, 3]]), self.weights, self.config, 2,
                                    eos_token_id=None, kv_cache='paged', kv_pool=pool)
        prepare.assert_not_called()

    def test_reference_fallback_after_growth_matches_contiguous(self):
        prepared = model.prepare_model(self.weights, self.config, attention_backend='torch')
        pool = pool_for(self.weights, self.config, num_pages=8, page_size=2)
        other = PagedKVCache.for_pool(pool, max_tokens=8)
        cache = PagedKVCache.for_pool(pool, max_tokens=8)
        dense = ContiguousKVCache.allocate(num_layers=2, batch_size=1, num_kv_heads=1,
                                           capacity=8, head_dim=8, dtype=torch.float32,
                                           device='cpu')
        for pages in pool._keys + pool._values:
            pages.fill_(float('nan'))
        ids = torch.tensor([[2, 4, 5, 6, 7, 3, 4, 5]])
        offset = 0
        with patch.object(PagedKVCache, 'gather', autospec=True,
                          side_effect=PagedKVCache.gather) as gather:
            for count in (3, 1, 2, 2):
                other.reserve(1)
                other.append(0, *(torch.zeros((1, 1, 1, 8)),) * 2)
                other.append(1, *(torch.zeros((1, 1, 1, 8)),) * 2)
                chunk = ids[:, offset:offset + count]
                actual = prepared.forward(chunk, cache=cache)
                expected = prepared.forward(chunk, cache=dense)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                offset += count
        self.assertEqual(gather.call_count, 3 * self.config.num_hidden_layers)
        self.assertEqual(cache.page_table, (1, 2, 4, 5))
        self.assertEqual(cache.sequence_length, 8)


@unittest.skipUnless(GPU_TESTS, 'set RUN_GPU_TESTS=1 for CUDA pool generation')
class PoolGenerationCUDATests(PoolGenerationMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = make_qwen(accelerated=True, device='cuda', dtype=torch.float16)
        cls.weights, cls.config = checkpoint.state_dict(), checkpoint.config
        cls.tokenizer = make_tokenizer()

    def test_sequential_requests_reuse_poisoned_pages_on_native_paths(self):
        for backend, page_size in (('auto', 3), ('cuda', 2), ('torch', 3)):
            with self.subTest(backend=backend, page_size=page_size):
                pool = pool_for(self.weights, self.config,
                                num_pages=(6 + page_size) // page_size, page_size=page_size)
                with (patch.object(PagedKVCache, 'gather', autospec=True,
                                   side_effect=PagedKVCache.gather) as gather,
                      patch('micro_vllm.model.decode_attention_paged',
                            wraps=cuda_attention.decode_attention_paged) as native):
                    self.assert_sequential_requests_reuse_poisoned_pages(
                        pool, page_size, attention_backend=backend, budget=4)
                if backend == 'torch':
                    native.assert_not_called()
                    self.assertGreater(gather.call_count, 0)
                else:
                    gather.assert_not_called()
                    self.assertEqual(native.call_count, 2 * 3 * self.config.num_hidden_layers)

    def test_mid_generation_exhaustion_retires_then_frees_pages(self):
        self.assert_mid_generation_exhaustion_releases_request(2, 4)

    def test_injected_failure_releases_pages_through_cleanup(self):
        pool = pool_for(self.weights, self.config, num_pages=4, page_size=2)
        self.assert_injected_failure_releases_pages(pool, 2)

    def test_release_retires_pages_until_generation_work_completes(self):
        pool = pool_for(self.weights, self.config, num_pages=4, page_size=2)
        real_forward = generate.forward

        def delayed_forward(*args, **kwargs):
            result = real_forward(*args, **kwargs)
            torch.cuda._sleep(400_000_000)
            return result

        with patch('micro_vllm.generate.forward', side_effect=delayed_forward):
            self.generate(kv_pool=pool, page_size=2)
        self.assertEqual(self.states(pool), [PageState.RETIRING] * 4)
        torch.cuda.synchronize()
        self.assertEqual(pool.reclaim(), 4)


if __name__ == '__main__':
    unittest.main()

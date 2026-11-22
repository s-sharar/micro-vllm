"""Batched paged execution: per-row tables, lengths, RoPE positions and failure semantics."""
import os
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from micro_vllm import attention_backend, cuda_attention, cuda_paged_cache, generate, model, reference
from micro_vllm.attention import causal_attention
from micro_vllm.kv_pool import KVPool, KVPoolExhausted, PageState
from micro_vllm.paged_kv_batch import PagedKVBatch
from micro_vllm.paged_kv_cache import PagedKVCache
from test.test_accelerated_generation import make_qwen

GPU_TESTS = os.environ.get('RUN_GPU_TESTS') == '1' and torch.cuda.is_available()
MODEL_TESTS = os.environ.get('RUN_MODEL_TESTS') == '1'
FREE, OWNED = PageState.FREE, PageState.OWNED


def make_pool(checkpoint, *, num_pages, page_size):
    config, weight = checkpoint.config, checkpoint.get_input_embeddings().weight
    return KVPool.allocate(
        num_layers=config.num_hidden_layers, num_kv_heads=config.num_key_value_heads,
        num_pages=num_pages, page_size=page_size,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=weight.dtype, device=weight.device)


def settle(pool):
    if pool.device.type == 'cuda':
        torch.cuda.synchronize(pool.device)
    pool.reclaim()


def shuffle_free_pages(pool, seed=0):
    """Reorder the FIFO free list so rows receive interleaved, non-identity pages."""
    fillers = [PagedKVCache.for_pool(pool, max_tokens=pool.page_size).reserve(1)
               for _ in range(pool.num_free)]
    order = torch.randperm(len(fillers), generator=torch.Generator().manual_seed(seed)).tolist()
    for index in order:
        fillers[index].release()
    settle(pool)


def poison(pool):
    for buffer in pool._keys + pool._values:
        buffer.fill_(float('nan'))


def snapshot(pool, caches):
    return dict(
        states=[pool.state(page) for page in range(pool.num_pages)],
        owners=list(pool._owners), free=list(pool._free),
        tables=[cache.page_table for cache in caches],
        lengths=[list(cache._lengths) for cache in caches],
        device_tables=[cache._device_page_table.clone() for cache in caches],
        flags=[(cache._failed, cache._released) for cache in caches],
        buffers=[buffer.clone() for buffer in pool._keys + pool._values])


class BatchedForwardMixin:
    backends = ('auto', 'torch')
    page_size = 3

    def prepared(self, backend='auto'):
        return model.prepare_model(self.weights, self.config, attention_backend=backend)

    def prompts(self, lengths=(5, 1, 8, 3), seed=4):
        generator = torch.Generator().manual_seed(seed)
        return [torch.randint(1, self.config.vocab_size, (length,), generator=generator)
                .to(self.device) for length in lengths]

    def assert_snapshot(self, before, after):
        for name in ('states', 'owners', 'free', 'tables', 'lengths', 'flags'):
            self.assertEqual(before[name], after[name], name)
        for left, right in zip(before['device_tables'] + before['buffers'],
                               after['device_tables'] + after['buffers']):
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)

    def run_single(self, prepared, prompt, steps):
        pool = make_pool(self.checkpoint, num_pages=-(-(prompt.numel() + steps) // self.page_size),
                         page_size=self.page_size)
        cache = PagedKVCache.for_pool(pool, max_tokens=prompt.numel() + steps)
        logits = [prepared.forward(prompt[None], cache=cache, logits_to_keep=1)[:, -1]]
        for _ in range(steps):
            token = logits[-1].argmax(-1, keepdim=True)
            logits.append(prepared.forward(token, cache=cache, logits_to_keep=1)[:, -1])
        stored = [cache.gather(layer) for layer in range(cache.num_layers)]
        cache.release()
        return torch.cat(logits), stored

    def run_batch(self, prepared, prompts, steps, pool):
        caches = [PagedKVCache.for_pool(pool, max_tokens=prompt.numel() + steps)
                  for prompt in prompts]
        logits = [prepared.forward_batch(prompts, caches)]
        for _ in range(steps):
            tokens = logits[-1].argmax(-1, keepdim=True)
            logits.append(prepared.forward_batch(list(tokens), caches))
        return torch.stack(logits, dim=1), caches

    def batch_pool(self, spare=4, prompts=None, steps=5):
        prompts = self.prompts() if prompts is None else prompts
        pages = sum(-(-(prompt.numel() + steps) // self.page_size) for prompt in prompts)
        pool = make_pool(self.checkpoint, num_pages=pages + spare, page_size=self.page_size)
        shuffle_free_pages(pool)
        poison(pool)
        return pool

    def test_rows_match_independent_requests_across_growth_and_shuffled_pages(self):
        prompts, steps = self.prompts(), 5
        shuffled = False
        for backend in self.backends:
            prepared = self.prepared(backend)
            pool = self.batch_pool(prompts=prompts, steps=steps)
            batched, caches = self.run_batch(prepared, prompts, steps, pool)
            self.assertEqual(batched.shape, (len(prompts), steps + 1, self.config.vocab_size))
            for row, (prompt, cache) in enumerate(zip(prompts, caches)):
                with self.subTest(backend=backend, row=row):
                    expected, stored = self.run_single(prepared, prompt, steps)
                    torch.testing.assert_close(batched[row], expected, **self.tolerance)
                    self.assertTrue(torch.equal(batched[row].argmax(-1), expected.argmax(-1)))
                    self.assertEqual(cache.sequence_length, prompt.numel() + steps)
                    pages = cache.page_table
                    self.assertEqual(len(pages), -(-(prompt.numel() + steps) // self.page_size))
                    shuffled = shuffled or list(pages) != sorted(pages)
                    for layer, (key, value) in enumerate(stored):
                        actual = cache.gather(layer)
                        torch.testing.assert_close(actual[0], key, **self.tolerance)
                        torch.testing.assert_close(actual[1], value, **self.tolerance)
            for cache in caches:
                cache.release()
        self.assertTrue(shuffled)

    def test_single_row_batch_matches_forward_bit_for_bit(self):
        prompt, steps = self.prompts((7,))[0], 4
        for backend in self.backends:
            with self.subTest(backend=backend):
                prepared = self.prepared(backend)
                expected, stored = self.run_single(prepared, prompt, steps)
                pool = self.batch_pool(prompts=[prompt], steps=steps)
                actual, (cache,) = self.run_batch(prepared, [prompt], steps, pool)
                torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
                for layer, pair in enumerate(stored):
                    for left, right in zip(cache.gather(layer), pair):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_padding_never_enters_cache_or_changes_valid_logits(self):
        prompts = self.prompts((5, 1, 8, 3))
        counts = [prompt.numel() for prompt in prompts]
        real_embed = model.embed_tokens

        def poisoned(ids, weight):
            hidden = real_embed(ids, weight)
            if ids.shape == (len(prompts), max(counts)):
                for row, count in enumerate(counts):
                    hidden[row, count:] = float('nan')
            return hidden

        for backend in self.backends:
            prepared, runs = self.prepared(backend), []
            for embed in (real_embed, poisoned):
                pool = self.batch_pool(prompts=prompts, steps=0)
                caches = [PagedKVCache.for_pool(pool, max_tokens=count) for count in counts]
                with patch('micro_vllm.runtime.model.embed_tokens', side_effect=embed):
                    logits = prepared.forward_batch(prompts, caches)
                stored = [cache.gather(layer) for cache in caches for layer in range(cache.num_layers)]
                runs.append((logits, stored, pool, caches))
            (clean, clean_stored, _, _), (dirty, dirty_stored, _, _) = runs
            with self.subTest(backend=backend):
                self.assertTrue(torch.isfinite(dirty).all())
                torch.testing.assert_close(dirty, clean, rtol=0, atol=0)
                for left, right in zip(dirty_stored, clean_stored):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                # The clean run's padding K/V is finite, so any padded write would show here.
                for _, _, pool, caches in runs:
                    valid = {(cache.page_table[position // self.page_size],
                              position % self.page_size)
                             for cache, count in zip(caches, counts) for position in range(count)}
                    for buffer in pool._keys + pool._values:
                        finite = torch.isfinite(buffer).all(dim=(1, 3)).tolist()
                        for page in range(pool.num_pages):
                            for slot in range(pool.page_size):
                                self.assertEqual(finite[page][slot], (page, slot) in valid,
                                                 (page, slot))

    def test_exhaustion_leaves_every_row_unchanged_and_usable(self):
        prepared = self.prepared()
        prompts = self.prompts((6, 3, 5))
        pool = make_pool(self.checkpoint, num_pages=5, page_size=self.page_size)
        blocker = PagedKVCache.for_pool(pool, max_tokens=self.page_size).reserve(1)
        caches = [PagedKVCache.for_pool(pool, max_tokens=prompt.numel() + 2) for prompt in prompts]
        before = snapshot(pool, caches)
        with self.assertRaises(KVPoolExhausted):
            prepared.forward_batch(prompts, caches)
        self.assert_snapshot(before, snapshot(pool, caches))
        tokens = prepared.forward_batch(prompts[:2], caches[:2]).argmax(-1, keepdim=True)
        self.assertEqual(pool.num_free, 1)
        decode_before = snapshot(pool, caches)
        with self.assertRaises(KVPoolExhausted):
            prepared.forward_batch(list(tokens), caches[:2])
        self.assert_snapshot(decode_before, snapshot(pool, caches))
        blocker.release()
        settle(pool)
        actual = prepared.forward_batch(list(tokens), caches[:2])
        expected = [self.run_single(prepared, prompt, 1)[0][-1] for prompt in prompts[:2]]
        torch.testing.assert_close(actual, torch.stack(expected), **self.tolerance)
        self.assertEqual([cache.sequence_length for cache in caches], [7, 4, 0])

    def test_prewrite_validation_failures_change_nothing(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=3)
        other = make_pool(self.checkpoint, num_pages=4, page_size=self.page_size)
        caches = [PagedKVCache.for_pool(pool, max_tokens=prompt.numel() + 3) for prompt in prompts]
        prepared.forward_batch(prompts, caches)
        token = prompts[0][:1]
        released = PagedKVCache.for_pool(pool, max_tokens=3)
        released.release()
        failed = PagedKVCache.for_pool(pool, max_tokens=3)
        failed._failed = True
        fresh = PagedKVCache.for_pool(pool, max_tokens=4)
        small = PagedKVCache.for_pool(pool, max_tokens=prompts[0].numel() - 1)
        misaligned = PagedKVCache.for_pool(pool, max_tokens=6)
        misaligned.reserve(2)
        misaligned._lengths[0] = 1
        wrong_geometry = KVPool.allocate(
            num_layers=self.config.num_hidden_layers, num_kv_heads=self.config.num_key_value_heads,
            num_pages=4, page_size=self.page_size, head_dim=2 * pool.head_dim,
            dtype=pool.dtype, device=pool.device)
        long_prompt = torch.ones(self.config.max_position_embeddings + 1, dtype=torch.int64,
                                 device=self.device)
        huge_pool = make_pool(self.checkpoint, num_pages=self.config.max_position_embeddings,
                              page_size=self.page_size)
        cases = [
            ([], [], 'nonempty'),
            ([token], caches, 'one cache per row'),
            ([token, token], [caches[0], caches[0]], 'distinct'),
            ([token, prompts[0]], [caches[0], PagedKVCache.for_pool(other, max_tokens=4)],
             'same KVPool'),
            ([token, token], [caches[0], released], 'released'),
            ([token, token], [caches[0], failed], 'failed'),
            ([token, prompts[1]], [caches[0], fresh], 'mixed'),
            ([prompts[1], token], caches, 'one token'),
            ([token, token[None]], caches, 'one-dimensional'),
            ([token, token.float()], caches, 'integer'),
            ([token, [1]], caches, 'tensor'),
            ([token, token[:0]], caches, 'nonempty'),
            ([prompts[0], prompts[0]], [fresh, small], 'capacity'),
            ([token, token], [caches[0], misaligned], 'aligned'),
            ([token, token], [caches[0], ContiguousCacheStandIn()], 'PagedKVCache'),
            ([prompts[0]], [PagedKVCache.for_pool(wrong_geometry, max_tokens=4)], 'geometry'),
            ([long_prompt], [PagedKVCache.for_pool(huge_pool, max_tokens=long_prompt.numel())],
             'context length'),
        ]
        everything = caches + [fresh, small, misaligned]
        for rows, batch_caches, message in cases:
            before = snapshot(pool, everything)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                prepared.forward_batch(rows, batch_caches)
            self.assert_snapshot(before, snapshot(pool, everything))
        with self.assertRaisesRegex(ValueError, 'attention_backend'):
            prepared.forward_batch([token, token], caches, attention_backend='flash')
        batch = PagedKVBatch.prepare(
            caches, [1, 1], num_layers=pool.num_layers, num_kv_heads=pool.num_kv_heads,
            head_dim=pool.head_dim, dtype=pool.dtype, device=pool.device, max_positions=128)
        hidden = torch.zeros(2, 1, self.config.hidden_size, dtype=pool.dtype, device=self.device)
        rope = torch.zeros(2, 1, pool.head_dim, dtype=pool.dtype, device=self.device)
        with self.assertRaisesRegex(ValueError, 'per-row masks'):
            model.decoder_layer(hidden, prepared.layers[0], rope, rope,
                                num_attention_heads=self.config.num_attention_heads,
                                num_key_value_heads=self.config.num_key_value_heads, eps=1e-6,
                                cache=batch, layer_idx=0,
                                attention_mask=torch.zeros(1, 2, dtype=torch.bool))

    def test_failure_after_layer_writes_releases_every_row(self):
        prompts = self.prompts((4, 2, 5))
        for failing_layer, error in ((1, RuntimeError('injected layer failure')),
                                     (1, KeyboardInterrupt()),
                                     (0, RuntimeError('injected first layer failure'))):
            with self.subTest(layer=failing_layer, error=type(error).__name__):
                prepared = self.prepared()
                pool = self.batch_pool(prompts=prompts, steps=3, spare=0)
                caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 3) for p in prompts]
                prepared.forward_batch(prompts, caches)
                tokens = [prompt[:1] for prompt in prompts]
                real_layer = model.decoder_layer

                def layer(*args, **kwargs):
                    if kwargs['layer_idx'] == failing_layer:
                        raise error
                    return real_layer(*args, **kwargs)

                with patch('micro_vllm.runtime.model.decoder_layer', side_effect=layer), \
                        self.assertRaises(type(error)):
                    prepared.forward_batch(tokens, caches)
                for cache in caches:
                    self.assertTrue(cache._released)
                    self.assertEqual(cache.page_table, ())
                    self.assertEqual(list(cache._lengths), [0] * cache.num_layers)
                    with self.assertRaisesRegex(ValueError, 'released'):
                        cache.reserve(1)
                settle(pool)
                self.assertEqual([pool.state(page) for page in range(pool.num_pages)],
                                 [FREE] * pool.num_pages)

    def test_failure_after_reservation_before_writes_releases_every_row(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=0, spare=0)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel()) for p in prompts]
        with patch('micro_vllm.runtime.model.embed_tokens', side_effect=RuntimeError('embed')), \
                self.assertRaisesRegex(RuntimeError, 'embed'):
            prepared.forward_batch(prompts, caches)
        self.assertTrue(all(cache._released for cache in caches))
        settle(pool)
        self.assertEqual(pool.num_free, pool.num_pages)

    def test_publication_failure_releases_published_and_unpublished_pages(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 5, 2))
        pool = self.batch_pool(prompts=prompts, steps=0, spare=0)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel()) for p in prompts]
        real, calls = PagedKVCache._publish, []

        def publish(cache, first, pages):
            calls.append(cache)
            if len(calls) == 2:
                raise RuntimeError('injected publication failure')
            return real(cache, first, pages)

        with patch.object(PagedKVCache, '_publish', autospec=True, side_effect=publish), \
                self.assertRaisesRegex(RuntimeError, 'publication'):
            prepared.forward_batch(prompts, caches)
        self.assertTrue(all(cache._released for cache in caches))
        settle(pool)
        self.assertEqual(pool.num_free, pool.num_pages)

    def test_release_failure_during_cleanup_is_noted_and_retryable(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=0, spare=0)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel()) for p in prompts]
        real_release, calls = KVPool._release, []

        def release(pool_, *args):
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError('injected release failure')
            return real_release(pool_, *args)

        with patch.object(KVPool, '_release', autospec=True, side_effect=release), \
                patch('micro_vllm.runtime.model.embed_tokens', side_effect=RuntimeError('embed')), \
                self.assertRaisesRegex(RuntimeError, 'embed') as raised:
            prepared.forward_batch(prompts, caches)
        self.assertTrue(any('injected release failure' in note
                            for note in getattr(raised.exception, '__notes__', [])))
        self.assertTrue(caches[0]._failed and not caches[0]._released)
        self.assertTrue(caches[1]._released)
        with self.assertRaisesRegex(ValueError, 'failed'):
            caches[0].reserve(1)
        caches[0].release()
        settle(pool)
        self.assertEqual(pool.num_free, pool.num_pages)

    def test_finished_batch_cannot_touch_pages_again(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=2)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 2) for p in prompts]
        batches, real_prepare = [], PagedKVBatch.prepare

        def capture(*args, **kwargs):
            batches.append(real_prepare(*args, **kwargs))
            return batches[-1]

        with patch.object(PagedKVBatch, 'prepare', side_effect=capture):
            prepared.forward_batch(prompts, caches)
            with patch('micro_vllm.runtime.model.embed_tokens', side_effect=RuntimeError('embed')), \
                    self.assertRaises(RuntimeError):
                prepared.forward_batch([prompt[:1] for prompt in prompts], caches)
        key = torch.zeros(2, pool.num_kv_heads, 1, pool.head_dim, dtype=pool.dtype,
                          device=self.device)
        for batch in batches:
            with self.subTest(committed=batch is batches[0]):
                with self.assertRaisesRegex(ValueError, 'live'):
                    batch.append(0, key, key)
                with self.assertRaisesRegex(ValueError, 'live'):
                    batch.gather(0)

    def test_unpublished_grant_stays_releasable_when_cleanup_release_fails(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 5, 2))
        pool = self.batch_pool(prompts=prompts, steps=0, spare=0)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel()) for p in prompts]
        real_publish, real_release, failed = PagedKVCache._publish, KVPool._release, []

        def publish(cache, first, pages):
            if cache is caches[1]:
                raise RuntimeError('injected publication failure')
            return real_publish(cache, first, pages)

        def release(pool_, owner, *args):
            if owner is caches[1]._owner and not failed:
                failed.append(owner)
                raise RuntimeError('injected release failure')
            return real_release(pool_, owner, *args)

        with patch.object(PagedKVCache, '_publish', autospec=True, side_effect=publish), \
                patch.object(KVPool, '_release', autospec=True, side_effect=release), \
                self.assertRaisesRegex(RuntimeError, 'publication'):
            prepared.forward_batch(prompts, caches)
        self.assertEqual(failed, [caches[1]._owner])
        self.assertTrue(caches[0]._released and caches[2]._released)
        self.assertTrue(caches[1]._failed and not caches[1]._released)
        caches[1].release()
        settle(pool)
        self.assertEqual(pool.num_free, pool.num_pages)

    def test_cleanup_reporting_works_without_exception_notes(self):
        class LegacyError(RuntimeError):
            add_note = None  # Python 3.10 exceptions have no add_note.

        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=0, spare=0)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel()) for p in prompts]
        real_release, calls = KVPool._release, []

        def release(pool_, *args):
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError('injected release failure')
            return real_release(pool_, *args)

        with patch.object(KVPool, '_release', autospec=True, side_effect=release), \
                patch('micro_vllm.runtime.model.embed_tokens', side_effect=LegacyError('embed')), \
                self.assertWarnsRegex(RuntimeWarning, 'injected release failure'), \
                self.assertRaises(LegacyError):
            prepared.forward_batch(prompts, caches)
        self.assertTrue(caches[1]._released)
        caches[0].release()
        settle(pool)
        self.assertEqual(pool.num_free, pool.num_pages)

    def test_interrupted_length_commit_releases_every_row(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=0, spare=0)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel()) for p in prompts]

        def partial_commit(batch):
            batch.caches[0]._lengths[:] = [batch.ends[0]] * batch.caches[0].num_layers
            raise KeyboardInterrupt

        with patch.object(PagedKVBatch, 'commit', autospec=True, side_effect=partial_commit), \
                self.assertRaises(KeyboardInterrupt):
            prepared.forward_batch(prompts, caches)
        self.assertTrue(all(cache._released for cache in caches))
        self.assertEqual([list(cache._lengths) for cache in caches], [[0, 0], [0, 0]])
        settle(pool)
        self.assertEqual(pool.num_free, pool.num_pages)

    def test_lengths_commit_only_after_the_whole_forward_succeeds(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=1)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 1) for p in prompts]
        seen, real_layer = [], model.decoder_layer

        def layer(*args, **kwargs):
            result = real_layer(*args, **kwargs)
            seen.append([list(cache._lengths) for cache in caches])
            return result

        with patch('micro_vllm.runtime.model.decoder_layer', side_effect=layer):
            prepared.forward_batch(prompts, caches)
        self.assertTrue(all(lengths == [[0, 0], [0, 0]] for lengths in seen))
        self.assertEqual([list(cache._lengths) for cache in caches], [[4, 4], [2, 2]])

    def test_per_row_reference_mask_matches_separate_attention(self):
        generator = torch.Generator().manual_seed(1)
        query = torch.randn(3, 4, 1, 8, generator=generator)
        key = torch.randn(3, 2, 6, 8, generator=generator)
        value = torch.randn(3, 2, 6, 8, generator=generator)
        lengths = [6, 2, 4]
        mask = torch.arange(6)[None, None, :] >= torch.tensor(lengths)[:, None, None]
        value[1, :, 2:] = float('nan')
        actual = causal_attention(query, key, value.nan_to_num(0), past_length=5, row_mask=mask)
        for row, length in enumerate(lengths):
            expected = causal_attention(query[row:row + 1], key[row:row + 1, :, :length],
                                        value[row:row + 1, :, :length], past_length=length - 1)
            torch.testing.assert_close(actual[row:row + 1], expected)
        for invalid in (mask[:2], mask.int(), mask[0]):
            with self.assertRaisesRegex(ValueError, 'row_mask'):
                causal_attention(query, key, value, past_length=5, row_mask=invalid)
        with self.assertRaisesRegex(ValueError, 'not both'):
            causal_attention(query, key, value, past_length=5, row_mask=mask,
                             attention_mask=mask[0])


class ContiguousCacheStandIn:
    pass


class BatchedForwardCPUTests(BatchedForwardMixin, unittest.TestCase):
    device = 'cpu'
    tolerance = dict(rtol=1e-4, atol=1e-5)

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = make_qwen()
        cls.weights, cls.config = cls.checkpoint.state_dict(), cls.checkpoint.config

    def test_cuda_backend_on_cpu_rejects_batch_before_reservation(self):
        pool = make_pool(self.checkpoint, num_pages=2, page_size=3)
        caches = [PagedKVCache.for_pool(pool, max_tokens=3) for _ in range(2)]
        before = snapshot(pool, caches)
        with self.assertRaisesRegex(ValueError, 'cuda'):
            self.prepared().forward_batch(self.prompts((2, 3)), caches, attention_backend='cuda')
        self.assert_snapshot(before, snapshot(pool, caches))


@unittest.skipUnless(GPU_TESTS, 'set RUN_GPU_TESTS=1 for batched CUDA paged execution')
class BatchedForwardCUDAFP16Tests(BatchedForwardMixin, unittest.TestCase):
    device = 'cuda'
    dtype = torch.float16
    backends = ('auto', 'cuda', 'torch')
    tolerance = dict(rtol=5e-3, atol=5e-3)

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = make_qwen(accelerated=True, device='cuda', dtype=cls.dtype)
        cls.weights, cls.config = cls.checkpoint.state_dict(), cls.checkpoint.config

    def forbid_gather(self):
        error = AssertionError('native batched decode must not gather')
        return [patch.object(PagedKVCache, 'gather', side_effect=error),
                patch.object(PagedKVCache, '_gather_buffer', side_effect=error),
                patch.object(PagedKVBatch, 'gather', side_effect=error)]

    def test_eligible_decode_issues_one_batched_scatter_and_attention_per_layer(self):
        layers = self.config.num_hidden_layers
        prompts = self.prompts((5, 1, 8, 3))
        for backend in ('auto', 'cuda'):
            with self.subTest(backend=backend):
                prepared = self.prepared(backend)
                pool = self.batch_pool(prompts=prompts, steps=3)
                caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 3) for p in prompts]
                with patch.object(attention_backend, 'flash_attention',
                                  wraps=attention_backend.flash_attention) as flash:
                    logits = prepared.forward_batch(prompts, caches)
                self.assertEqual(flash.call_count, 0)
                for _ in range(3):
                    tokens = list(logits.argmax(-1, keepdim=True))
                    guards = self.forbid_gather()
                    for guard in guards:
                        guard.start()
                    try:
                        with patch.object(cuda_paged_cache, '_write_paged_kv_batch',
                                          wraps=cuda_paged_cache._write_paged_kv_batch) as write, \
                                patch.object(cuda_paged_cache, '_write_paged_kv',
                                             side_effect=AssertionError('single write')), \
                                patch.object(model, 'decode_attention_paged_batch',
                                             wraps=cuda_attention.decode_attention_paged_batch) as decode, \
                                patch.object(model, 'decode_attention_paged',
                                             side_effect=AssertionError('single decode')):
                            logits = prepared.forward_batch(tokens, caches)
                    finally:
                        for guard in guards:
                            guard.stop()
                    self.assertEqual(write.call_count, layers)
                    self.assertEqual(decode.call_count, layers)

    def test_direct_batch_operations_reject_invalid_use_before_launch(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=1)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 1) for p in prompts]
        prepared.forward_batch(prompts, caches)
        batch = PagedKVBatch.prepare(
            caches, [1, 1], num_layers=pool.num_layers, num_kv_heads=pool.num_kv_heads,
            head_dim=pool.head_dim, dtype=pool.dtype, device=pool.device, max_positions=128)
        query = torch.zeros(2, self.config.num_attention_heads, 1, pool.head_dim,
                            dtype=pool.dtype, device='cuda')
        with self.assertRaisesRegex(ValueError, 'reserved'):
            cuda_attention.decode_attention_paged_batch(query, batch, layer_idx=0)
        batch.reserve()
        with self.assertRaisesRegex(ValueError, 'layer index'):
            cuda_attention.decode_attention_paged_batch(query, batch, layer_idx=pool.num_layers)
        with self.assertRaisesRegex(ValueError, 'query shape'):
            cuda_attention.decode_attention_paged_batch(query[:1], batch, layer_idx=0)
        key = torch.zeros(2, pool.num_kv_heads, 2, pool.head_dim, dtype=pool.dtype, device='cuda')
        with self.assertRaisesRegex(ValueError, 'batched K/V must have shape'):
            batch.append(0, key, key)
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaisesRegex(ValueError, 'stream'):
                cuda_attention.decode_attention_paged_batch(query, batch, layer_idx=0)
            with self.assertRaisesRegex(ValueError, 'stream'):
                batch.append(0, key[:, :, :1], key[:, :, :1])
        for cache in caches:
            cache.release()

    def test_first_use_on_another_stream_waits_for_table_initialization(self):
        prepared = self.prepared()
        prompts = self.prompts((5, 2))
        expected_pool = self.batch_pool(prompts=prompts, steps=0)
        expected = prepared.forward_batch(prompts, [
            PagedKVCache.for_pool(expected_pool, max_tokens=p.numel()) for p in prompts])
        creator, user = torch.cuda.Stream(), torch.cuda.Stream()
        pool = self.batch_pool(prompts=prompts, steps=0, spare=3)
        torch.cuda.synchronize()
        with torch.cuda.stream(creator):
            torch.cuda._sleep(400_000_000)
            single = PagedKVCache.for_pool(pool, max_tokens=4)
            caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel()) for p in prompts]
        with torch.cuda.stream(user):
            single.reserve(4)
            actual = prepared.forward_batch(prompts, caches)
        torch.cuda.synchronize()
        self.assertEqual(single._device_page_table[:2].tolist(), list(single.page_table))
        for cache in caches:
            used = len(cache.page_table)
            self.assertEqual(cache._device_page_table[:used].tolist(), list(cache.page_table))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_exhaustion_leaves_rows_unbound_for_a_retry_on_another_stream(self):
        prepared = self.prepared()
        prompts = self.prompts((6, 3))
        pool = make_pool(self.checkpoint, num_pages=3, page_size=self.page_size)
        blocker = PagedKVCache.for_pool(pool, max_tokens=self.page_size).reserve(1)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel()) for p in prompts]
        single = PagedKVCache.for_pool(pool, max_tokens=9)
        with self.assertRaises(KVPoolExhausted):
            prepared.forward_batch(prompts, caches)
        with self.assertRaises(KVPoolExhausted):
            single.reserve(9)
        self.assertTrue(all(cache._stream is None for cache in caches + [single]))
        blocker.release()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            prepared.forward_batch(prompts, caches)
        self.assertTrue(all(cache._stream == stream for cache in caches))
        stream.synchronize()

    def test_finished_batch_cannot_decode_from_recycled_pages(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=1)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 1) for p in prompts]
        prepared.forward_batch(prompts, caches)
        batch = PagedKVBatch.prepare(
            caches, [1, 1], num_layers=pool.num_layers, num_kv_heads=pool.num_kv_heads,
            head_dim=pool.head_dim, dtype=pool.dtype, device=pool.device, max_positions=128)
        batch.reserve()
        for cache in caches:
            cache.release()
        query = torch.zeros(2, self.config.num_attention_heads, 1, pool.head_dim,
                            dtype=pool.dtype, device='cuda')
        with self.assertRaisesRegex(ValueError, 'live'):
            cuda_attention.decode_attention_paged_batch(query, batch, layer_idx=0)
        key = torch.zeros(2, pool.num_kv_heads, 1, pool.head_dim, dtype=pool.dtype, device='cuda')
        with self.assertRaisesRegex(ValueError, 'live'):
            batch.append(0, key, key)

    def test_single_row_auto_prefill_keeps_flash(self):
        prepared = self.prepared('auto')
        prompt = self.prompts((6,))
        pool = self.batch_pool(prompts=prompt, steps=0)
        with patch.object(attention_backend, 'flash_attention',
                          wraps=attention_backend.flash_attention) as flash:
            prepared.forward_batch(prompt, [PagedKVCache.for_pool(pool, max_tokens=6)])
        flash_layers = self.config.num_hidden_layers if self.dtype == torch.float16 else 0
        self.assertEqual(flash.call_count, flash_layers)

    def test_torch_backend_decode_uses_padded_reference_gather(self):
        prepared = self.prepared('torch')
        prompts = self.prompts((5, 2))
        pool = self.batch_pool(prompts=prompts, steps=1)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 1) for p in prompts]
        tokens = list(prepared.forward_batch(prompts, caches).argmax(-1, keepdim=True))
        with patch.object(PagedKVBatch, 'gather', autospec=True,
                          side_effect=PagedKVBatch.gather) as gather, \
                patch.object(model, 'decode_attention_paged_batch',
                             side_effect=AssertionError('reference mode')):
            prepared.forward_batch(tokens, caches)
        self.assertEqual(gather.call_count, self.config.num_hidden_layers)

    def test_decode_and_prefill_issue_no_host_synchronization(self):
        prepared = self.prepared('auto')
        prompts = self.prompts((5, 1, 8))
        pool = self.batch_pool(prompts=prompts, steps=2)
        warm = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 2) for p in prompts]
        prepared.forward_batch([prompt[:1] for prompt in prompts], warm)
        prepared.forward_batch([prompt[:1] for prompt in prompts], warm)
        for cache in warm:
            cache.release()
        settle(pool)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 2) for p in prompts]
        tokens = [prompt[:1] for prompt in prompts]
        torch.cuda.set_sync_debug_mode('error')
        try:
            prepared.forward_batch(prompts, caches)
            prepared.forward_batch(tokens, caches)
            prepared.forward_batch(tokens, caches)
        finally:
            torch.cuda.set_sync_debug_mode('default')

    def test_stream_binding_is_checked_for_every_row_before_changes(self):
        prepared = self.prepared()
        prompts = self.prompts((4, 2))
        pool = self.batch_pool(prompts=prompts, steps=1)
        caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 1) for p in prompts]
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            caches[1].reserve(1)
        before = snapshot(pool, caches)
        with self.assertRaisesRegex(ValueError, 'stream'):
            prepared.forward_batch(prompts, caches)
        self.assert_snapshot(before, snapshot(pool, caches))
        self.assertIsNone(caches[0]._stream)
        fresh = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + 1) for p in prompts]
        with torch.cuda.stream(stream):
            caches[1].release()
            prepared.forward_batch(prompts, fresh)
        self.assertTrue(all(cache._stream == stream for cache in fresh))
        with self.assertRaisesRegex(ValueError, 'stream'):
            prepared.forward_batch([prompt[:1] for prompt in prompts], fresh)
        stream.synchronize()


@unittest.skipUnless(GPU_TESTS, 'set RUN_GPU_TESTS=1 for batched CUDA paged execution')
class BatchedForwardCUDAFP32Tests(BatchedForwardCUDAFP16Tests):
    dtype = torch.float32
    tolerance = dict(rtol=1e-4, atol=1e-4)


@unittest.skipUnless(GPU_TESTS and MODEL_TESTS,
                     'set RUN_GPU_TESTS=1 RUN_MODEL_TESTS=1 for offline real-Qwen batching')
class RealQwenBatchedTests(unittest.TestCase):
    # Same tolerance as the existing real-Qwen paged-vs-HF logit checks.
    tolerance = dict(rtol=5e-3, atol=.125)

    @classmethod
    def setUpClass(cls):
        name = 'Qwen/Qwen2.5-0.5B-Instruct'
        cls.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        cls.checkpoint = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=torch.float16,
            attn_implementation='eager').to('cuda').eval()
        cls.weights, cls.config = cls.checkpoint.state_dict(), cls.checkpoint.config

    @classmethod
    def tearDownClass(cls):
        del cls.weights, cls.config, cls.checkpoint, cls.tokenizer

    def ids(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors='pt')[0].to('cuda')

    def test_batched_greedy_ids_match_single_requests_and_hf(self):
        texts = ('What is 2 + 2?', 'Explain a KV cache in one sentence.',
                 'List three primary colors and say which one you like most.')
        prompts, steps = [self.ids(text) for text in texts], 8
        self.assertEqual(len({prompt.numel() for prompt in prompts}), 3)
        with patch.object(self.checkpoint.generation_config, 'eos_token_id', None):
            expected = [reference.generate_tokens(self.checkpoint, self.tokenizer, text,
                                                  steps + 1).tolist() for text in texts]
        errors = {}
        for backend in ('auto', 'cuda'):
            prepared = model.prepare_model(self.weights, self.config, attention_backend=backend)
            for page_size in (3, 16):
                with self.subTest(backend=backend, page_size=page_size):
                    pool = KVPool.allocate(
                        num_layers=self.config.num_hidden_layers,
                        num_kv_heads=self.config.num_key_value_heads,
                        num_pages=sum(-(-(p.numel() + steps) // page_size) for p in prompts) + 2,
                        page_size=page_size, head_dim=64, dtype=torch.float16, device='cuda')
                    shuffle_free_pages(pool)
                    caches = [PagedKVCache.for_pool(pool, max_tokens=p.numel() + steps)
                              for p in prompts]
                    logits = [prepared.forward_batch(prompts, caches)]
                    for _ in range(steps):
                        tokens = list(logits[-1].argmax(-1, keepdim=True))
                        with patch.object(PagedKVBatch, 'gather',
                                          side_effect=AssertionError('must not gather')):
                            logits.append(prepared.forward_batch(tokens, caches))
                    batched = torch.stack(logits, dim=1)
                    for row, prompt in enumerate(prompts):
                        single = generate.generate_tokens(
                            self.weights, self.config, self.tokenizer, texts[row], steps + 1,
                            eos_token_id=None, attention_backend=backend, kv_cache='paged',
                            page_size=page_size)
                        self.assertEqual(batched[row].argmax(-1).tolist(), single.tolist())
                        self.assertEqual(single.tolist(), expected[row])
                        reference_logits = self.single_logits(prepared, prompt, single, page_size)
                        torch.testing.assert_close(batched[row], reference_logits, **self.tolerance)
                        errors[backend, page_size, row] = (
                            batched[row].float() - reference_logits.float()).abs().max().item()
                    for cache in caches:
                        cache.release()
        worst = {backend: max(error for (name, _, _), error in errors.items() if name == backend)
                 for backend in ('auto', 'cuda')}
        print(f'\nreal-Qwen batched-vs-single max abs logit error: {worst}')

    def single_logits(self, prepared, prompt, tokens, page_size):
        pool = KVPool.allocate(num_layers=self.config.num_hidden_layers,
                               num_kv_heads=self.config.num_key_value_heads,
                               num_pages=-(-(prompt.numel() + tokens.numel()) // page_size),
                               page_size=page_size, head_dim=64, dtype=torch.float16, device='cuda')
        cache = PagedKVCache.for_pool(pool, max_tokens=prompt.numel() + tokens.numel())
        logits = [prepared.forward(prompt[None], cache=cache, logits_to_keep=1)[:, -1]]
        for token in tokens[:-1]:
            logits.append(prepared.forward(token.view(1, 1), cache=cache, logits_to_keep=1)[:, -1])
        cache.release()
        return torch.cat(logits)


if __name__ == '__main__':
    unittest.main()

"""Continuous batching: tick order, budgets, buffered events, preemption, replay and failures."""
import contextlib
import os
import unittest
import warnings
from collections import deque
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from micro_vllm import cuda_attention, cuda_paged_cache, generate, model
from micro_vllm.engine import Engine, RequestState
from micro_vllm.kv_pool import KVPool, KVPoolExhausted, PageState
from micro_vllm.paged_kv_batch import PagedKVBatch
from micro_vllm.paged_kv_cache import PagedKVCache
from micro_vllm.runtime import PreparedModel
from test.test_accelerated_generation import make_qwen

GPU_TESTS = os.environ.get('RUN_GPU_TESTS') == '1' and torch.cuda.is_available()
MODEL_TESTS = os.environ.get('RUN_MODEL_TESTS') == '1'
WAITING, RUNNING = RequestState.WAITING, RequestState.RUNNING
COMPLETED, CANCELLED, FAILED = RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED


def settle(pool):
    if pool.device.type == 'cuda':
        torch.cuda.synchronize(pool.device)
    pool.reclaim()


def block_free_pages(pool):
    """Own every free page from outside the engine."""
    settle(pool)
    tokens = pool.num_free * pool.page_size
    return PagedKVCache.for_pool(pool, max_tokens=tokens).reserve(tokens)


class SchedulerMixin:
    backend = 'auto'
    page_size = 2

    def make_engine(self, num_pages=32, **options):
        options.setdefault('page_size', self.page_size)
        options.setdefault('attention_backend', self.backend)
        return Engine(self.weights, self.config, num_pages=num_pages, **options)

    def prompt(self, length, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(1, self.config.vocab_size, (length,), generator=generator).tolist()

    def standalone(self, prompt, budget, eos=None, backend=None):
        ids = torch.tensor([prompt], device=self.device)
        return generate._greedy_decode(
            ids, self.weights, self.config, budget, eos_token_id=eos,
            attention_backend=backend or self.backend, kv_cache='paged',
            page_size=self.page_size).tolist()

    @contextlib.contextmanager
    def forwards(self):
        calls, real = [], PreparedModel.forward_batch

        def record(prepared, input_ids, caches, **kwargs):
            call = dict(prefill=caches[0].sequence_length == 0,
                        rows=[row.tolist() for row in input_ids], ok=False)
            calls.append(call)
            logits = real(prepared, input_ids, caches, **kwargs)
            call['ok'] = True
            return logits
        with patch.object(PreparedModel, 'forward_batch', autospec=True, side_effect=record):
            yield calls

    @contextlib.contextmanager
    def pool_releases(self):
        owners, real = [], KVPool._release

        def release(pool, owner, *args):
            owners.append(owner)
            return real(pool, owner, *args)
        with patch.object(KVPool, '_release', autospec=True, side_effect=release):
            yield owners

    @contextlib.contextmanager
    def created_caches(self):
        caches, real = [], PagedKVCache.for_pool

        def create(pool, **kwargs):
            caches.append(real(pool, **kwargs))
            return caches[-1]
        with patch.object(PagedKVCache, 'for_pool', side_effect=create):
            yield caches

    def failing_forward(self, failing_call, error):
        real, calls = PreparedModel.forward_batch, []

        def forward(prepared, input_ids, caches, **kwargs):
            calls.append(len(caches))
            if len(calls) == failing_call:
                raise error
            return real(prepared, input_ids, caches, **kwargs)
        return patch.object(PreparedModel, 'forward_batch', autospec=True, side_effect=forward)

    def failing_layer(self, failing_call, error):
        real_layer, calls = model.decoder_layer, []

        def layer(*args, **kwargs):
            calls.append(kwargs['layer_idx'])
            if len(calls) == failing_call:
                raise error
            return real_layer(*args, **kwargs)
        return patch('micro_vllm.model.decoder_layer', side_effect=layer)

    def shape(self, calls):
        return [('prefill' if call['prefill'] else 'decode', [len(row) for row in call['rows']])
                for call in calls]

    def tick(self, engine, calls):
        start = len(calls)
        events = engine.step_batch()
        return events, calls[start:]

    def drain(self, engine, limit=400):
        events, ticks = [], 0
        while engine.has_work:
            if ticks == limit:
                raise AssertionError('engine still has work after the tick limit')
            events.extend(engine.step_batch())
            ticks += 1
        return events, ticks

    def drain_ticks(self, engine, count):
        return [engine.step_batch() for _ in range(count)]

    def tokens(self, engine, request):
        return list(engine.result(request).tokens)

    def owned_by(self, pool, owner):
        return [page for page in range(pool.num_pages)
                if pool.state(page) is PageState.OWNED and pool._owners[page] is owner]

    def assert_events_match_tokens(self, engine, events, requests):
        for request in requests:
            self.assertEqual([token for event_id, token in events if event_id == request],
                             self.tokens(engine, request))

    def test_requests_join_while_others_decode_and_leave_when_finished(self):
        engine = self.make_engine()
        a, b, c = self.prompt(5, 1), self.prompt(2, 2), self.prompt(3, 3)
        with self.forwards() as calls:
            first = engine.submit(a, 4, None)
            events, forwards = self.tick(engine, calls)
            self.assertEqual(self.shape(forwards), [('prefill', [5])])
            self.assertEqual([request for request, _ in events], [first])
            second = engine.submit(b, 2, None)
            events, forwards = self.tick(engine, calls)
            self.assertEqual(self.shape(forwards), [('decode', [1]), ('prefill', [2])])
            self.assertEqual([request for request, _ in events], [first, second])
            events, forwards = self.tick(engine, calls)
            self.assertEqual(self.shape(forwards), [('decode', [1, 1])])
            self.assertEqual([request for request, _ in events], [first, second])
            self.assertEqual((engine.result(first).state, engine.result(second).state),
                             (RUNNING, COMPLETED))
            third = engine.submit(c, 1, None)
            events, forwards = self.tick(engine, calls)
            self.assertEqual(self.shape(forwards), [('decode', [1]), ('prefill', [3])])
            self.assertEqual([request for request, _ in events], [first, third])
        self.assertFalse(engine.has_work)
        for request, (prompt, budget) in zip((first, second, third), ((a, 4), (b, 2), (c, 1))):
            self.assertEqual(engine.result(request).state, COMPLETED)
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, budget))
        settle(engine.pool)
        self.assertEqual(engine.pool.num_free, engine.pool.num_pages)

    def test_step_returns_one_event_per_call_and_ticks_only_on_an_empty_buffer(self):
        engine = self.make_engine()
        requests = [engine.submit(self.prompt(n, seed), 2, None) for n, seed in ((3, 4), (1, 5), (4, 6))]
        with self.forwards() as calls:
            first = engine.step()
            self.assertEqual((first[0], len(calls)), (requests[0], 1))
            self.assertTrue(engine.has_work)
            rest = [engine.step(), engine.step()]
            self.assertEqual(len(calls), 1)
            self.assertEqual([event[0] for event in rest], requests[1:])
            decoded = [engine.step()]
            self.assertEqual(self.shape(calls[1:]), [('decode', [1, 1, 1])])
            self.assertEqual({engine.result(r).state for r in requests}, {COMPLETED})
            self.assertTrue(engine.has_work)
            decoded += [engine.step(), engine.step()]
            self.assertEqual(len(calls), 2)
            self.assertFalse(engine.has_work)
            self.assertIsNone(engine.step())
            self.assertEqual(engine.step_batch(), [])
            self.assertEqual(len(calls), 2)
        self.assertEqual([event[0] for event in decoded], requests)
        self.assert_events_match_tokens(engine, [first] + rest + decoded, requests)

    def test_mixing_step_and_step_batch_delivers_each_event_once(self):
        specs = ((3, 5, 7), (2, 3, 8), (5, 4, 9))

        def submitted():
            engine = self.make_engine()
            return engine, [engine.submit(self.prompt(n, seed), budget, None)
                            for n, budget, seed in specs]
        reference, _ = submitted()
        ticks = []
        while reference.has_work:
            ticks.append(reference.step_batch())
        engine, requests = submitted()
        delivered = []
        with self.forwards() as calls:
            delivered.append(engine.step())
            delivered += engine.step_batch()
            self.assertEqual(len(calls), 1)
            self.assertEqual(delivered, ticks[0])
            for index, expected in enumerate(ticks[1:], start=1):
                before = len(calls)
                if index % 2:
                    tick = [engine.step()]
                    self.assertGreater(len(calls), before)
                    while len(tick) < len(expected):
                        tick.append(engine.step())
                else:
                    tick = engine.step_batch()
                self.assertEqual(tick, expected)
                delivered += tick
            forwards = len(calls)
            self.assertFalse(engine.has_work)
            self.assertIsNone(engine.step())
            self.assertEqual(len(calls), forwards)
        self.assertEqual(delivered, [event for tick in ticks for event in tick])
        self.assert_events_match_tokens(engine, delivered, requests)

    def test_has_work_counts_buffered_events_after_requests_finish(self):
        engine = self.make_engine()
        requests = [engine.submit(self.prompt(2, seed), 1, None) for seed in (10, 11)]
        first = engine.step()
        self.assertEqual({engine.result(r).state for r in requests}, {COMPLETED})
        self.assertTrue(engine.has_work)
        second = engine.step()
        self.assertFalse(engine.has_work)
        self.assertEqual([first[0], second[0]], requests)

    def test_active_request_limit_caps_rows_and_admits_in_queue_order(self):
        engine = self.make_engine(max_active_requests=2)
        lengths = (3, 1, 4, 2)
        requests = [engine.submit(self.prompt(n, 20 + n), 2, None) for n in lengths]
        with self.forwards() as calls:
            events, _ = self.drain(engine)
        self.assertEqual(self.shape(calls), [('prefill', [3, 1]), ('decode', [1, 1]),
                                             ('prefill', [4, 2]), ('decode', [1, 1])])
        self.assertEqual([request for request, _ in events],
                         [requests[0], requests[1]] * 2 + [requests[2], requests[3]] * 2)

    def test_token_budget_counts_decode_rows_and_prompt_lengths(self):
        engine = self.make_engine(max_active_requests=4, max_tick_tokens=8)
        prompts = [self.prompt(n, 30 + index) for index, n in enumerate((3, 5, 4, 3))]
        requests = [engine.submit(prompt, 3, None) for prompt in prompts]
        with self.forwards() as calls:
            self.drain(engine)
        self.assertEqual(self.shape(calls), [
            ('prefill', [3, 5]), ('decode', [1, 1]), ('prefill', [4]), ('decode', [1, 1, 1]),
            ('prefill', [3]), ('decode', [1, 1]), ('decode', [1])])
        for request, prompt in zip(requests, prompts):
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, 3))

    def test_prompts_longer_than_the_token_budget_are_rejected_before_enqueueing(self):
        engine = self.make_engine(max_active_requests=2, max_tick_tokens=4)
        self.assertEqual((engine.max_active_requests, engine.max_tick_tokens), (2, 4))
        with self.created_caches() as caches:
            with self.assertRaisesRegex(ValueError, 'token budget'):
                engine.submit(self.prompt(5, 40), 1, None)
            self.assertFalse(engine.has_work)
            self.assertEqual(engine.step_batch(), [])
            accepted = engine.submit(self.prompt(4, 41), 2, None)
            self.drain(engine)
        self.assertEqual(len(caches), 1)
        self.assertEqual(engine.result(accepted).state, COMPLETED)
        default = self.make_engine()
        self.assertEqual(default.max_tick_tokens, self.config.max_position_embeddings)
        self.assertEqual(default.max_active_requests, 8)

    def test_constructor_validates_scheduler_limits_before_preparation(self):
        cases = ({'max_active_requests': 0}, {'max_active_requests': True},
                 {'max_active_requests': 1.5}, {'max_active_requests': None},
                 {'max_tick_tokens': 0}, {'max_tick_tokens': True}, {'max_tick_tokens': 2.0},
                 {'max_active_requests': 4, 'max_tick_tokens': 3})
        for options in cases:
            with (self.subTest(options=options),
                  patch('micro_vllm.engine.prepare_model') as prepare,
                  self.assertRaisesRegex(ValueError, 'max_')):
                self.make_engine(**options)
            prepare.assert_not_called()

    def test_requests_complete_and_cancel_independently(self):
        engine = self.make_engine()
        specs = ((self.prompt(4, 50), 6), (self.prompt(3, 51), 6), (self.prompt(5, 52), 2))
        with self.created_caches() as caches, self.pool_releases() as released:
            a, b, c = (engine.submit(prompt, budget, None) for prompt, budget in specs)
            self.drain_ticks(engine, 2)
            self.assertEqual([engine.result(r).state for r in (a, b, c)],
                             [RUNNING, RUNNING, COMPLETED])
            self.assertIs(engine.cancel(b), CANCELLED)
            self.assertIs(engine.cancel(b), CANCELLED)
            self.assertEqual(self.owned_by(engine.pool, caches[1]._owner), [])
            self.assertEqual(released.count(caches[1]._owner), 1)
            events, _ = self.drain(engine)
        self.assertEqual({request for request, _ in events}, {a})
        self.assertEqual(engine.result(b).tokens, tuple(self.standalone(*specs[1])[:2]))
        self.assertEqual(engine.result(b).error, None)
        for request, (prompt, budget) in ((a, specs[0]), (c, specs[2])):
            self.assertEqual(engine.result(request).state, COMPLETED)
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, budget))
        self.assertEqual([released.count(cache._owner) for cache in caches], [1, 1, 1])

    def test_prefill_exhaustion_leaves_requests_waiting_without_preemption(self):
        engine = self.make_engine(num_pages=4)
        long_prompt, short_prompt = self.prompt(2, 60), self.prompt(4, 61)
        with self.created_caches() as caches, self.forwards() as calls:
            a = engine.submit(long_prompt, 7, None)
            b = engine.submit(short_prompt, 2, None)
            for _ in range(6):
                engine.step_batch()
                self.assertEqual(engine.result(b).state, WAITING)
                self.assertEqual(engine.result(a).state, RUNNING)
            self.assertEqual(len(caches), 1)
            self.assertEqual(self.shape(calls), [('prefill', [2])] + [('decode', [1])] * 5)
            events = engine.step_batch()
            self.assertEqual([request for request, _ in events], [a, b])
            self.drain(engine)
        self.assertTrue(all(call['ok'] for call in calls))
        self.assertEqual([engine.result(r).preemptions for r in (a, b)], [0, 0])
        self.assertEqual(self.tokens(engine, a), self.standalone(long_prompt, 7))
        self.assertEqual(self.tokens(engine, b), self.standalone(short_prompt, 2))

    def test_admission_keeps_decode_headroom_so_new_rows_are_not_preempted(self):
        engine = self.make_engine(num_pages=4)
        first, second = self.prompt(2, 180), self.prompt(2, 181)
        with self.forwards() as calls:
            a = engine.submit(first, 6, None)
            self.drain_ticks(engine, 2)
            b = engine.submit(second, 2, None)
            self.drain(engine)
        self.assertTrue(all(call['ok'] for call in calls))
        self.assertEqual(self.shape(calls), [('prefill', [2])] + [('decode', [1])] * 5
                         + [('prefill', [2]), ('decode', [1])])
        self.assertEqual([engine.result(r).preemptions for r in (a, b)], [0, 0])
        self.assertEqual(self.tokens(engine, a), self.standalone(first, 6))
        self.assertEqual(self.tokens(engine, b), self.standalone(second, 2))

    def test_unexpected_prefill_exhaustion_requeues_rows_and_keeps_decoders(self):
        engine = self.make_engine()
        prompts = [self.prompt(3, 70), self.prompt(2, 71), self.prompt(4, 72)]
        a = engine.submit(prompts[0], 4, None)
        engine.step_batch()
        b, c = engine.submit(prompts[1], 3, None), engine.submit(prompts[2], 3, None)
        real, injected = PreparedModel.forward_batch, []

        def exhaust_prefill_once(prepared, input_ids, caches, **kwargs):
            if caches[0].sequence_length == 0 and not injected:
                injected.append(len(caches))
                raise KVPoolExhausted('injected prefill exhaustion')
            return real(prepared, input_ids, caches, **kwargs)
        with (self.created_caches() as caches,
              patch.object(PreparedModel, 'forward_batch', autospec=True,
                           side_effect=exhaust_prefill_once)):
            events = engine.step_batch()
            self.assertEqual(injected, [2])
            self.assertEqual([request for request, _ in events], [a])
            self.assertEqual([engine.result(r).state for r in (a, b, c)], [RUNNING, WAITING, WAITING])
            self.assertEqual([self.owned_by(engine.pool, cache._owner) for cache in caches], [[], []])
            events = engine.step_batch()
        self.assertEqual([request for request, _ in events], [a, b, c])
        self.drain(engine)
        for request, prompt, budget in zip((a, b, c), prompts, (4, 3, 3)):
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, budget))
            self.assertEqual(engine.result(request).preemptions, 0)

    def test_idle_head_blocked_by_outside_pages_waits_until_they_are_freed(self):
        engine = self.make_engine(num_pages=3)
        blocker = PagedKVCache.for_pool(engine.pool, max_tokens=4).reserve(4)
        starved_prompt = self.prompt(4, 80)
        starved = engine.submit(starved_prompt, 2, None)
        later = engine.submit(self.prompt(1, 81), 1, None)
        with self.forwards() as calls, self.created_caches() as caches:
            for _ in range(3):
                self.assertEqual(engine.step_batch(), [])
            self.assertEqual((calls, caches), ([], []))
        self.assertEqual([engine.result(r).state for r in (starved, later)], [WAITING, WAITING])
        self.assertTrue(engine.has_work)
        blocker.release()
        events, _ = self.drain(engine)
        self.assertEqual([request for request, _ in events], [starved, starved, later])
        self.assertEqual(self.tokens(engine, starved), self.standalone(starved_prompt, 2))

    def test_unexpected_prefill_exhaustion_while_idle_requeues_rows(self):
        engine = self.make_engine()
        prompts = [self.prompt(2, 82), self.prompt(3, 83)]
        requests = [engine.submit(prompt, 2, None) for prompt in prompts]
        with self.failing_forward(1, KVPoolExhausted('injected')):
            self.assertEqual(engine.step_batch(), [])
        self.assertEqual([engine.result(r).state for r in requests], [WAITING, WAITING])
        self.drain(engine)
        for request, prompt in zip(requests, prompts):
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, 2))

    def test_release_failure_while_requeueing_keeps_rows_queued_and_raises_after_the_tick(self):
        engine = self.make_engine()
        prompts = [self.prompt(3, 84), self.prompt(2, 85), self.prompt(4, 86)]
        a = engine.submit(prompts[0], 4, None)
        engine.step_batch()
        b, c = engine.submit(prompts[1], 3, None), engine.submit(prompts[2], 3, None)
        real_release, releases = KVPool._release, []

        def flaky(pool, *args):
            releases.append(args)
            if len(releases) == 1:
                raise RuntimeError('injected requeue release failure')
            return real_release(pool, *args)
        with (self.failing_forward(2, KVPoolExhausted('injected prefill exhaustion')),
              patch.object(KVPool, '_release', autospec=True, side_effect=flaky)):
            with self.assertRaisesRegex(RuntimeError, 'injected requeue release'):
                engine.step_batch()
        self.assertEqual([engine.result(r).state for r in (a, b, c)], [RUNNING, WAITING, WAITING])
        self.assertEqual(len(releases), 2)
        self.assertEqual([request for request, _ in engine.step_batch()], [a])
        events = engine.step_batch()
        self.assertEqual([request for request, _ in events], [a, b, c])
        self.drain(engine)
        for request, prompt, budget in zip((a, b, c), prompts, (4, 3, 3)):
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, budget))

    def test_interrupt_between_emitted_rows_cancels_the_forward_rows(self):
        engine = self.make_engine(max_active_requests=2)
        prompts = [self.prompt(3, 87), self.prompt(2, 88), self.prompt(4, 89)]
        a, b = engine.submit(prompts[0], 5, None), engine.submit(prompts[1], 5, None)
        c = engine.submit(prompts[2], 2, None)
        delivered = engine.step_batch()

        class InterruptFirstAppend(deque):
            interrupted = False

            def append(self, item):
                if not InterruptFirstAppend.interrupted:
                    InterruptFirstAppend.interrupted = True
                    raise KeyboardInterrupt
                super().append(item)
        engine._events = InterruptFirstAppend()
        with self.created_caches() as caches, self.assertRaises(KeyboardInterrupt):
            engine.step_batch()
        self.assertEqual(caches, [])
        self.assertEqual([engine.result(r).state for r in (a, b, c)], [CANCELLED, CANCELLED, WAITING])
        settle(engine.pool)
        self.assertEqual(engine.pool.num_free, engine.pool.num_pages)
        more, _ = self.drain(engine)
        delivered += more
        for request in (a, b):
            tokens = [token for event_id, token in delivered if event_id == request]
            self.assertEqual(tokens, self.tokens(engine, request)[:len(tokens)])
        self.assertEqual(self.tokens(engine, c), self.standalone(prompts[2], 2))

    def test_decode_exhaustion_preempts_newest_first_and_protects_oldest(self):
        engine = self.make_engine(num_pages=16)
        prompts = [self.prompt(2, seed) for seed in (90, 91, 92)]
        late_prompt = self.prompt(2, 93)
        with self.forwards() as calls, self.pool_releases() as released:
            a, b, c = (engine.submit(prompt, 6, None) for prompt in prompts)
            delivered, _ = self.tick(engine, calls)
            blocker = block_free_pages(engine.pool)
            late = engine.submit(late_prompt, 2, None)
            events, forwards = self.tick(engine, calls)
            delivered += events
            self.assertEqual([request for request, _ in events], [a])
            self.assertEqual(self.shape(forwards),
                             [('decode', [1, 1, 1]), ('decode', [1, 1]), ('decode', [1])])
            self.assertEqual([call['ok'] for call in forwards], [False, False, True])
            results = [engine.result(r) for r in (a, b, c, late)]
            self.assertEqual([r.state for r in results], [RUNNING, WAITING, WAITING, WAITING])
            self.assertEqual([r.preemptions for r in results], [0, 1, 1, 0])
            self.assertEqual([len(r.tokens) for r in results], [2, 1, 1, 0])
            self.assertEqual(len(released), 2)
            replays = [prompts[1] + list(results[1].tokens), prompts[2] + list(results[2].tokens)]
            blocker.release()
            events, forwards = self.tick(engine, calls)
            self.assertEqual(self.shape(forwards), [('decode', [1]), ('prefill', [3, 3, 2])])
            self.assertEqual(forwards[1]['rows'], replays + [late_prompt])
            self.assertEqual([request for request, _ in events], [a, b, c, late])
            rest, _ = self.drain(engine)
        self.assert_events_match_tokens(engine, delivered + events + rest, (a, b, c, late))
        for request, prompt, budget in zip((a, b, c, late), prompts + [late_prompt], (6, 6, 6, 2)):
            self.assertEqual(engine.result(request).state, COMPLETED)
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, budget))
        self.assertEqual(engine.result(a).preemptions, 0)

    def test_replay_emits_only_new_tokens_and_matches_standalone(self):
        engine = self.make_engine(num_pages=16)
        prompts = [self.prompt(3, 100), self.prompt(2, 101)]
        delivered = []
        with self.forwards() as calls:
            a, b = (engine.submit(prompt, 7, None) for prompt in prompts)
            delivered += engine.step_batch() + engine.step_batch()
            blocker = block_free_pages(engine.pool)
            delivered += engine.step_batch()
            self.assertEqual(engine.result(b).state, WAITING)
            generated = self.tokens(engine, b)
            blocker.release()
            more, _ = self.drain(engine)
            delivered += more
        replay = [call for call in calls if call['prefill'] and call['rows'][0] != prompts[0]][-1]
        self.assertEqual(replay['rows'], [prompts[1] + generated])
        self.assertEqual(engine.result(b).preemptions, 1)
        self.assert_events_match_tokens(engine, delivered, (a, b))
        for request, prompt in zip((a, b), prompts):
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, 7))

    def test_oversized_replay_runs_alone_in_a_replay_only_tick(self):
        engine = self.make_engine(num_pages=16, max_active_requests=2, max_tick_tokens=4)
        prompts = [self.prompt(3, 110), self.prompt(3, 111)]
        with self.forwards() as calls:
            a, b = (engine.submit(prompt, 6, None) for prompt in prompts)
            ticks = [self.tick(engine, calls) for _ in range(3)]
            self.assertEqual([self.shape(forwards) for _, forwards in ticks], [
                [('prefill', [3])], [('decode', [1]), ('prefill', [3])], [('decode', [1, 1])]])
            blocker = block_free_pages(engine.pool)
            events, forwards = self.tick(engine, calls)
            self.assertEqual([request for request, _ in events], [a])
            self.assertEqual(engine.result(b).state, WAITING)
            generated, a_tokens = self.tokens(engine, b), self.tokens(engine, a)
            self.assertEqual(len(prompts[1] + generated), 5)
            blocker.release()
            events, forwards = self.tick(engine, calls)
            self.assertEqual(self.shape(forwards), [('prefill', [5])])
            self.assertEqual(forwards[0]['rows'], [prompts[1] + generated])
            self.assertEqual([request for request, _ in events], [b])
            self.assertEqual(self.tokens(engine, a), a_tokens)
            events, forwards = self.tick(engine, calls)
            self.assertEqual(self.shape(forwards), [('decode', [1, 1])])
            self.drain(engine)
        for request, prompt in zip((a, b), prompts):
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, 6))

    def test_every_request_finishes_under_a_tight_pool_with_bounded_preemption(self):
        specs = [(length, budget, 120 + index) for index, (length, budget) in
                 enumerate(((5, 8), (1, 9), (6, 5), (3, 9), (2, 7), (4, 6)))]
        pages = max(-(-(length + budget - 1) // self.page_size) for length, budget, _ in specs)
        engine = self.make_engine(num_pages=pages)
        prompts = [self.prompt(length, seed) for length, _, seed in specs]
        requests = [engine.submit(prompt, budget, None) for prompt, (_, budget, _) in zip(prompts, specs)]
        events, ticks = self.drain(engine)
        self.assertLessEqual(ticks, sum(budget for _, budget, _ in specs))
        preemptions = [engine.result(r).preemptions for r in requests]
        self.assertGreater(sum(preemptions), 0)
        self.assert_events_match_tokens(engine, events, requests)
        for request, prompt, (_, budget, _), count in zip(requests, prompts, specs, preemptions):
            self.assertEqual(engine.result(request).state, COMPLETED)
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, budget))
            self.assertLessEqual(count, budget - 1)
        settle(engine.pool)
        self.assertEqual(engine.pool.num_free, pages)

    def test_decode_failure_fails_its_rows_and_the_prefill_phase_still_runs(self):
        engine = self.make_engine(max_active_requests=2)
        prompts = [self.prompt(3, 130), self.prompt(4, 131), self.prompt(2, 132)]
        a, b = engine.submit(prompts[0], 5, None), engine.submit(prompts[1], 5, None)
        c = engine.submit(prompts[2], 2, None)
        engine.step_batch()
        error = RuntimeError('injected decode failure')
        with (self.created_caches() as caches, self.pool_releases() as released,
              self.failing_layer(2, error)):
            events = engine.step_batch()
        self.assertEqual([request for request, _ in events], [c])
        for request in (a, b):
            result = engine.result(request)
            self.assertEqual((result.state, len(result.tokens)), (FAILED, 1))
            self.assertIs(result.error, error)
            self.assertIn('injected decode failure', result.error_traceback)
            self.assertIsNone(result.error.__traceback__)
        self.assertEqual(len(released), 2)
        self.assertNotIn(caches[0]._owner, released)
        self.drain(engine)
        self.assertEqual(self.tokens(engine, c), self.standalone(prompts[2], 2))

    def test_interrupt_during_prefill_cancels_prefill_rows_and_keeps_decoders(self):
        engine = self.make_engine()
        decoding, waiting = self.prompt(3, 140), self.prompt(4, 141)
        a = engine.submit(decoding, 4, None)
        engine.step_batch()
        b = engine.submit(waiting, 3, None)
        layers = self.config.num_hidden_layers
        with (self.created_caches() as caches, self.pool_releases() as released,
              self.failing_layer(layers + 1, KeyboardInterrupt())):
            with self.assertRaises(KeyboardInterrupt):
                engine.step_batch()
        result = engine.result(b)
        self.assertEqual((result.state, result.tokens, result.error), (CANCELLED, (), None))
        self.assertEqual(released, [caches[0]._owner])
        self.assertEqual(engine.result(a).state, RUNNING)
        with self.forwards() as calls:
            events = engine.step_batch()
            self.assertEqual(calls, [])
        self.assertEqual(events, [(a, self.tokens(engine, a)[1])])
        self.drain(engine)
        self.assertEqual(self.tokens(engine, a), self.standalone(decoding, 4))
        self.assertIs(engine.cancel(b), CANCELLED)

    def test_interrupt_during_decode_cancels_decode_rows_and_skips_prefill(self):
        engine = self.make_engine(max_active_requests=2)
        prompts = [self.prompt(3, 150), self.prompt(2, 151), self.prompt(4, 152)]
        a, b = engine.submit(prompts[0], 4, None), engine.submit(prompts[1], 4, None)
        c = engine.submit(prompts[2], 2, None)
        engine.step_batch()
        with self.created_caches() as caches, self.failing_layer(1, KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                engine.step_batch()
        self.assertEqual(caches, [])
        self.assertEqual([engine.result(r).state for r in (a, b, c)], [CANCELLED, CANCELLED, WAITING])
        self.assertEqual([len(engine.result(r).tokens) for r in (a, b)], [1, 1])
        settle(engine.pool)
        self.assertEqual(engine.pool.num_free, engine.pool.num_pages)
        self.drain(engine)
        self.assertEqual(self.tokens(engine, c), self.standalone(prompts[2], 2))

    def test_preemption_release_failure_fails_the_victim_and_keeps_pages_for_retry(self):
        engine = self.make_engine(num_pages=16)
        prompts = [self.prompt(3, 160), self.prompt(2, 161)]
        with self.created_caches() as caches:
            a, b = (engine.submit(prompt, 5, None) for prompt in prompts)
            engine.step_batch()
        blocker = block_free_pages(engine.pool)
        real, calls = KVPool._release, []

        def flaky(pool, *args):
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError('injected release failure')
            return real(pool, *args)
        with patch.object(KVPool, '_release', autospec=True, side_effect=flaky):
            events = engine.step_batch()
            self.assertEqual([request for request, _ in events], [a])
            result = engine.result(b)
            self.assertEqual((result.state, result.preemptions, len(result.tokens)), (FAILED, 0, 1))
            self.assertEqual(str(result.error), 'injected release failure')
            self.assertIsNone(result.error.__traceback__)
            self.assertEqual(len(self.owned_by(engine.pool, caches[1]._owner)), 1)
            self.assertIs(engine.cancel(b), FAILED)
            self.assertEqual(self.owned_by(engine.pool, caches[1]._owner), [])
        blocker.release()
        self.drain(engine)
        self.assertEqual(self.tokens(engine, a), self.standalone(prompts[0], 5))

    def test_interrupt_while_releasing_a_preemption_victim_cancels_it(self):
        engine = self.make_engine(num_pages=16)
        prompts = [self.prompt(3, 200), self.prompt(2, 201)]
        with self.created_caches() as caches:
            a, b = (engine.submit(prompt, 5, None) for prompt in prompts)
            engine.step_batch()
        blocker = block_free_pages(engine.pool)
        real, calls = KVPool._release, []

        def interrupted(pool, *args):
            calls.append(args)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return real(pool, *args)
        with patch.object(KVPool, '_release', autospec=True, side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                engine.step_batch()
            self.assertEqual([engine.result(r).state for r in (a, b)], [RUNNING, CANCELLED])
            self.assertEqual(len(self.owned_by(engine.pool, caches[1]._owner)), 1)
            self.assertIs(engine.cancel(b), CANCELLED)
            self.assertEqual(self.owned_by(engine.pool, caches[1]._owner), [])
        blocker.release()
        self.drain(engine)
        self.assertEqual(self.tokens(engine, a), self.standalone(prompts[0], 5))

    def test_interrupt_while_completing_rows_keeps_their_completed_state(self):
        engine = self.make_engine()
        prompts = [self.prompt(3, 220), self.prompt(2, 221)]
        requests = [engine.submit(prompt, 2, None) for prompt in prompts]
        with self.created_caches() as caches:
            first = engine.step_batch()
        real, calls = PagedKVCache.release, []

        def interrupted(cache):
            calls.append(cache)
            if cache is caches[1] and calls.count(cache) == 1:
                raise KeyboardInterrupt
            return real(cache)
        with patch.object(PagedKVCache, 'release', autospec=True, side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                engine.step_batch()
            self.assertEqual([engine.result(r).state for r in requests], [COMPLETED, COMPLETED])
            self.assertTrue(self.owned_by(engine.pool, caches[1]._owner))
            self.assertIs(engine.cancel(requests[1]), COMPLETED)
        settle(engine.pool)
        self.assertEqual(engine.pool.num_free, engine.pool.num_pages)
        self.assert_events_match_tokens(engine, first + engine.step_batch(), requests)
        for request, prompt in zip(requests, prompts):
            self.assertEqual(self.tokens(engine, request), self.standalone(prompt, 2))

    def test_several_release_failures_in_one_tick_raise_the_first_with_notes(self):
        engine = self.make_engine()
        requests = [engine.submit(self.prompt(2, seed), 1, None) for seed in (210, 211)]
        real, calls = KVPool._release, []

        def flaky(pool, *args):
            calls.append(args)
            if len(calls) <= 2:
                raise RuntimeError(f'injected release failure {len(calls)}')
            return real(pool, *args)
        with patch.object(KVPool, '_release', autospec=True, side_effect=flaky):
            with (warnings.catch_warnings(record=True) as warned,
                  self.assertRaisesRegex(RuntimeError, 'failure 1') as raised):
                warnings.simplefilter('always')
                engine.step_batch()
            # Python 3.10 has no add_note, so cleanup failures arrive as RuntimeWarnings there.
            reported = list(getattr(raised.exception, '__notes__', ()))
            reported += [str(warning.message) for warning in warned]
            self.assertIn('failure 2', ' '.join(reported))
            self.assertEqual([engine.result(r).state for r in requests], [COMPLETED, COMPLETED])
            self.assertEqual(engine.pool.num_free, engine.pool.num_pages - 2)
            for request in requests:
                self.assertIs(engine.cancel(request), COMPLETED)
        settle(engine.pool)
        self.assertEqual(engine.pool.num_free, engine.pool.num_pages)
        self.assertEqual([request for request, _ in engine.step_batch()], requests)
        self.assertFalse(engine.has_work)


class SchedulerCPUTests(SchedulerMixin, unittest.TestCase):
    device = 'cpu'

    @classmethod
    def setUpClass(cls):
        checkpoint = make_qwen()
        cls.weights, cls.config = checkpoint.state_dict(), checkpoint.config


@unittest.skipUnless(GPU_TESTS, 'set RUN_GPU_TESTS=1 for CUDA continuous batching')
class SchedulerCUDAFP16Tests(SchedulerMixin, unittest.TestCase):
    device = 'cuda'
    dtype = torch.float16

    @classmethod
    def setUpClass(cls):
        checkpoint = make_qwen(accelerated=True, device='cuda', dtype=cls.dtype)
        cls.weights, cls.config = checkpoint.state_dict(), checkpoint.config

    def test_eligible_decode_uses_the_batched_native_path(self):
        layers = self.config.num_hidden_layers
        for backend in ('auto', 'cuda'):
            with self.subTest(backend=backend):
                engine = self.make_engine(attention_backend=backend)
                for length, seed in ((5, 1), (2, 2), (7, 3)):
                    engine.submit(self.prompt(length, seed), 4, None)
                engine.step_batch()
                error = AssertionError('eligible batched decode must not gather or run per request')
                with (patch.object(PagedKVCache, 'gather', side_effect=error),
                      patch.object(PagedKVCache, '_gather_buffer', side_effect=error),
                      patch.object(PagedKVBatch, 'gather', side_effect=error),
                      patch('micro_vllm.model.decode_attention_paged', side_effect=error),
                      patch.object(cuda_paged_cache, '_write_paged_kv', side_effect=error),
                      patch.object(cuda_paged_cache, '_write_paged_kv_batch',
                                   wraps=cuda_paged_cache._write_paged_kv_batch) as writes,
                      patch('micro_vllm.model.decode_attention_paged_batch',
                            wraps=cuda_attention.decode_attention_paged_batch) as attention):
                    events = engine.step_batch()
                self.assertEqual(len(events), 3)
                self.assertEqual((writes.call_count, attention.call_count), (layers, layers))
                self.assertEqual(attention.call_args.args[0].shape[0], 3)
        engine = self.make_engine(attention_backend='torch')
        for length, seed in ((5, 1), (2, 2)):
            engine.submit(self.prompt(length, seed), 3, None)
        engine.step_batch()
        with (patch.object(PagedKVBatch, 'gather', autospec=True,
                           side_effect=PagedKVBatch.gather) as gather,
              patch('micro_vllm.model.decode_attention_paged_batch') as attention):
            engine.step_batch()
        self.assertEqual(gather.call_count, layers)
        attention.assert_not_called()

    def test_parity_holds_for_every_backend(self):
        prompts = [self.prompt(n, 170 + n) for n in (5, 1, 7, 3)]
        for backend in ('auto', 'cuda', 'torch'):
            with self.subTest(backend=backend):
                engine = self.make_engine(num_pages=6, attention_backend=backend)
                requests = [engine.submit(prompt, 6, None) for prompt in prompts]
                self.drain(engine)
                self.assertGreater(sum(engine.result(r).preemptions for r in requests), 0)
                for request, prompt in zip(requests, prompts):
                    self.assertEqual(self.tokens(engine, request),
                                     self.standalone(prompt, 6, backend=backend))


class SchedulerCUDAFP32Tests(SchedulerCUDAFP16Tests):
    dtype = torch.float32


@unittest.skipUnless(GPU_TESTS and MODEL_TESTS,
                     'set RUN_GPU_TESTS=1 RUN_MODEL_TESTS=1 for offline real-Qwen engine runs')
class RealQwenEngineFP16Tests(unittest.TestCase):
    dtype, exact_tokens = torch.float16, False
    # The existing real-Qwen FP16 logit tolerance; elements outside it are reported, not asserted.
    tolerance = dict(rtol=5e-3, atol=.125)
    texts = ('What is 2 + 2?', 'Explain a KV cache in one sentence.',
             'List three primary colors and say which one you like most.',
             'Write a haiku about GPUs.')

    @classmethod
    def setUpClass(cls):
        name = 'Qwen/Qwen2.5-0.5B-Instruct'
        cls.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        checkpoint = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=cls.dtype,
            attn_implementation='eager').to('cuda').eval()
        cls.weights, cls.config = checkpoint.state_dict(), checkpoint.config
        del checkpoint

    @classmethod
    def tearDownClass(cls):
        del cls.weights, cls.config, cls.tokenizer

    def ids(self, text):
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=True,
            add_generation_prompt=True, return_tensors='pt')[0].tolist()

    def single_logits(self, prepared, prompt, tokens, page_size):
        pool = KVPool.allocate(num_layers=self.config.num_hidden_layers,
                               num_kv_heads=self.config.num_key_value_heads,
                               num_pages=-(-(len(prompt) + len(tokens)) // page_size),
                               page_size=page_size, head_dim=64, dtype=self.dtype, device='cuda')
        cache = PagedKVCache.for_pool(pool, max_tokens=len(prompt) + len(tokens))
        ids = torch.tensor([prompt + tokens], device='cuda')
        logits = [prepared.forward(ids[:, :len(prompt)], cache=cache, logits_to_keep=1)[0, -1]]
        for position in range(len(prompt), len(prompt) + len(tokens) - 1):
            logits.append(prepared.forward(ids[:, position:position + 1], cache=cache,
                                           logits_to_keep=1)[0, -1])
        cache.release()
        return torch.stack(logits).float().cpu()

    def run_engine(self, prompts, budget, backend, page_size, num_pages):
        engine = Engine(self.weights, self.config, num_pages=num_pages, page_size=page_size,
                        attention_backend=backend)
        captured, real = [], PreparedModel.forward_batch

        def capture(prepared, input_ids, caches, **kwargs):
            logits = real(prepared, input_ids, caches, **kwargs)
            captured.extend(logits.float().cpu())
            return logits
        requests = [engine.submit(prompt, budget, None) for prompt in prompts]
        events = []
        with patch.object(PreparedModel, 'forward_batch', autospec=True, side_effect=capture):
            while engine.has_work:
                events.extend(engine.step_batch())
        self.assertEqual(len(captured), len(events))
        logits = {request: [] for request in requests}
        for (request, token), row in zip(events, captured):
            self.assertEqual(row.argmax().item(), token)
            logits[request].append(row)
        return engine, requests, logits

    def test_engine_matches_standalone_generation_under_batching_and_preemption(self):
        prompts, budget = [self.ids(text) for text in self.texts], 24
        report = []
        for backend in ('auto', 'cuda'):
            prepared = model.prepare_model(self.weights, self.config, attention_backend=backend)
            for page_size in (3, 16):
                admitted = sum(-(-(len(prompt) + 1) // page_size) for prompt in prompts) + 2
                with self.subTest(backend=backend, page_size=page_size):
                    engine, requests, logits = self.run_engine(
                        prompts, budget, backend, page_size, admitted)
                    preemptions = [engine.result(r).preemptions for r in requests]
                    self.assertGreater(sum(preemptions), 0)
                    for request, prompt, text in zip(requests, prompts, self.texts):
                        actual = list(engine.result(request).tokens)
                        expected = generate.generate_tokens(
                            self.weights, self.config, self.tokenizer, text, budget,
                            eos_token_id=None, attention_backend=backend, kv_cache='paged',
                            page_size=page_size).tolist()
                        self.assertEqual(engine.result(request).state, COMPLETED)
                        self.assertEqual(len(actual), budget)
                        if self.exact_tokens:
                            self.assertEqual(actual, expected)
                        reference = self.single_logits(prepared, prompt, expected, page_size)
                        engine_logits = torch.stack(logits[request])
                        divergent = next((i for i, (x, y) in enumerate(zip(actual, expected))
                                          if x != y), None)
                        compared = budget if divergent is None else divergent + 1
                        left, right = engine_logits[:compared], reference[:compared]
                        outside = (~torch.isclose(left, right, **self.tolerance)).sum().item()
                        entry = dict(backend=backend, page_size=page_size, request=request,
                                     preemptions=preemptions[request],
                                     max_abs_logit_error=(left - right).abs().max().item(),
                                     outside_tolerance=f'{outside}/{left.numel()}',
                                     first_divergent_token=divergent)
                        if divergent is not None:
                            entry.update(engine_token=actual[divergent],
                                         standalone_token=expected[divergent],
                                         engine_top2_margin=self.margin(engine_logits[divergent]),
                                         standalone_top2_margin=self.margin(reference[divergent]))
                        report.append(entry)
        print(f'\nreal-Qwen {self.dtype} engine vs standalone paged generation '
              f'(tolerance {self.tolerance}):')
        for entry in report:
            print(entry)

    def margin(self, logits):
        top = logits.topk(2).values
        return (top[0] - top[1]).item()


class RealQwenEngineFP32Tests(RealQwenEngineFP16Tests):
    dtype, exact_tokens = torch.float32, True


if __name__ == '__main__':
    unittest.main()

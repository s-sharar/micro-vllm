"""Single-threaded engine lifecycle: FIFO, cancellation, failure, interruption and page reuse."""
import contextlib
import gc
import os
import unittest
import weakref
from unittest.mock import patch

import torch

from micro_vllm import cuda_attention, engine as engine_module, generate, model
from micro_vllm.engine import Engine, RequestState
from micro_vllm.kv_pool import KVPool, KVPoolExhausted, PageState
from micro_vllm.paged_kv_cache import PagedKVCache
from micro_vllm.runtime import PreparedModel
from test import test_generate as fixtures
from test.test_accelerated_generation import make_qwen, make_tokenizer

GPU_TESTS = os.environ.get('RUN_GPU_TESTS') == '1' and torch.cuda.is_available()
WAITING, RUNNING = RequestState.WAITING, RequestState.RUNNING
COMPLETED, CANCELLED, FAILED = RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED


def drain(engine, limit=100):
    events = []
    while engine.has_work:
        if len(events) == limit:
            raise AssertionError('engine still has work after the step limit')
        events.append(engine.step())
    return events


class EngineMixin:
    backends = ('auto', 'torch')

    def make_engine(self, num_pages=8, page_size=3, attention_backend='auto'):
        return Engine(self.weights, self.config, num_pages=num_pages, page_size=page_size,
                      attention_backend=attention_backend)

    def ids(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors='pt')[0].tolist()

    def expected(self, prompt='hello world', budget=4, eos=None, **options):
        options.setdefault('kv_cache', 'paged')
        return generate.generate_tokens(self.weights, self.config, self.tokenizer, prompt,
                                        budget, eos_token_id=eos, **options).tolist()

    def states(self, pool):
        return [pool.state(page) for page in range(pool.num_pages)]

    def settle(self, pool):
        if pool.device.type == 'cuda':
            torch.cuda.synchronize(pool.device)
        pool.reclaim()

    @contextlib.contextmanager
    def tracked_caches(self):
        caches, real = [], PagedKVCache.for_pool

        def create(pool, **kwargs):
            caches.append(real(pool, **kwargs))
            return caches[-1]
        with (patch.object(PagedKVCache, 'for_pool', side_effect=create),
              patch.object(PagedKVCache, 'release', autospec=True,
                           side_effect=PagedKVCache.release) as release):
            yield caches, release

    def failing_layer(self, failing_call, error):
        real_layer, calls = model.decoder_layer, []

        def layer(*args, **kwargs):
            calls.append(kwargs['layer_idx'])
            if len(calls) == failing_call:
                raise error
            return real_layer(*args, **kwargs)
        return patch('micro_vllm.model.decoder_layer', side_effect=layer)

    def test_idle_step_returns_none_without_forward(self):
        engine = self.make_engine()
        with patch.object(PreparedModel, 'forward', autospec=True) as forward:
            self.assertIsNone(engine.step())
            self.assertFalse(engine.has_work)
        forward.assert_not_called()
        request = engine.submit(self.ids('hello'), 1, None)
        self.assertEqual(drain(engine), [(request, engine.result(request).tokens[0])])
        with patch.object(PreparedModel, 'forward', autospec=True) as forward:
            self.assertIsNone(engine.step())
        forward.assert_not_called()

    def test_fifo_runs_one_request_at_a_time_and_reports_states(self):
        engine = self.make_engine()
        prompts = (('hello', 2), ('hello world', 3), ('hello world answer yes', 1))
        requests = [engine.submit(self.ids(prompt), budget, None) for prompt, budget in prompts]
        self.assertEqual([engine.result(request).state for request in requests], [WAITING] * 3)
        first = engine.step()
        self.assertEqual(first[0], requests[0])
        self.assertEqual([engine.result(request).state for request in requests],
                         [RUNNING, WAITING, WAITING])
        events = [first] + drain(engine)
        self.assertEqual([request for request, _ in events],
                         [requests[0]] * 2 + [requests[1]] * 3 + [requests[2]])
        for request, (prompt, budget) in zip(requests, prompts):
            result = engine.result(request)
            self.assertEqual((result.request_id, result.state, result.error),
                             (request, COMPLETED, None))
            self.assertEqual(list(result.tokens), self.expected(prompt, budget, page_size=3))
            self.assertEqual(list(result.tokens),
                             [token for event_id, token in events if event_id == request])
        self.settle(engine.pool)
        self.assertEqual(self.states(engine.pool), [PageState.FREE] * engine.pool.num_pages)

    def test_tokens_match_generate_tokens_for_each_backend(self):
        cases = (('hello', 1, None), ('hello world', 2, None), ('hello world answer yes', 6, None))
        for backend in self.backends:
            for page_size in (3, 16):
                with self.subTest(backend=backend, page_size=page_size):
                    options = dict(attention_backend=backend, page_size=page_size)
                    stop = self.expected('hello world answer yes', 6, **options)[2]
                    runs = cases + (('hello world answer yes', 6, stop),
                                    ('hello world answer yes', 6, [stop, 0]))
                    engine = self.make_engine(num_pages=-(-11 // page_size), **options)
                    requests = [engine.submit(self.ids(prompt), budget, eos)
                                for prompt, budget, eos in runs]
                    with self.native_path_checks(backend) as checks:
                        drain(engine)
                    for request, (prompt, budget, eos) in zip(requests, runs):
                        result = engine.result(request)
                        self.assertEqual(result.state, COMPLETED)
                        self.assertEqual(list(result.tokens),
                                         self.expected(prompt, budget, eos, **options))
                    checks()

    @contextlib.contextmanager
    def native_path_checks(self, backend):
        yield lambda: None

    def test_eos_and_token_limit_complete_with_final_token_returned(self):
        engine = self.make_engine()
        full = self.expected('hello world answer yes', 6, page_size=3)
        stop = next((token for index, token in enumerate(full)
                     if index and token not in full[:index]), full[0])
        cut = full.index(stop) + 1
        prompt = self.ids('hello world answer yes')
        stopped = engine.submit(prompt, 6, stop)
        limited = engine.submit(prompt, 3, None)
        events = drain(engine)
        self.assertEqual(events, [(stopped, token) for token in full[:cut]]
                         + [(limited, token) for token in full[:3]])
        self.assertEqual(engine.result(stopped).tokens, tuple(full[:cut]))
        self.assertEqual(engine.result(limited).tokens, tuple(full[:3]))
        self.assertEqual({engine.result(stopped).state, engine.result(limited).state}, {COMPLETED})

    def test_oversized_requests_are_rejected_before_enqueueing(self):
        engine = self.make_engine(num_pages=2, page_size=3)
        prompt = self.ids('hello world')
        with patch.object(PagedKVCache, 'for_pool') as create:
            with self.assertRaisesRegex(ValueError, 'pool'):
                engine.submit(prompt, 4, None)
            context = self.config.max_position_embeddings
            large = self.make_engine(num_pages=-(-(context + 1) // 3), page_size=3)
            with self.assertRaisesRegex(ValueError, 'context'):
                large.submit(prompt, context - len(prompt) + 2, None)
        create.assert_not_called()
        self.assertFalse(engine.has_work or large.has_work)
        self.assertIsNone(engine.step())
        accepted = engine.submit(prompt, 3, None)
        drain(engine)
        self.assertEqual(engine.result(accepted).state, COMPLETED)
        accepted = large.submit(prompt, context - len(prompt) + 1, None)
        self.assertEqual(large.result(accepted).state, WAITING)

    def test_cancel_waiting_request_before_allocation(self):
        engine = self.make_engine()
        first = engine.submit(self.ids('hello'), 2, None)
        second = engine.submit(self.ids('hello world'), 2, None)
        third = engine.submit(self.ids('hello world'), 2, None)
        with self.tracked_caches() as (caches, release):
            self.assertIs(engine.cancel(second), CANCELLED)
            self.assertEqual(engine.result(second).tokens, ())
            events = drain(engine)
        self.assertEqual([request for request, _ in events], [first] * 2 + [third] * 2)
        self.assertEqual(len(caches), 2)
        self.assertEqual(release.call_count, 2)
        self.assertEqual(engine.result(second).state, CANCELLED)

    def test_cancel_running_request_between_forwards_releases_once(self):
        engine = self.make_engine()
        running = engine.submit(self.ids('hello world'), 4, None)
        following = engine.submit(self.ids('hello world'), 2, None)
        with self.tracked_caches() as (caches, release):
            token = engine.step()[1]
            self.assertIs(engine.cancel(running), CANCELLED)
            self.assertIs(engine.cancel(running), CANCELLED)
            release.assert_called_once_with(caches[0])
            self.assertEqual(caches[0].page_table, ())
            self.settle(engine.pool)
            self.assertEqual(self.states(engine.pool), [PageState.FREE] * engine.pool.num_pages)
            events = drain(engine)
        self.assertEqual(engine.result(running).tokens, (token,))
        self.assertEqual(engine.result(running).state, CANCELLED)
        self.assertEqual([request for request, _ in events], [following] * 2)
        self.assertEqual(release.call_count, 2)

    def test_unknown_terminal_and_repeated_cancellation(self):
        engine = self.make_engine()
        with self.assertRaises(KeyError):
            engine.cancel(0)
        with self.assertRaises(KeyError):
            engine.result(0)
        done = engine.submit(self.ids('hello'), 1, None)
        waiting = engine.submit(self.ids('hello'), 1, None)
        engine.step()
        with self.tracked_caches() as (_, release):
            self.assertIs(engine.cancel(done), COMPLETED)
            for _ in range(3):
                self.assertIs(engine.cancel(waiting), CANCELLED)
            with self.assertRaises(KeyError):
                engine.cancel(waiting + 1)
        release.assert_not_called()
        self.assertEqual(engine.result(done).state, COMPLETED)
        self.assertFalse(engine.has_work)

    def test_pool_exhaustion_fails_request_and_next_request_runs(self):
        partial, full = self.expected(budget=3, page_size=2), self.expected(page_size=2)
        engine = self.make_engine(num_pages=4, page_size=2)
        blocker = PagedKVCache.for_pool(engine.pool, max_tokens=2).reserve(1)
        starved = engine.submit(self.ids('hello world'), 4, None)
        later = engine.submit(self.ids('hello world'), 4, None)
        with self.tracked_caches() as (caches, release):
            events = [engine.step() for _ in range(4)]
            self.assertIsNone(events[-1])
            result = engine.result(starved)
            self.assertEqual(result.state, FAILED)
            self.assertIsInstance(result.error, KVPoolExhausted)
            self.assertEqual(list(result.tokens), partial)
            release.assert_called_once_with(caches[0])
            blocker.release()
            drain(engine)
        self.assertEqual(engine.result(later).state, COMPLETED)
        self.assertEqual(list(engine.result(later).tokens), full)
        self.settle(engine.pool)
        self.assertEqual(self.states(engine.pool), [PageState.FREE] * 4)

    def test_injected_forward_failure_marks_failed_and_next_request_runs(self):
        engine = self.make_engine()
        failing = engine.submit(self.ids('hello world'), 4, None)
        later = engine.submit(self.ids('hello world'), 4, None)
        error = RuntimeError('injected layer failure')
        with (self.tracked_caches() as (caches, release),
              self.failing_layer(self.config.num_hidden_layers + 1, error)):
            first = engine.step()
            self.assertIsNone(engine.step())
            self.assertEqual(engine.result(failing).state, FAILED)
            self.assertIs(engine.result(failing).error, error)
            self.assertEqual(engine.result(failing).tokens, (first[1],))
            release.assert_called_once_with(caches[0])
            drain(engine)
        self.assertEqual(list(engine.result(later).tokens), self.expected(page_size=3))
        self.assertEqual(release.call_count, 2)

    def test_interruption_cancels_request_releases_pages_and_propagates(self):
        engine = self.make_engine()
        interrupted = engine.submit(self.ids('hello world'), 4, None)
        later = engine.submit(self.ids('hello world'), 4, None)
        with (self.tracked_caches() as (caches, release),
              self.failing_layer(self.config.num_hidden_layers + 1, KeyboardInterrupt())):
            token = engine.step()[1]
            with self.assertRaises(KeyboardInterrupt):
                engine.step()
            release.assert_called_once_with(caches[0])
        result = engine.result(interrupted)
        self.assertEqual((result.state, result.tokens, result.error), (CANCELLED, (token,), None))
        self.assertIs(engine.cancel(interrupted), CANCELLED)
        self.settle(engine.pool)
        self.assertEqual(self.states(engine.pool), [PageState.FREE] * engine.pool.num_pages)
        self.assertEqual(engine.result(later).state, WAITING)
        drain(engine)
        self.assertEqual(list(engine.result(later).tokens), self.expected(page_size=3))

    def test_terminal_requests_drop_their_cache(self):
        engine, refs, real = self.make_engine(), [], PagedKVCache.for_pool

        def create(pool, **kwargs):
            cache = real(pool, **kwargs)
            refs.append(weakref.ref(cache))
            return cache
        requests = [engine.submit(self.ids('hello world'), 3, None) for _ in range(3)]
        with patch.object(PagedKVCache, 'for_pool', side_effect=create):
            engine.step()
            engine.cancel(requests[0])
            with self.failing_layer(1, RuntimeError('injected')):
                engine.step()
            drain(engine)
        gc.collect()
        self.assertEqual([ref() for ref in refs], [None] * 3)
        self.assertEqual([engine.result(r).state for r in requests], [CANCELLED, FAILED, COMPLETED])

    @contextlib.contextmanager
    def flaky_pool_release(self):
        real, calls = KVPool._release, []

        def release(pool, *args):
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError('injected release failure')
            return real(pool, *args)
        with patch.object(KVPool, '_release', autospec=True, side_effect=release):
            yield calls

    def owned(self, pool):
        return [page for page, state in enumerate(self.states(pool)) if state is PageState.OWNED]

    def test_failed_cancel_release_keeps_pages_reachable_for_retry(self):
        expected = self.expected(budget=2, page_size=3)
        engine = self.make_engine()
        running = engine.submit(self.ids('hello world'), 4, None)
        later = engine.submit(self.ids('hello world'), 2, None)
        engine.step()
        with self.flaky_pool_release() as calls:
            with self.assertRaisesRegex(RuntimeError, 'injected release'):
                engine.cancel(running)
            self.assertEqual(engine.result(running).state, CANCELLED)
            self.assertEqual(len(self.owned(engine.pool)), 2)
            for _ in range(2):
                self.assertIs(engine.cancel(running), CANCELLED)
            self.assertEqual(len(calls), 2)
        self.settle(engine.pool)
        self.assertEqual(self.owned(engine.pool), [])
        drain(engine)
        self.assertEqual(list(engine.result(later).tokens), expected)

    def test_failed_completion_release_is_retried_by_cancel(self):
        expected = self.expected(budget=2, page_size=3)
        engine = self.make_engine()
        done = engine.submit(self.ids('hello world'), 2, None)
        later = engine.submit(self.ids('hello world'), 2, None)
        engine.step()
        with self.flaky_pool_release() as calls:
            with self.assertRaisesRegex(RuntimeError, 'injected release'):
                engine.step()
            self.assertEqual(engine.result(done).state, COMPLETED)
            self.assertEqual(list(engine.result(done).tokens), expected)
            self.assertEqual(len(self.owned(engine.pool)), 2)
            self.assertIs(engine.cancel(done), COMPLETED)
            self.assertEqual(len(calls), 2)
        self.settle(engine.pool)
        self.assertEqual(self.owned(engine.pool), [])
        drain(engine)
        self.assertEqual(list(engine.result(later).tokens), expected)

    def test_failed_result_does_not_retain_engine_or_forward_tensors(self):
        device, probes = self.weights['model.embed_tokens.weight'].device, []

        def inner():
            probe = torch.ones(4, device=device)
            probes.append(weakref.ref(probe))
            raise ValueError('inner failure')

        def layer(*args, **kwargs):
            try:
                inner()
            except ValueError:
                raise RuntimeError('outer failure')
        engine = self.make_engine()
        engine_ref, pool_ref = weakref.ref(engine), weakref.ref(engine.pool)
        request = engine.submit(self.ids('hello'), 2, None)
        with patch('micro_vllm.model.decoder_layer', side_effect=layer):
            self.assertIsNone(engine.step())
        result = engine.result(request)
        del engine
        gc.collect()
        self.assertEqual([engine_ref(), pool_ref(), probes[0]()], [None] * 3)
        self.assertEqual(str(result.error), 'outer failure')
        self.assertEqual(str(result.error.__context__), 'inner failure')
        self.assertIsNone(result.error.__traceback__)
        self.assertIsNone(result.error.__context__.__traceback__)
        self.assertIn('inner failure', result.error_traceback)
        self.assertIn('in inner', result.error_traceback)

    def test_next_request_reuses_poisoned_pages(self):
        engine = self.make_engine(num_pages=3, page_size=3)
        expected = self.expected(page_size=3)
        tables = []
        real_forward = PreparedModel.forward

        def capture(prepared, ids, **kwargs):
            result = real_forward(prepared, ids, **kwargs)
            tables.append(kwargs['cache'].page_table)
            return result
        first = engine.submit(self.ids('hello world'), 4, None)
        with patch.object(PreparedModel, 'forward', autospec=True, side_effect=capture):
            drain(engine)
            first_pages = tables[-1]
            self.assertEqual(sorted(first_pages), [0, 1, 2])
            self.settle(engine.pool)
            for buffer in engine.pool._keys + engine.pool._values:
                buffer.fill_(float('nan'))
            second = engine.submit(self.ids('hello world'), 4, None)
            drain(engine)
        self.assertEqual(tables[-1], first_pages)
        for request in (first, second):
            self.assertEqual(list(engine.result(request).tokens), expected)


class EngineCPUTests(EngineMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.GenerationTests.setUpClass.__func__(cls)
        cls.config = cls.model.config

    @classmethod
    def tearDownClass(cls):
        fixtures.GenerationTests.tearDownClass.__func__(cls)

    def test_model_is_prepared_once_with_the_requested_backend(self):
        with patch('micro_vllm.engine.prepare_model',
                   side_effect=engine_module.prepare_model) as prepare:
            engine = self.make_engine(attention_backend='torch')
            for _ in range(3):
                engine.submit(self.ids('hello world'), 2, None)
            drain(engine)
        prepare.assert_called_once()
        self.assertEqual(prepare.call_args.kwargs['attention_backend'], 'torch')

    def test_constructor_options_use_generation_validation(self):
        for options, message in (({'page_size': 0}, 'page_size'), ({'page_size': True}, 'page_size'),
                                 ({'attention_backend': 'flash'}, 'attention_backend'),
                                 ({'attention_backend': None}, 'attention_backend')):
            with (self.subTest(options=options),
                  patch('micro_vllm.engine.prepare_model') as prepare,
                  self.assertRaisesRegex(ValueError, message)):
                self.make_engine(**options)
            prepare.assert_not_called()
            with self.assertRaisesRegex(ValueError, message):
                self.expected(**options)
        for pages in (0, 1.5, True):
            with self.subTest(num_pages=pages), self.assertRaises(ValueError):
                self.make_engine(num_pages=pages)

    def test_cuda_backend_on_cpu_fails_requests_like_generation(self):
        engine = self.make_engine(attention_backend='cuda')
        request = engine.submit(self.ids('hello'), 2, None)
        self.assertIsNone(engine.step())
        self.assertEqual(engine.result(request).state, FAILED)
        self.assertIsInstance(engine.result(request).error, ValueError)
        with self.assertRaisesRegex(ValueError, "attention_backend='cuda'"):
            self.expected(attention_backend='cuda')

    def test_submit_validates_ids_budget_and_eos_before_enqueueing(self):
        engine = self.make_engine()
        cases = (([], 2, None), ('234', 2, None), ([2, 8], 2, None), ([2, -1], 2, None),
                 ([2, True], 2, None), (torch.tensor([[2, 3]]), 2, None),
                 (torch.tensor([2.0, 3.0]), 2, None), ([2, 3], 0, None), ([2, 3], True, None),
                 ([2, 3], 2, 8), ([2, 3], 2, [1, 'x']), ([2, 3], 2, 1.0))
        for prompt, budget, eos in cases:
            with self.subTest(prompt=prompt, budget=budget, eos=eos), self.assertRaises(ValueError):
                engine.submit(prompt, budget, eos)
        self.assertFalse(engine.has_work)
        tensor_request = engine.submit(torch.tensor(self.ids('hello')), 1, ())
        list_request = engine.submit(tuple(self.ids('hello')), 1, None)
        drain(engine)
        self.assertEqual(engine.result(tensor_request).tokens, engine.result(list_request).tokens)
        self.assertEqual(list_request, tensor_request + 1)

    def test_result_is_an_immutable_snapshot(self):
        engine = self.make_engine()
        request = engine.submit(self.ids('hello world'), 3, None)
        engine.step()
        snapshot = engine.result(request)
        drain(engine)
        self.assertEqual((snapshot.state, len(snapshot.tokens)), (RUNNING, 1))
        self.assertEqual(len(engine.result(request).tokens), 3)
        with self.assertRaises(AttributeError):
            snapshot.state = COMPLETED


@unittest.skipUnless(GPU_TESTS, 'set RUN_GPU_TESTS=1 for CUDA engine tests')
class EngineCUDATests(EngineMixin, unittest.TestCase):
    backends = ('auto', 'cuda', 'torch')

    @classmethod
    def setUpClass(cls):
        checkpoint = make_qwen(accelerated=True, device='cuda', dtype=torch.float16)
        cls.weights, cls.config = checkpoint.state_dict(), checkpoint.config
        cls.tokenizer = make_tokenizer()

    @contextlib.contextmanager
    def native_path_checks(self, backend):
        with (patch.object(PagedKVCache, 'gather', autospec=True,
                           side_effect=PagedKVCache.gather) as gather,
              patch('micro_vllm.model.decode_attention_paged',
                    wraps=cuda_attention.decode_attention_paged) as native):
            def check():
                if backend == 'torch':
                    native.assert_not_called()
                    self.assertGreater(gather.call_count, 0)
                else:
                    gather.assert_not_called()
                    self.assertGreater(native.call_count, 0)
            yield check

    def test_steps_and_cache_work_run_on_the_construction_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            engine = self.make_engine()
        seen, real_forward = [], PreparedModel.forward

        def capture(prepared, ids, **kwargs):
            seen.append(torch.cuda.current_stream())
            return real_forward(prepared, ids, **kwargs)
        running = engine.submit(self.ids('hello world'), 4, None)
        later = engine.submit(self.ids('hello world'), 4, None)
        with (self.tracked_caches() as (caches, release),
              patch.object(PreparedModel, 'forward', autospec=True, side_effect=capture)):
            engine.step()
            engine.cancel(running)
            drain(engine)
        self.assertEqual(seen, [stream] * 5)
        self.assertEqual([cache._stream for cache in caches], [stream, stream])
        self.assertEqual(release.call_count, 2)
        self.assertEqual(list(engine.result(later).tokens), self.expected(page_size=3))

    def test_completed_pages_retire_until_stream_work_ends_then_are_reused(self):
        engine = self.make_engine(num_pages=3, page_size=3)
        engine.submit(self.ids('hello'), 2, None)
        drain(engine)
        self.settle(engine.pool)
        real_release, real_forward, tables = PagedKVCache.release, PreparedModel.forward, []

        def delayed_release(cache):
            torch.cuda._sleep(400_000_000)
            real_release(cache)

        def capture(prepared, ids, **kwargs):
            result = real_forward(prepared, ids, **kwargs)
            tables.append(kwargs['cache'].page_table)
            return result
        first = engine.submit(self.ids('hello world'), 4, None)
        second = engine.submit(self.ids('hello world'), 4, None)
        with (patch.object(PagedKVCache, 'release', autospec=True, side_effect=delayed_release),
              patch.object(PreparedModel, 'forward', autospec=True, side_effect=capture)):
            while engine.result(first).state in (WAITING, RUNNING):
                engine.step()
            self.assertEqual(engine.result(first).state, COMPLETED)
            self.assertEqual(self.states(engine.pool), [PageState.RETIRING] * 3)
            first_pages = tables[-1]
            drain(engine)
        self.assertEqual(tables[-1], first_pages)
        expected = self.expected(page_size=3)
        self.assertEqual([list(engine.result(r).tokens) for r in (first, second)], [expected] * 2)


if __name__ == '__main__':
    unittest.main()

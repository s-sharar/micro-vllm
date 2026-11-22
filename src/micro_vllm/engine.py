"""Continuous-batching request engine over one prepared model and one owned KVPool.

submit() rejects, before queueing, a request whose KV writes (prompt +
max_new_tokens - 1) exceed the pool or model context, or whose prompt exceeds
max_tick_tokens. Each tick runs one decode batch over the active requests, then
one prefill batch of waiting requests admitted in queue order while they fit
max_active_requests, the token budget (one token per decode row, the whole
prefill per prefill row) and free plus retiring pages with one decode page of
headroom for every request that continues. Only decode exhaustion preempts:
the newest active request releases its pages, waits at the queue front and
resumes by prefilling prompt + generated tokens. The oldest active request is
never preempted; its KV requirement fits the pool by submit()'s check, so it
can reserve once newer requests are gone unless pages are held outside the
engine. Requests that cannot reserve a prefill wait. A replay longer than
max_tick_tokens runs alone in a replay-only tick.

The engine owns each request's PagedKVCache from admission until preemption
or a terminal state, and releases it once (on CUDA, on the construction
stream). step() and step_batch() share one event buffer, so each token event is
delivered exactly once. A forward's Exception fails its rows; a BaseException
cancels them and propagates. Single host thread only.
"""
import contextlib
import itertools
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch

from .attention_backend import validate_attention_backend
from .generate import _cache_dimensions, _eos_ids, _validate_cache_options
from .kv_pool import KVPool, KVPoolExhausted
from .paged_kv_batch import _report_cleanup_failure
from .paged_kv_cache import PagedKVCache
from .runtime import prepare_model


class RequestState(Enum):
    WAITING = 'waiting'
    RUNNING = 'running'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'
    FAILED = 'failed'


TERMINAL_STATES = frozenset({RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED})


@dataclass(frozen=True)
class RequestResult:
    """Snapshot of one request; FAILED keeps the exception and its formatted traceback text."""

    request_id: int
    state: RequestState
    tokens: tuple[int, ...]
    error: Exception | None = None
    error_traceback: str | None = None
    preemptions: int = 0


def _detached(error):
    # Traceback frames would pin the engine, its pool and the failed forward's tensors.
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        pending.extend((current.__cause__, current.__context__,
                        *getattr(current, 'exceptions', ())))
    return error


def _failure(error):
    text = ''.join(traceback.format_exception(error))
    return _detached(error), text


@dataclass(eq=False)
class _Request:
    request_id: int
    prompt: tuple[int, ...]
    max_new_tokens: int
    eos_ids: tuple[int, ...]
    state: RequestState = RequestState.WAITING
    tokens: list[int] = field(default_factory=list)
    error: Exception | None = None
    error_traceback: str | None = None
    cache: PagedKVCache | None = None
    next_input: torch.Tensor | None = None
    preemptions: int = 0

    @property
    def kv_tokens(self):
        return len(self.prompt) + self.max_new_tokens - 1

    @property
    def prefill_tokens(self):
        """Its next prefill replays the prompt plus every token generated so far."""
        return len(self.prompt) + len(self.tokens)


class Engine:
    """FIFO admission into continuous batches; cache options validate like generate_tokens()."""

    def __init__(self, weights, config, *, num_pages, page_size=16, attention_backend='auto',
                 max_active_requests=8, max_tick_tokens=None):
        _validate_cache_options('paged', page_size, True)
        validate_attention_backend(attention_backend)
        if type(max_active_requests) is not int or max_active_requests <= 0:
            raise ValueError('max_active_requests must be a positive integer')
        if max_tick_tokens is None:
            max_tick_tokens = max(config.max_position_embeddings, max_active_requests)
        if type(max_tick_tokens) is not int or max_tick_tokens < max_active_requests:
            raise ValueError('max_tick_tokens must be an integer of at least max_active_requests')
        self._prepared = prepare_model(weights, config, attention_backend=attention_backend)
        self._pool = KVPool.allocate(**_cache_dimensions(weights, config),
                                     num_pages=num_pages, page_size=page_size)
        self._vocab_size = config.vocab_size
        self._max_active, self._max_tick_tokens = max_active_requests, max_tick_tokens
        self._stream = (torch.cuda.current_stream(self._pool.device)
                        if self._pool.device.type == 'cuda' else None)
        self._requests, self._waiting, self._active = {}, deque(), []
        self._events, self._release_errors = deque(), []
        self._ids = itertools.count()

    @property
    def pool(self):
        return self._pool

    @property
    def max_active_requests(self):
        return self._max_active

    @property
    def max_tick_tokens(self):
        return self._max_tick_tokens

    @property
    def has_work(self):
        return bool(self._active or self._waiting or self._events)

    def _checked_prompt(self, prompt_token_ids):
        if isinstance(prompt_token_ids, torch.Tensor):
            if prompt_token_ids.ndim != 1 or prompt_token_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError('prompt tensor must hold one-dimensional integer token IDs')
            prompt_token_ids = prompt_token_ids.tolist()
        if not isinstance(prompt_token_ids, (list, tuple)):
            raise ValueError('prompt_token_ids must be a list, tuple or 1-D integer tensor')
        prompt = tuple(prompt_token_ids)
        if not prompt or any(type(token) is not int or not 0 <= token < self._vocab_size
                             for token in prompt):
            raise ValueError('prompt_token_ids must be nonempty IDs within the vocabulary')
        return prompt

    def submit(self, prompt_token_ids, max_new_tokens, eos_ids):
        """Validate and queue one request; return its ID. Oversized requests are never queued."""
        prompt = self._checked_prompt(prompt_token_ids)
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be a positive integer')
        eos = _eos_ids(eos_ids, self._vocab_size)
        written = len(prompt) + max_new_tokens - 1
        if written > self._prepared.max_position_embeddings:
            raise ValueError('request would require a forward beyond the configured context length')
        if written > self._pool.token_capacity:
            raise ValueError(f'request needs {written} KV tokens but the whole pool holds '
                             f'{self._pool.token_capacity}')
        if len(prompt) > self._max_tick_tokens:
            raise ValueError(f'prompt has {len(prompt)} tokens but the per-tick token budget is '
                             f'{self._max_tick_tokens}')
        request = _Request(next(self._ids), prompt, max_new_tokens, eos)
        self._requests[request.request_id] = request
        self._waiting.append(request)
        return request.request_id

    def result(self, request_id):
        """Return a snapshot of the request's state, tokens and error; KeyError if unknown."""
        request = self._requests[request_id]
        return RequestResult(request.request_id, request.state, tuple(request.tokens),
                             request.error, request.error_traceback, request.preemptions)

    def cancel(self, request_id):
        """Cancel a waiting or running request; terminal requests keep their state.

        Already buffered events of a cancelled request are still delivered. A
        terminal request whose cache release previously raised retries that release.
        """
        request = self._requests[request_id]
        if request.state in TERMINAL_STATES:
            self._release(request)
            return request.state
        if request.state is RequestState.WAITING:
            self._waiting.remove(request)
        self._finish(request, RequestState.CANCELLED)
        return request.state

    @torch.inference_mode()
    def step_batch(self):
        """Return the buffered rest of a tick, otherwise run one tick; [] if it emitted nothing.

        Events are (request_id, token): decode rows, then prefill rows, each in
        admission order.
        """
        if not self._events:
            self._tick()
        events = list(self._events)
        self._events.clear()
        return events

    @torch.inference_mode()
    def step(self):
        """Return one buffered event, running a tick only when none is left; None if no token."""
        if not self._events:
            self._tick()
        return self._events.popleft() if self._events else None

    def _on_stream(self):
        if self._stream is None:
            return contextlib.nullcontext()
        return torch.cuda.stream(self._stream)

    def _tick(self):
        with self._on_stream():
            try:
                if self._replay_alone():
                    self._prefill([self._waiting.popleft()])
                else:
                    decoded = self._decode()
                    self._prefill(self._admit(self._max_tick_tokens - decoded))
            except BaseException as error:
                self._raise_release_errors(error)
                raise
            self._raise_release_errors()

    def _raise_release_errors(self, propagating=None):
        errors, self._release_errors = self._release_errors, []
        if not errors:
            return
        primary = errors[0] if propagating is None else propagating
        for error in errors:
            if error is not primary:
                _report_cleanup_failure(primary, f'releasing a finished request failed: {error!r}; '
                                                 'call cancel() on it to retry')
        if propagating is None:
            raise primary

    def _pages(self, tokens):
        return -(-tokens // self._pool.page_size)

    def _prefill_pages(self, request):
        """Pages for its prefill plus, when it may continue, its first decode token."""
        continues = len(request.tokens) + 1 < request.max_new_tokens
        return self._pages(request.prefill_tokens + continues)

    def _free_after_decode_headroom(self):
        growth = sum(self._pages(request.cache.sequence_length + 1) - len(request.cache.page_table)
                     for request in self._active)
        return self._pool.num_available - growth

    def _replay_alone(self):
        """Whether the queue head is a replay over the token budget that may run now, alone."""
        head = self._waiting[0] if self._waiting else None
        if (head is None or head.prefill_tokens <= self._max_tick_tokens
                or len(self._active) >= self._max_active):
            return False
        return self._prefill_pages(head) <= self._free_after_decode_headroom()

    def _admit(self, budget):
        """Pop the longest queue prefix that fits the active limit, token budget and pages."""
        rows, pages = [], self._free_after_decode_headroom()
        while self._waiting and len(self._active) + len(rows) < self._max_active:
            head = self._waiting[0]
            needed = self._prefill_pages(head)
            if head.prefill_tokens > budget or needed > pages:
                break
            rows.append(self._waiting.popleft())
            budget, pages = budget - head.prefill_tokens, pages - needed
        return rows

    def _decode(self):
        """Decode every active request, preempting the newest on exhaustion; return rows run."""
        rows = list(self._active)
        while rows:
            try:
                self._forward(rows)
            except KVPoolExhausted as error:
                if len(rows) > 1:
                    self._preempt(rows.pop())
                    continue
                self._fail(rows, error)
            except Exception as error:
                self._fail(rows, error)
            break
        return len(rows)

    def _prefill(self, rows):
        if not rows:
            return
        for row in rows:
            row.state = RequestState.RUNNING
        try:
            self._forward(rows, admit=True)
        except KVPoolExhausted:
            self._requeue(rows)
        except Exception as error:
            self._fail(rows, error)

    def _requeue(self, rows):
        """Return rows whose prefill could not reserve to the queue front, then drop their caches."""
        caches = [row.cache for row in rows if row.cache is not None]
        for row in reversed(rows):
            row.state, row.cache, row.next_input = RequestState.WAITING, None, None
            self._waiting.appendleft(row)
        for cache in caches:
            try:
                with self._on_stream():
                    cache.release()
            except Exception as error:
                self._release_errors.append(error)

    def _forward(self, rows, *, admit=False):
        """Run one forward_batch and emit its tokens; a BaseException cancels unfinished rows.

        Emission is inside the guard because the forward already advanced every row's cache.
        """
        try:
            for row in rows:
                if row.cache is None:
                    row.cache = PagedKVCache.for_pool(self._pool, max_tokens=row.kv_tokens)
                    row.next_input = torch.tensor(row.prompt + tuple(row.tokens), dtype=torch.int64,
                                                  device=self._pool.device)
            logits = self._prepared.forward_batch([row.next_input for row in rows],
                                                  [row.cache for row in rows])
            ids = logits.argmax(dim=-1)
            tokens = ids.tolist()
            if admit:
                self._active.extend(rows)
            self._emit(rows, ids, tokens)
        except BaseException as error:
            if not isinstance(error, Exception):
                self._finish_rows(rows, RequestState.CANCELLED)
            raise

    def _emit(self, rows, device_ids, tokens):
        finished = []
        for index, (row, token) in enumerate(zip(rows, tokens)):
            row.tokens.append(token)
            row.next_input = device_ids[index:index + 1]
            self._events.append((row.request_id, token))
            if len(row.tokens) >= row.max_new_tokens or token in row.eos_ids:
                finished.append(row)
        self._finish_rows(finished, RequestState.COMPLETED)

    def _preempt(self, request):
        """Release the newest request's pages and queue it first; a failed release fails it."""
        self._active.remove(request)
        request.next_input = None
        try:
            self._release(request)
        except Exception as error:
            request.state = RequestState.FAILED
            request.error, request.error_traceback = _failure(error)
            return
        except BaseException:
            request.state = RequestState.CANCELLED
            raise
        request.state = RequestState.WAITING
        request.preemptions += 1
        self._waiting.appendleft(request)

    def _fail(self, rows, error):
        error, text = _failure(error)
        self._finish_rows(rows, RequestState.FAILED, error, text)

    def _finish_rows(self, rows, state, error=None, text=None):
        for row in rows:
            if row.state in TERMINAL_STATES:
                continue
            try:
                self._finish(row, state, error, text)
            except Exception as release_error:
                self._release_errors.append(release_error)

    def _finish(self, request, state, error=None, text=None):
        request.state, request.error, request.error_traceback = state, error, text
        request.next_input = None
        if request in self._active:
            self._active.remove(request)
        self._release(request)

    def _release(self, request):
        if request.cache is not None:
            with self._on_stream():
                request.cache.release()
            request.cache = None

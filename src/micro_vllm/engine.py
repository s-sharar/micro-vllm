"""Single-threaded request lifecycle over one prepared model and one owned KVPool.

The engine prepares the model once and allocates its pool. submit() takes
token IDs, rejects a request whose KV writes (prompt + max_new_tokens - 1)
exceed the whole pool or the model context, and queues it FIFO. step() runs
at most one forward for the single active request: the whole prompt first,
then one decode token per step. A request owns a PagedKVCache from its first
forward until it reaches a terminal state, which releases that cache exactly
once; CUDA pages then retire behind the engine stream before reuse.

WAITING -> RUNNING -> COMPLETED | CANCELLED | FAILED; waiting requests may be
cancelled directly. An Exception from a forward, including KVPoolExhausted,
fails only that request. A BaseException cancels the active request, releases
its cache and propagates. On CUDA every step and release runs on the stream
current at construction. Recovery covers host-side exceptions; asynchronous
CUDA faults can invalidate the context. Results are kept for the engine's
lifetime. Single host thread only.
"""
import contextlib
import itertools
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch

from .attention_backend import validate_attention_backend
from .generate import _cache_dimensions, _eos_ids, _validate_cache_options, select_next_token
from .kv_pool import KVPool
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
    """Snapshot of one request; error is set only for FAILED, with frame locals cleared."""

    request_id: int
    state: RequestState
    tokens: tuple[int, ...]
    error: Exception | None = None


@dataclass
class _Request:
    request_id: int
    prompt: tuple[int, ...]
    max_new_tokens: int
    eos_ids: tuple[int, ...]
    state: RequestState = RequestState.WAITING
    tokens: list[int] = field(default_factory=list)
    error: Exception | None = None
    cache: PagedKVCache | None = None
    next_input: torch.Tensor | None = None

    @property
    def kv_tokens(self):
        return len(self.prompt) + self.max_new_tokens - 1


class Engine:
    """FIFO queue plus one active request; options validate like generate_tokens()."""

    def __init__(self, weights, config, *, num_pages, page_size=16, attention_backend='auto'):
        _validate_cache_options('paged', page_size, True)
        validate_attention_backend(attention_backend)
        self._prepared = prepare_model(weights, config, attention_backend=attention_backend)
        self._pool = KVPool.allocate(**_cache_dimensions(weights, config),
                                     num_pages=num_pages, page_size=page_size)
        self._vocab_size = config.vocab_size
        self._stream = (torch.cuda.current_stream(self._pool.device)
                        if self._pool.device.type == 'cuda' else None)
        self._requests, self._waiting, self._active = {}, deque(), None
        self._ids = itertools.count()

    @property
    def pool(self):
        return self._pool

    @property
    def has_work(self):
        return self._active is not None or bool(self._waiting)

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
        request = _Request(next(self._ids), prompt, max_new_tokens, eos)
        self._requests[request.request_id] = request
        self._waiting.append(request)
        return request.request_id

    def result(self, request_id):
        """Return a snapshot of the request's state, tokens and error; KeyError if unknown."""
        request = self._requests[request_id]
        return RequestResult(request.request_id, request.state, tuple(request.tokens),
                             request.error)

    def cancel(self, request_id):
        """Cancel a waiting or running request; terminal requests keep their state."""
        request = self._requests[request_id]
        if request.state in TERMINAL_STATES:
            return request.state
        if request.state is RequestState.WAITING:
            self._waiting.remove(request)
        self._finish(request, RequestState.CANCELLED)
        return request.state

    @torch.inference_mode()
    def step(self):
        """Run at most one forward; return (request_id, token), or None if idle or failed."""
        with self._on_stream():
            request = self._activate()
            if request is None:
                return None
            # Only a BaseException escapes the forward with this initial state.
            state, error = RequestState.CANCELLED, None
            try:
                token, finished = self._forward(request)
                state = RequestState.COMPLETED if finished else None
                return request.request_id, token
            except Exception as caught:
                # A stored traceback would otherwise pin the failed forward's cache and tensors.
                traceback.clear_frames(caught.__traceback__)
                state, error = RequestState.FAILED, caught
                return None
            finally:
                if state is not None:
                    self._finish(request, state, error)

    def _on_stream(self):
        if self._stream is None:
            return contextlib.nullcontext()
        return torch.cuda.stream(self._stream)

    def _activate(self):
        if self._active is None and self._waiting:
            self._active = self._waiting.popleft()
            self._active.state = RequestState.RUNNING
        return self._active

    def _forward(self, request):
        if request.cache is None:
            request.cache = PagedKVCache.for_pool(self._pool, max_tokens=request.kv_tokens)
            request.next_input = torch.tensor([request.prompt], dtype=torch.int64,
                                              device=self._pool.device)
        logits = self._prepared.forward(request.next_input, cache=request.cache, logits_to_keep=1)
        token, finished = select_next_token(
            logits, generated=len(request.tokens), max_new_tokens=request.max_new_tokens,
            eos_ids=request.eos_ids)
        request.next_input = token
        request.tokens.append(token.item())
        return request.tokens[-1], finished

    def _finish(self, request, state, error=None):
        request.state, request.error = state, error
        cache, request.cache, request.next_input = request.cache, None, None
        if request is self._active:
            self._active = None
        if cache is not None:
            with self._on_stream():
                cache.release()

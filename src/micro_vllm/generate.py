"""Greedy generation using our own cached Qwen forward pass.

Run in WSL:
    source .venv/bin/activate
    python -m pip install -e .
    python -m micro_vllm.generate --prompt 'Explain a KV cache in one sentence.'

Hugging Face loads the tokenizer and pretrained tensors. Every generation
forward uses model.py with a fresh request cache. Contiguous storage remains
the default; --kv-cache paged grows a request page table over a KVPool, either
caller-supplied (kv_pool) or a per-call pool sized for this request. Pages are
released when generation returns or fails.
Paged auto uses Flash prefill and direct native CUDA decode when eligible;
torch and CPU/BF16 auto use the reference attention fallback.
Single request, no padding or sampling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .kv_cache import ContiguousKVCache
from .kv_pool import KVPool
from .paged_kv_cache import PagedKVCache
from .attention_backend import validate_attention_backend
from .model import forward, prepare_model
from .reference import parse_args


def _eos_ids(eos_token_id, vocab_size):
    """Normalize stop IDs; None or an empty sequence disables EOS stopping."""
    if eos_token_id is None:
        return ()
    ids = (eos_token_id,) if isinstance(eos_token_id, int) else eos_token_id
    if (not isinstance(ids, (tuple, list))
            or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)):
        raise ValueError('EOS must be an integer ID, a list/tuple of valid IDs, or None')
    return tuple(ids)


def select_next_token(logits, *, generated, max_new_tokens, eos_ids):
    """Greedy-pick the last position; return ([1, 1] token, whether the request is finished).

    generated counts tokens selected before this one; the final token counts toward the budget.
    """
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finished = (generated + 1 >= max_new_tokens
                or bool(eos_ids) and next_token.item() in eos_ids)
    return next_token, finished


def generate_tokens(weights, config, tokenizer, prompt, max_new_tokens, *, eos_token_id,
                    use_cache=True, attention_backend='auto', kv_cache='contiguous', page_size=16,
                    kv_pool=None):
    """Format one chat request and return only new token IDs, including generated EOS."""
    _validate_cache_options(kv_cache, page_size, use_cache)
    _validate_pool(kv_pool, weights, config, kv_cache, page_size)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt must contain text')
    input_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors='pt',
    ).to(weights['model.embed_tokens.weight'].device)
    return _greedy_decode(input_ids, weights, config, max_new_tokens,
                          eos_token_id=eos_token_id, use_cache=use_cache,
                          attention_backend=attention_backend, kv_cache=kv_cache,
                          page_size=page_size, kv_pool=kv_pool)


def _validate_cache_options(kv_cache, page_size, use_cache):
    if not isinstance(kv_cache, str) or kv_cache not in ('contiguous', 'paged'):
        raise ValueError('kv_cache must be contiguous or paged')
    if type(page_size) is not int or page_size <= 0:
        raise ValueError('page_size must be a positive integer')
    if kv_cache == 'paged' and not use_cache:
        raise ValueError('kv_cache=paged requires use_cache=True')


def _cache_dimensions(weights, config):
    embedding = weights['model.embed_tokens.weight']
    return dict(
        num_layers=config.num_hidden_layers, num_kv_heads=config.num_key_value_heads,
        head_dim=getattr(config, 'head_dim', None) or (
            config.hidden_size // config.num_attention_heads),
        dtype=embedding.dtype, device=embedding.device,
    )


def _validate_pool(kv_pool, weights, config, kv_cache, page_size):
    if kv_pool is None:
        return
    if not isinstance(kv_pool, KVPool):
        raise ValueError('kv_pool must be a KVPool or None')
    if kv_cache != 'paged':
        raise ValueError('kv_pool requires kv_cache=paged')
    dimensions = _cache_dimensions(weights, config)
    expected = (*dimensions.values(), page_size)
    actual = (kv_pool.num_layers, kv_pool.num_kv_heads, kv_pool.head_dim,
              kv_pool.dtype, kv_pool.device, kv_pool.page_size)
    if actual != expected:
        raise ValueError('kv_pool layers, KV heads, head dimension, dtype, device and '
                         'page size must match the model and page_size')


def _allocate_cache(weights, config, capacity, kv_cache, page_size, kv_pool=None):
    dimensions = _cache_dimensions(weights, config)
    if kv_cache == 'paged':
        if kv_pool is None:
            kv_pool = KVPool.allocate(**dimensions, page_size=page_size,
                                      num_pages=(capacity + page_size - 1) // page_size)
        return PagedKVCache.for_pool(kv_pool, max_tokens=capacity)
    return ContiguousKVCache.allocate(**dimensions, batch_size=1, capacity=capacity)


@torch.inference_mode()
def _greedy_decode(input_ids, weights, config, max_new_tokens, *, eos_token_id,
                   use_cache=True, attention_backend='auto', kv_cache='contiguous', page_size=16,
                   kv_pool=None):
    """Prefill once, then decode single tokens; use_cache=False verifies full prefixes."""
    _validate_cache_options(kv_cache, page_size, use_cache)
    _validate_pool(kv_pool, weights, config, kv_cache, page_size)
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('generation requires one nonempty sequence [1, tokens]')
    written = input_ids.shape[1] + max_new_tokens - 1
    if written > config.max_position_embeddings:
        raise ValueError('generation would require a forward beyond the configured context length')
    if kv_pool is not None and written > kv_pool.token_capacity:
        raise ValueError(f'request needs {written} KV tokens but the whole kv_pool holds '
                         f'{kv_pool.token_capacity}')
    eos_ids = _eos_ids(eos_token_id, config.vocab_size)
    validate_attention_backend(attention_backend)
    prepared = prepare_model(weights, config, attention_backend=attention_backend)
    cache = None
    if use_cache:
        cache = _allocate_cache(weights, config, written, kv_cache, page_size, kv_pool)
    try:
        return _decode_loop(input_ids, prepared, cache, max_new_tokens, eos_ids)
    finally:
        if isinstance(cache, PagedKVCache):
            cache.release()


def _decode_loop(input_ids, prepared, cache, max_new_tokens, eos_ids):
    generated = []
    for step in range(max_new_tokens):
        logits = forward(input_ids, prepared, cache=cache, logits_to_keep=1)
        next_token, finished = select_next_token(
            logits, generated=step, max_new_tokens=max_new_tokens, eos_ids=eos_ids)
        generated.append(next_token)
        if finished:
            break
        input_ids = next_token if cache is not None else torch.cat((input_ids, next_token), dim=1)
    return torch.cat(generated, dim=1)[0]


def main(argv=None):
    args = parse_args(argv, description=__doc__, with_attention_backend=True,
                      with_kv_cache=True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Check WSL GPU setup or use --device cpu.')
    print(f'Loading {args.model} on {args.device} ...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.device == 'cuda' else torch.float32
    checkpoint = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation='eager',
    ).to(args.device).eval()
    weights, config = checkpoint.state_dict(), checkpoint.config
    eos = checkpoint.generation_config.eos_token_id
    del checkpoint  # The state dict retains the weight storage; no copy is needed.
    new_tokens = generate_tokens(weights, config, tokenizer, args.prompt,
                                 args.max_new_tokens, eos_token_id=eos,
                                 attention_backend=args.attention_backend,
                                 kv_cache=args.kv_cache, page_size=args.page_size)
    print(f'Prompt: {args.prompt}')
    print(f'Generated tokens: {new_tokens.numel()}')
    print(f'Token IDs: {new_tokens.tolist()}')
    print('Response:')
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == '__main__':
    main()

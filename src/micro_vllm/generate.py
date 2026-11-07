"""Greedy generation using our own cached Qwen forward pass.

Run in WSL:
    source .venv/bin/activate
    python -m pip install -e .
    python -m micro_vllm.generate --prompt 'Explain a KV cache in one sentence.'

Hugging Face loads the tokenizer and pretrained tensors. Every generation
forward uses model.py with a fresh contiguous KV cache for each request.
Single request, no padding or sampling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .kv_cache import ContiguousKVCache
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


def generate_tokens(weights, config, tokenizer, prompt, max_new_tokens, *, eos_token_id,
                    use_cache=True):
    """Format one chat request and return only new token IDs, including generated EOS."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt must contain text')
    input_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors='pt',
    ).to(weights['model.embed_tokens.weight'].device)
    return _greedy_decode(input_ids, weights, config, max_new_tokens,
                          eos_token_id=eos_token_id, use_cache=use_cache)


@torch.inference_mode()
def _greedy_decode(input_ids, weights, config, max_new_tokens, *, eos_token_id,
                   use_cache=True):
    """Prefill once, then decode single tokens; use_cache=False verifies full prefixes."""
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('generation requires one nonempty sequence [1, tokens]')
    if input_ids.shape[1] + max_new_tokens - 1 > config.max_position_embeddings:
        raise ValueError('generation would require a forward beyond the configured context length')
    eos_ids = _eos_ids(eos_token_id, config.vocab_size)
    prepared = prepare_model(weights, config)
    cache = None
    if use_cache:
        embedding = weights['model.embed_tokens.weight']
        cache = ContiguousKVCache.allocate(
            num_layers=config.num_hidden_layers, batch_size=1,
            num_kv_heads=config.num_key_value_heads,
            capacity=input_ids.shape[1] + max_new_tokens - 1,
            head_dim=getattr(config, 'head_dim', None) or (
                config.hidden_size // config.num_attention_heads),
            dtype=embedding.dtype, device=embedding.device,
        )
    generated = []
    for step in range(max_new_tokens):
        logits = forward(input_ids, prepared, cache=cache, logits_to_keep=1)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if step + 1 == max_new_tokens or (eos_ids and next_token.item() in eos_ids):
            break
        input_ids = next_token if use_cache else torch.cat((input_ids, next_token), dim=1)
    return torch.cat(generated, dim=1)[0]


def main(argv=None):
    args = parse_args(argv, description=__doc__)
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
                                 args.max_new_tokens, eos_token_id=eos)
    print(f'Prompt: {args.prompt}')
    print(f'Generated tokens: {new_tokens.numel()}')
    print(f'Token IDs: {new_tokens.tolist()}')
    print('Response:')
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == '__main__':
    main()

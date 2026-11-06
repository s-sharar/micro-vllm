"""Manual greedy generation using Hugging Face forward calls and its KV cache.

Run in WSL:
    source .venv/bin/activate
    python decode.py --prompt "Explain a KV cache in one sentence."

Single request, no padding or sampling. The model must be in evaluation mode.
The transformer and cache are still Hugging Face implementations.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reference import parse_args


@torch.inference_mode()
def generate_tokens(model, tokenizer, prompt, max_new_tokens):
    """Return response token IDs, including EOS when generated."""
    if not prompt.strip():
        raise ValueError("prompt must contain text")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    # Use exactly the same conversation formatting as reference.py.
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    return _greedy_decode(model, inputs, max_new_tokens)


def _greedy_decode(model, inputs, max_new_tokens):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    past_key_values = None  # Every request starts with a fresh cache.
    generated = ()
    eos = model.generation_config.eos_token_id
    if isinstance(eos, int):
        eos = (eos,)
    eos_ids = () if eos is None else tuple(eos)

    for step in range(max_new_tokens):
        # First call: the full prompt (prefill). Later calls: one token (decode).
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, cache_position=cache_position,
            use_cache=True, return_dict=True, logits_to_keep=1,
        )
        # [1, vocabulary] -> [1, 1]. No sampling or logits transformations.
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = (*generated, next_token)

        # Include EOS in the result, but never run another forward after stopping.
        if step + 1 == max_new_tokens or (eos_ids and next_token.item() in eos_ids):
            break

        past_key_values = outputs.past_key_values
        input_ids = next_token
        # The mask covers cached tokens plus the token being processed next.
        attention_mask = torch.cat(
            (attention_mask, attention_mask.new_ones((1, 1))), dim=1,
        )
        cache_position = cache_position[-1:] + 1

    return torch.cat(generated, dim=1)[0]


def main(argv=None):
    args = parse_args(argv, description=__doc__)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check WSL GPU setup or use --device cpu.")

    print(f"Loading {args.model} on {args.device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()

    new_tokens = generate_tokens(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"Prompt: {args.prompt}")
    print(f"Generated tokens: {new_tokens.numel()}")
    print(f"Token IDs: {new_tokens.tolist()}")
    print("Response:")
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()

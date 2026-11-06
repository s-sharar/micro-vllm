"""Hugging Face baseline: one prompt, greedy generation, one response.

Run in WSL:
    source .venv/bin/activate
    python reference.py --prompt "Explain a KV cache in one sentence."

The first run downloads Qwen/Qwen2.5-0.5B-Instruct into Hugging Face's cache.
This is a correctness reference, not a performance benchmark.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_args(argv=None, *, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--prompt", default="Explain a KV cache in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error("--prompt must contain text")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    return args


@torch.inference_mode()
def generate_tokens(model, tokenizer, prompt, max_new_tokens):
    """Return only the newly generated token IDs for a single user message."""
    # 3. Format the conversation and convert text into token IDs.
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 4. Let Hugging Face run the greedy token loop and manage its KV cache.
    # A fresh config avoids inheriting sampling/repetition settings from a model.
    generation = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    output_ids = model.generate(
        **inputs, generation_config=generation, use_model_defaults=False,
    )

    # generate() returns prompt + response. Keep only the response.
    prompt_length = inputs["input_ids"].shape[1]
    return output_ids[0, prompt_length:]


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check WSL GPU setup or use --device cpu.")

    # 1. Load the tokenizer that belongs to these pretrained weights.
    print(f"Loading {args.model} on {args.device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 2. Load weights, move them onto the device, and disable training behavior.
    # Eager attention keeps the reference independent of FlashAttention kernels.
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()

    new_tokens = generate_tokens(model, tokenizer, args.prompt, args.max_new_tokens)

    # 5. Convert response IDs back into readable text.
    print(f"Prompt: {args.prompt}")
    print(f"Generated tokens: {new_tokens.numel()}")
    print(f"Token IDs: {new_tokens.tolist()}")
    print("Response:")
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()

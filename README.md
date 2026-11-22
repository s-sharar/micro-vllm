# micro-vLLM

A small LLM inference engine written from scratch in Python, PyTorch and CUDA. It runs Qwen2.5-0.5B-Instruct from its pretrained Hugging Face weights, with its own transformer forward pass, KV caching, CUDA attention kernels, paged KV memory and a continuous-batching scheduler.

## Features

- **Model:** Qwen2 forward pass (RMSNorm, Q/K/V projections, RoPE, grouped-query attention, gated MLP) that matches Hugging Face greedy outputs
- **KV caches:** contiguous, and paged over a shared pool of physical pages with per-request page tables that grow on demand
- **CUDA kernels:** online-softmax decode attention (FP16/FP32, grouped-query) that reads KV pages directly, and paged KV writes; both have batched variants that handle every request in one launch per layer
- **Attention backends:** `torch` (reference), `cuda` (native decode) and `auto` (PyTorch FlashAttention prefill plus native decode)
- **Memory management:** all-or-nothing page reservations, safe page reuse after GPU work completes, and rejection of requests that can never fit
- **Continuous batching:** requests join while others decode and leave as soon as they finish. Admission respects an active-request limit, a per-tick token budget and free pages. When memory runs out mid-decode, the newest request is preempted and later resumed by recomputation.

## Installation

Requires Python 3.10+, PyTorch with CUDA, and `transformers`. The CUDA kernels compile on first use through `torch.utils.cpp_extension`, which needs the CUDA toolkit and an NVIDIA GPU. FlashAttention prefill needs compute capability 8.0 or newer.

```bash
pip install -e '.[test,cuda]'
```

To compile only for your GPU, set `TORCH_CUDA_ARCH_LIST` (for example `8.9` for RTX 40-series).

## Usage

Generate from the command line:

```bash
python -m micro_vllm.generate --kv-cache paged --attention-backend auto --prompt "What is 2 + 2?"
```

Serve several prompts together with the engine:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from micro_vllm.engine import Engine

name = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(name)
checkpoint = AutoModelForCausalLM.from_pretrained(
    name, dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
weights, config = checkpoint.state_dict(), checkpoint.config
eos = checkpoint.generation_config.eos_token_id

engine = Engine(weights, config, num_pages=256, page_size=16, max_active_requests=8)
requests = [
    engine.submit(tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                tokenize=True, add_generation_prompt=True),
                  max_new_tokens=64, eos_ids=eos)
    for text in ('What is 2 + 2?', 'Write a haiku about GPUs.')
]
while engine.has_work:
    for request_id, token in engine.step_batch():
        pass  # tokens arrive here as they are generated
for request_id in requests:
    print(tokenizer.decode(engine.result(request_id).tokens, skip_special_tokens=True))
```

`num_pages × page_size` is the total number of KV tokens shared by all requests. `cancel(request_id)` stops a request, and `result(request_id)` returns its state, tokens and any error.

## Tests

```bash
python -m unittest discover -s test -t .
```

CPU tests run by default. Set `RUN_GPU_TESTS=1` to add CUDA tests, and `RUN_MODEL_TESTS=1` to add tests against the real Qwen checkpoint (it must already be in the Hugging Face cache).

## Performance

Measured on an RTX 4070 Laptop GPU with Qwen2.5-0.5B-Instruct in FP16, 64 output tokens per request, compared with generating the same requests one after another:

| Concurrent requests | Engine tok/s | Sequential tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 71.7 | 80.2 | 0.89× |
| 2 | 126.9 | 71.0 | 1.79× |
| 4 | 244.9 | 69.8 | 3.51× |
| 8 | 454.8 | 64.6 | 7.04× |

A single request runs slightly slower through the engine because of per-step scheduling overhead. In FP32 the engine's output matches one-at-a-time generation token for token. In FP16, batching can change rounding where two tokens are nearly tied.

## Layout

```text
src/micro_vllm/
  model.py, runtime.py          Qwen2 layers and the prepared forward (single and batched)
  attention.py, attention_backend.py, cuda_attention.py
                                reference attention, backend selection, CUDA attention wrappers
  kv_cache.py                   contiguous KV cache
  kv_pool.py                    shared physical page pool
  paged_kv_cache.py             per-request page table over the pool
  paged_kv_batch.py             one batched forward's view over several requests
  cuda_paged_cache.py           CUDA paged KV write wrapper
  engine.py                     continuous-batching request engine
  generate.py, reference.py     command-line generation and the Hugging Face reference
  csrc/                         C++/CUDA kernels
test/                           unittest suite (CPU, CUDA, real-model)
```

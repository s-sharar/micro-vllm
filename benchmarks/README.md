# Benchmarks

Run these tools from the repository root with the existing WSL virtual environment activated:

```bash
cd /home/shiha/projects/micro-vllm
source .venv/bin/activate
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 HF_HUB_OFFLINE=1
```

## Isolated decode attention

```bash
python -m benchmarks.decode_attention --contexts 32 128 512 1024 --repeats 5 --calls-per-sample 100 --warmup 20 --graph-calls 32 --graph-replays 10 --include-torch --output benchmarks/results/2026-11-08-flash-decode-attention.json
```

Compare our direct CUDA GQA attention, PyTorch bundled FlashAttention through Flash-only SDPA, and optionally our PyTorch reference. This uses one query position, 14 query heads, two original KV heads, and 64 features per head in FP16. Flags allow other compatible geometries.

Every backend receives the same seeded Q/K/V objects. K/V are prefix views into spare-capacity buffers; unused capacity contains NaNs. Outputs must match reference attention within FP16 tolerance before timing. Untimed CPU/CUDA profiling records actual operator and kernel names and rejects Flash math fallback.

Three measurements are reported separately:

- **Ordinary call wall latency:** synchronized wall time for repeated full Python attention calls, including dispatch and allocation overhead.
- **CUDA graph replay wall latency:** synchronized wall time per attention operation in a graph containing 32 calls, replayed ten times. Python dispatch and allocations happen during capture. This compares captured operation sequences, including any auxiliary kernels, rather than a single raw kernel.
- **Raw CUDA-event latency:** a separate repeated-call measurement, preserved for diagnosis. It includes CPU launch gaps and must not be presented as pure kernel execution time.

Backend order rotates each repeat. Setup, correctness checks, warmup, compilation, profiling, and graph capture are outside timed samples. Raw samples and medians are retained.

## Complete model generation

```bash
python -m benchmarks.generation --contexts 32 128 512 1024 --decode-steps 16 --repeats 5 --output benchmarks/results/2026-11-08-flash-decode-generation.json
```

The same Qwen2.5-0.5B-Instruct pretrained tensors are bound to all four modes:

| Benchmark mode | Prefill attention | Cached one-token decode |
| --- | --- | --- |
| torch | Our PyTorch reference | Our PyTorch reference |
| cuda | Our PyTorch reference | Our CUDA kernel |
| auto | FlashAttention when eligible | Our CUDA kernel |
| flash | Same auto prefill | FlashAttention |

The benchmark-only flash mode temporarily remaps native decode selections inside a scoped context and restores routing on exit, including failure. It does not add a production backend or modify generation defaults. Benchmark runs are sequential; do not embed the scoped override in a concurrent server.

For the direct model comparison, use auto versus flash: both have the same prefill. Each run processes an exact-length shared-prefix input and performs 16 feedback decode forwards, selecting 17 IDs total. EOS stopping is disabled so work cannot shorten. Mode order rotates between repeats; token IDs, routes, raw samples, and medians are retained.

Decode milliseconds per token include all transformer layers, cache writes, final vocabulary projection, token selection, and Python dispatch. They are not attention-only timings. Loading, tokenization, tensor transfer, model preparation, cache allocation/reset, initialization, and compilation are excluded. Total latency includes the synchronization separating prefill and decode.

## Continuous batching engine

```bash
python -m benchmarks.continuous_batching --concurrency 1 2 4 8 --max-new-tokens 64 --repeats 5 --output benchmarks/results/2026-11-22-continuous-batching.json
```

The script compares the Step 4 `Engine` with sequential `generate_tokens(kv_cache='paged')` on real Qwen FP16, using the first C of eight fixed chat prompts at each concurrency C. Both paths use the same attention backend and page size and a pool of the same size, holding every request's full KV, so the engine never preempts. EOS is disabled, so every request produces the full `--max-new-tokens`.

Prompts are tokenized before timing. The engine receives IDs through `submit()`; `generate_tokens()` receives a tokenizer stand-in that returns the prepared IDs. Timed regions start and end with CUDA synchronization. They cover submission, every forward and token transfer to the host. Engine construction and pool allocation are setup; `generate_tokens()` prepares the model inside each call. Each path gets one untimed warmup per concurrency, then the order alternates every repeat.

Reported per path: tokens per second (C × budget ÷ wall time) and per-request completion time per output token, measured from the shared start. For the baseline, each request's own service time per token is also reported. Generated IDs are compared with the first sequential run. Any first divergent token gets top-two logit margins from an untimed engine rerun and a standalone forward. A concurrency whose pools do not fit in GPU memory is recorded as skipped and ends the sweep.

## Interpretation

The Flash baseline is PyTorch's bundled fused FlashAttention implementation for a single query, not the separate flash-attn package's specialized flash_attn_with_kvcache API. Strict Flash dispatch is requested; reference fallback is not accepted in the isolated comparison.

On this WSL machine earlier CUDA-event intervals were approximately 11% above synchronized host wall intervals, even for an independent delay control. The cause is unknown. Use synchronized wall metrics for reported latency and keep event clocks labeled separately. Graph replay timings help distinguish GPU work from ordinary Python-call overhead; the production generation path does not yet replay CUDA graphs.

Raw measurements and readable reports live in results/. The previous three-mode benchmark is retained as 2026-11-08-attention-paths.json and .md. The newer comparison is documented in results/2026-11-08-flash-decode.md.

The causal-mask behavior and forced Flash dispatch follow [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). For one cached query, every valid K/V row is visible; the existing wrapper uses lower-right causality.

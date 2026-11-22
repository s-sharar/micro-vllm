"""Offline CUDA FP16 greedy-generation benchmark (EOS stopping disabled).

Example: python -m benchmarks.generation --contexts 32 128 512 --output report.json
CUDA event intervals include Python launch gaps; they are not kernel-only times.
"""
import argparse
from contextlib import contextmanager
from unittest.mock import patch
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from micro_vllm import runtime
from micro_vllm.kv_cache import ContiguousKVCache
from micro_vllm.model import prepare_model

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
TEXT = (
    'A language model reads a sequence of tokens and predicts the next token. '
    'A key value cache keeps earlier attention results so generation can reuse '
    'them while new tokens arrive. This benchmark measures prefill and decode. '
)
MODES = {
    'torch': 'PyTorch prefill and PyTorch decode',
    'cuda': 'PyTorch prefill and native CUDA decode',
    'auto': 'FlashAttention prefill when supported; native CUDA decode',
    'flash': 'FlashAttention prefill and decode through bundled Flash-only SDPA (benchmark only)',
}


@contextmanager
def benchmark_mode(mode):
    """Temporarily force Flash decode for this sequential benchmark only.

    The production policy API remains unchanged. Keep the override outside
    timed work and restore it even on errors; never use this in a server.
    """
    if mode not in MODES:
        raise ValueError('unknown benchmark mode')
    if mode != 'flash':
        yield mode
        return
    original = runtime.select_attention_backend

    def select_flash_decode(*args, **kwargs):
        selected = original(*args, **kwargs)
        return 'flash' if selected == 'cuda' else selected

    with patch.object(runtime, 'select_attention_backend', new=select_flash_decode):
        yield 'auto'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL, help='cached Hugging Face ID or local directory')
    parser.add_argument('--contexts', type=int, nargs='+', default=[32, 128, 512])
    parser.add_argument('--decode-steps', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'results' / 'generation.json')
    args = parser.parse_args(argv)
    if any(value <= 0 for value in args.contexts):
        parser.error('--contexts must contain positive lengths')
    if args.decode_steps <= 0 or args.repeats <= 0:
        parser.error('--decode-steps and --repeats must be positive')
    if len(set(args.contexts)) != len(args.contexts):
        parser.error('--contexts must contain distinct lengths')
    return args


def make_prefix(tokenizer, length, device):
    """Repeat one coherent passage's IDs; every shorter input shares its prefix."""
    if type(length) is not int or length <= 0:
        raise ValueError('context length must be a positive integer')
    tokens = tokenizer.encode(TEXT, add_special_tokens=False)
    if not tokens:
        raise ValueError('the benchmark passage must tokenize to at least one ID')
    repeated = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
    return torch.tensor([repeated], dtype=torch.long, device=device)


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _events(device):
    if device.type == 'cuda':
        return tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
    return (None,) * 4


def _record(event):
    if event is not None:
        event.record()


def _elapsed(start, end):
    return None if start is None else start.elapsed_time(end)


@torch.inference_mode()
def timed_generation(prepared, prefix, cache, decode_steps):
    """One prefill token, then exactly N feedback steps; no EOS checks or host IDs."""
    if type(decode_steps) is not int or decode_steps <= 0:
        raise ValueError('decode_steps must be a positive integer')
    if prefix.ndim != 2 or prefix.shape[0] != 1 or prefix.shape[1] == 0:
        raise ValueError('prefix must contain one nonempty token sequence')
    device = prefix.device
    pre_start, pre_end, dec_start, dec_end = _events(device)
    # Materialize event handles before timing (CUDA initializes them lazily).
    for event in (pre_start, pre_end, dec_start, dec_end):
        _record(event)
    _synchronize(device)
    total_start = prefill_start = perf_counter()
    _record(pre_start)
    token = prepared.forward(prefix, cache=cache, logits_to_keep=1)[:, -1, :].argmax(-1, keepdim=True)
    generated = [token]
    _record(pre_end)
    _synchronize(device)
    prefill_ms = (perf_counter() - prefill_start) * 1000
    decode_start = perf_counter()
    _record(dec_start)
    for _ in range(decode_steps):
        token = prepared.forward(token, cache=cache, logits_to_keep=1)[:, -1, :].argmax(-1, keepdim=True)
        generated.append(token)
    _record(dec_end)
    _synchronize(device)
    decode_ms = (perf_counter() - decode_start) * 1000
    total_ms = (perf_counter() - total_start) * 1000
    return dict(
        prefill_wall_ms=prefill_ms, decode_wall_ms=decode_ms, total_wall_ms=total_ms,
        prefill_event_ms=_elapsed(pre_start, pre_end),
        decode_event_ms=_elapsed(dec_start, dec_end),
        token_ids=torch.cat(generated, dim=1)[0].tolist(),
    )


def summarize_runs(runs, decode_steps):
    """Retain raw measurements and token IDs, including any numerical divergence."""
    fields = ('prefill_wall_ms', 'decode_wall_ms', 'total_wall_ms',
              'prefill_event_ms', 'decode_event_ms')
    medians = {
        field: None if runs[0][field] is None else median(run[field] for run in runs)
        for field in fields
    }
    for clock in ('wall', 'event'):
        value = medians[f'decode_{clock}_ms']
        medians[f'decode_{clock}_ms_per_token'] = (
            None if value is None else value / decode_steps)
    return dict(median=medians, runs=runs,
                repeat_token_parity=all(run['token_ids'] == runs[0]['token_ids'] for run in runs))


def resolved_routes(prepared, prefix):
    """Resolve the same policy/geometry used by forward, outside measured work."""
    common = dict(
        device=prepared.embedding_weight.device, dtype=prepared.embedding_weight.dtype,
        batch_size=prefix.shape[0], head_dim=prepared.head_dim, has_cache=True)
    return dict(
        prefill=runtime.select_attention_backend(
            prepared.attention_backend, query_length=prefix.shape[1],
            past_length=0, **common),
        decode=runtime.select_attention_backend(
            prepared.attention_backend, query_length=1,
            past_length=prefix.shape[1], **common),
    )


def allocate_cache(prefix, config, decode_steps, dtype):
    return ContiguousKVCache.allocate(
        num_layers=config.num_hidden_layers, batch_size=1,
        num_kv_heads=config.num_key_value_heads,
        capacity=prefix.shape[1] + decode_steps,
        head_dim=getattr(config, 'head_dim', None) or (
            config.hidden_size // config.num_attention_heads),
        dtype=dtype, device=prefix.device)


def compare_context(prepared, prefix, config, decode_steps, repeats, dtype):
    """Warm every mode, then rotate order each repeat to reduce ordering bias."""
    caches, routes = {}, {}
    for mode, model in prepared.items():
        cache = caches[mode] = allocate_cache(prefix, config, decode_steps, dtype)
        with benchmark_mode(mode):
            routes[mode] = resolved_routes(model, prefix)
            timed_generation(model, prefix, cache, decode_steps)
    names, runs, orders = list(prepared), {mode: [] for mode in prepared}, []
    for repeat in range(repeats):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        orders.append(order)
        for mode in order:
            caches[mode].reset()
            print(f'Context {prefix.shape[1]}: {mode}, repeat {repeat + 1}/{repeats}', flush=True)
            with benchmark_mode(mode):
                runs[mode].append(timed_generation(prepared[mode], prefix, caches[mode], decode_steps))
    summaries = {mode: dict(summarize_runs(samples, decode_steps),
                            resolved_attention=routes[mode]) for mode, samples in runs.items()}
    return summaries, orders


def benchmark_context(prepared, prefix, config, decode_steps, repeats, dtype):
    routes = resolved_routes(prepared, prefix)
    cache = allocate_cache(prefix, config, decode_steps, dtype)
    # Complete warmup covers this context and all increasing decode lengths.
    timed_generation(prepared, prefix, cache, decode_steps)
    runs = []
    for _ in range(repeats):
        cache.reset()
        runs.append(timed_generation(prepared, prefix, cache, decode_steps))
    summary = summarize_runs(runs, decode_steps)
    summary['resolved_attention'] = routes
    return summary


def _metadata(args):
    return dict(
        model=args.model, torch_version=torch.__version__,
        cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        dtype='torch.float16', batch_size=1, contexts=args.contexts,
        repeats=args.repeats, decode_steps=args.decode_steps,
        generated_tokens_per_run=args.decode_steps + 1, eos_stopping=False,
        prefix_text=TEXT, modes=MODES,
        measurement_order='Rotate mode order each repeat; raw orders retained.',
        flash_implementation='PyTorch bundled FlashAttention through Flash-only SDPA; no fallback.',
        excluded_setup=[
            'offline tokenizer and checkpoint loading', 'model preparation',
            'input tokenization and device transfer', 'KV cache allocation/reset',
            'one complete warmup per mode and context (JIT/FlashAttention initialization)',
            'CUDA event creation/initialization', 'output token transfer and JSON serialization',
        ],
        timing_notes=[
            'All wall intervals end with CUDA synchronization.',
            'Prefill includes its first greedy argmax; decode includes exactly N forward/argmax steps.',
            'Total wall spans prefill plus decode, including the intervening synchronization.',
            'CUDA events include GPU stream idle time from Python launch gaps; not isolated kernel time.',
            'Each mode feeds its own generated token back; EOS never shortens the workload.',
            'Use synchronized wall times for comparisons; WSL raw CUDA event clock previously differed by ~11%.',
        ],
    )


def run_benchmark(args, weights, config, tokenizer):
    if max(args.contexts) + args.decode_steps > config.max_position_embeddings:
        raise ValueError('context plus decode steps exceeds the configured position limit')
    prepared = {mode: prepare_model(weights, config,
                                    attention_backend='auto' if mode == 'flash' else mode)
                for mode in MODES}
    report = dict(metadata=_metadata(args), results=[])
    for length in args.contexts:
        prefix = make_prefix(tokenizer, length, 'cuda')
        modes, orders = compare_context(prepared, prefix, config, args.decode_steps,
                                        args.repeats, weights['model.embed_tokens.weight'].dtype)
        result = dict(context_length=length, input_ids=prefix[0].tolist(),
                      modes=modes, measurement_orders=orders)
        baseline = result['modes']['torch']['runs'][0]['token_ids']
        result['token_parity_across_modes'] = all(
            run['token_ids'] == baseline
            for mode in result['modes'].values() for run in mode['runs'])
        report['results'].append(result)
    return report


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit('This benchmark requires CUDA.')
    print(f'Loading cached {args.model} in CUDA FP16 (network disabled).', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    checkpoint = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float16,
        attn_implementation='eager').to('cuda').eval()
    weights, config = checkpoint.state_dict(), checkpoint.config
    del checkpoint
    report = run_benchmark(args, weights, config, tokenizer)
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
        print(f'Saved report: {args.output}')
    print(encoded)


if __name__ == '__main__':
    main()

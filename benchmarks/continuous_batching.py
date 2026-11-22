"""Offline real-Qwen continuous batching benchmark: Engine versus sequential generate_tokens().

Example: python -m benchmarks.continuous_batching --concurrency 1 2 4 8 --output report.json
Both paths start from IDs tokenized before timing, disable EOS so every request
produces its full budget, and end their timed region with CUDA synchronization.
"""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from micro_vllm import generate
from micro_vllm.engine import Engine, RequestState
from micro_vllm.generate import _cache_dimensions
from micro_vllm.kv_pool import KVPool
from micro_vllm.runtime import PreparedModel, prepare_model

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
PROMPTS = (
    'What is 2 + 2?',
    'Explain a KV cache in one sentence.',
    'List three primary colors and say which one you like most.',
    'Write a haiku about GPUs.',
    'Describe how one transformer layer processes a token.',
    'Give two reasons to batch inference requests on a GPU.',
    'Summarize the rules of chess in three short sentences.',
    'What is the capital of Canada, and why was that city chosen?',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL, help='cached Hugging Face ID or local directory')
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--attention-backend', choices=('auto', 'torch', 'cuda'), default='auto')
    parser.add_argument('--page-size', type=int, default=16)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results'
                        / 'continuous_batching.json')
    args = parser.parse_args(argv)
    if any(not 0 < value <= len(PROMPTS) for value in args.concurrency):
        parser.error(f'--concurrency values must be between 1 and {len(PROMPTS)}')
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error('--concurrency values must be distinct')
    if min(args.max_new_tokens, args.repeats, args.page_size) <= 0:
        parser.error('--max-new-tokens, --repeats and --page-size must be positive')
    return args


class Pretokenized:
    """Tokenizer stand-in that hands generate_tokens() chat IDs prepared before timing."""

    def __init__(self, tokenizer, texts):
        self._ids = {text: tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True,
            return_tensors='pt') for text in texts}

    def apply_chat_template(self, messages, **kwargs):
        return self._ids[messages[0]['content']]

    def ids(self, text):
        return self._ids[text][0].tolist()


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def pages_needed(prompts, max_new_tokens, page_size):
    """Pages that hold every request's full KV at once, so neither path preempts or waits."""
    return sum(-(-(len(prompt) + max_new_tokens - 1) // page_size) for prompt in prompts)


def time_engine(engine, prompts, max_new_tokens, device):
    """Submit every prompt at t=0, then tick until idle; completion is a request's final event."""
    _synchronize(device)
    start = perf_counter()
    requests = [engine.submit(prompt, max_new_tokens, None) for prompt in prompts]
    counts, done, ticks = dict.fromkeys(requests, 0), {}, 0
    while engine.has_work:
        events = engine.step_batch()
        ticks += 1
        now = perf_counter()
        for request, _ in events:
            counts[request] += 1
            if counts[request] == max_new_tokens:
                done[request] = now
    _synchronize(device)
    wall = perf_counter() - start
    results = [engine.result(request) for request in requests]
    if any(result.state is not RequestState.COMPLETED for result in results):
        raise RuntimeError('every benchmark request must complete')
    return dict(wall_ms=wall * 1000, ticks=ticks,
                completion_ms=[(done[request] - start) * 1000 for request in requests],
                preemptions=[result.preemptions for result in results],
                token_ids=[list(result.tokens) for result in results])


def time_sequential(weights, config, tokenizer, texts, max_new_tokens, *, attention_backend,
                    page_size, pool, device):
    """Run generate_tokens() back to back from t=0; each request ends with its synchronized IDs."""
    _synchronize(device)
    start = perf_counter()
    starts, done, token_ids = [], [], []
    for text in texts:
        starts.append(perf_counter())
        tokens = generate.generate_tokens(
            weights, config, tokenizer, text, max_new_tokens, eos_token_id=None,
            attention_backend=attention_backend, kv_cache='paged', page_size=page_size,
            kv_pool=pool)
        token_ids.append(tokens.tolist())
        done.append(perf_counter())
    _synchronize(device)
    wall = perf_counter() - start
    return dict(wall_ms=wall * 1000,
                completion_ms=[(end - start) * 1000 for end in done],
                service_ms=[(end - begin) * 1000 for begin, end in zip(starts, done)],
                token_ids=token_ids)


def _first_divergence(left, right):
    return next((index for index, (x, y) in enumerate(zip(left, right)) if x != y),
                None if len(left) == len(right) else min(len(left), len(right)))


def _top2_margin(logits):
    top = logits.topk(2).values
    return (top[0] - top[1]).item()


@torch.inference_mode()
def _standalone_logits(weights, config, prompt, tokens, position, *, attention_backend, page_size):
    prepared = prepare_model(weights, config, attention_backend=attention_backend)
    device = weights['model.embed_tokens.weight'].device
    cache = generate._allocate_cache(weights, config, len(prompt) + position, 'paged', page_size)
    ids = torch.tensor([prompt + tokens[:position]], device=device)
    logits = prepared.forward(ids[:, :len(prompt)], cache=cache, logits_to_keep=1)
    for index in range(len(prompt), ids.shape[1]):
        logits = prepared.forward(ids[:, index:index + 1], cache=cache, logits_to_keep=1)
    cache.release()
    return logits[0, -1].float().cpu()


def divergence_margins(weights, config, prompts, max_new_tokens, divergent, reference, timed, *,
                       attention_backend, page_size):
    """Untimed engine rerun recording top-two logit margins at each first divergent token."""
    engine = Engine(weights, config, num_pages=pages_needed(prompts, max_new_tokens, page_size),
                    page_size=page_size, attention_backend=attention_backend,
                    max_active_requests=len(prompts))
    rows, real = [], PreparedModel.forward_batch

    def capture(prepared, input_ids, caches, **kwargs):
        logits = real(prepared, input_ids, caches, **kwargs)
        rows.extend(logits.float().cpu())
        return logits
    requests = [engine.submit(prompt, max_new_tokens, None) for prompt in prompts]
    events = []
    with patch.object(PreparedModel, 'forward_batch', autospec=True, side_effect=capture):
        while engine.has_work:
            events.extend(engine.step_batch())
    by_request = {request: [] for request in requests}
    for (request, _), row in zip(events, rows):
        by_request[request].append(row)
    report = []
    for index, position in divergent:
        engine_logits = by_request[requests[index]][position]
        standalone = _standalone_logits(weights, config, prompts[index], reference[index], position,
                                        attention_backend=attention_backend, page_size=page_size)
        report.append(dict(
            request=index, position=position,
            engine_token=engine_logits.argmax().item(), standalone_token=reference[index][position],
            rerun_matches_timed_engine=engine_logits.argmax().item() == timed[index][position],
            engine_top2_margin=_top2_margin(engine_logits),
            standalone_top2_margin=_top2_margin(standalone),
            max_abs_logit_difference=(engine_logits - standalone).abs().max().item()))
    return report


def summarize(runs, concurrency, max_new_tokens):
    """Medians over repeats of throughput and per-request wall time per output token."""
    tokens = concurrency * max_new_tokens

    def per_token(run, key):
        return [value / max_new_tokens for value in run[key]]
    summary = dict(
        wall_ms=median(run['wall_ms'] for run in runs),
        tokens_per_s=median(tokens / (run['wall_ms'] / 1000) for run in runs),
        mean_completion_ms_per_token=median(
            sum(per_token(run, 'completion_ms')) / concurrency for run in runs),
        max_completion_ms_per_token=median(max(per_token(run, 'completion_ms')) for run in runs),
        per_request_completion_ms_per_token=[
            median(values) for values in zip(*(per_token(run, 'completion_ms') for run in runs))])
    if 'service_ms' in runs[0]:
        summary['mean_service_ms_per_token'] = median(
            sum(per_token(run, 'service_ms')) / concurrency for run in runs)
    if 'ticks' in runs[0]:
        summary['ticks'] = median(run['ticks'] for run in runs)
    return summary


def benchmark_concurrency(weights, config, tokenizer, texts, *, max_new_tokens, repeats,
                          attention_backend, page_size):
    """Warm both paths, then alternate their order each repeat; token IDs are compared per run."""
    device = weights['model.embed_tokens.weight'].device
    pretokenized = Pretokenized(tokenizer, texts)
    prompts = [pretokenized.ids(text) for text in texts]
    num_pages = pages_needed(prompts, max_new_tokens, page_size)
    engine = Engine(weights, config, num_pages=num_pages, page_size=page_size,
                    attention_backend=attention_backend, max_active_requests=len(texts))
    pool = KVPool.allocate(**_cache_dimensions(weights, config), num_pages=num_pages,
                           page_size=page_size)
    options = dict(attention_backend=attention_backend, page_size=page_size, pool=pool,
                   device=device)

    def run(name):
        if name == 'engine':
            return time_engine(engine, prompts, max_new_tokens, device)
        return time_sequential(weights, config, pretokenized, texts, max_new_tokens, **options)
    for name in ('engine', 'sequential'):
        run(name)
    runs, orders = {'engine': [], 'sequential': []}, []
    for repeat in range(repeats):
        order = ['engine', 'sequential'][::1 if repeat % 2 == 0 else -1]
        orders.append(order)
        for name in order:
            print(f'Concurrency {len(texts)}: {name}, repeat {repeat + 1}/{repeats}', flush=True)
            runs[name].append(run(name))
    reference = runs['sequential'][0]['token_ids']
    divergence = {name: [[_first_divergence(actual, expected)
                          for actual, expected in zip(run['token_ids'], reference)]
                         for run in samples] for name, samples in runs.items()}
    engine_summary = summarize(runs['engine'], len(texts), max_new_tokens)
    sequential_summary = summarize(runs['sequential'], len(texts), max_new_tokens)
    divergent = sorted({(index, position) for samples in divergence.values() for run in samples
                        for index, position in enumerate(run) if position is not None})
    margins = divergence_margins(
        weights, config, prompts, max_new_tokens, divergent, reference,
        runs['engine'][0]['token_ids'], attention_backend=attention_backend,
        page_size=page_size) if divergent else []
    return dict(
        concurrency=len(texts), prompts=texts, prompt_lengths=[len(p) for p in prompts],
        num_pages=num_pages, engine=engine_summary, sequential=sequential_summary,
        throughput_speedup=engine_summary['tokens_per_s'] / sequential_summary['tokens_per_s'],
        token_ids_match=all(index is None for samples in divergence.values()
                            for run in samples for index in run),
        first_divergent_token=divergence, divergence_margins=margins, measurement_orders=orders,
        preemptions=[run['preemptions'] for run in runs['engine']], runs=runs)


def _metadata(args):
    return dict(
        model=args.model, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), dtype='torch.float16',
        attention_backend=args.attention_backend, page_size=args.page_size,
        max_new_tokens=args.max_new_tokens, repeats=args.repeats, eos_stopping=False,
        concurrency_requested=args.concurrency,
        paths={
            'engine': 'Engine(max_active_requests=concurrency): all requests submitted at t=0, '
                      'step_batch() until idle',
            'sequential': "generate_tokens(kv_cache='paged') per request, back to back, with a "
                          'preallocated KVPool of the same size',
        },
        routes=('auto: one-row prefill uses Flash, multi-row prefill uses reference attention, '
                'decode uses the native paged kernel (batched for several rows)'),
        excluded_setup=[
            'offline tokenizer and checkpoint loading', 'chat-template tokenization',
            'Engine construction (model preparation, pool allocation) and the baseline pool',
            'one untimed warmup run of each path per concurrency',
            'JSON serialization',
        ],
        timing_notes=[
            'Timed regions start after torch.cuda.synchronize() and end with it.',
            'Both include submission/validation, input transfer, every forward and token '
            'transfer to the host; generate_tokens also prepares the model per call.',
            'Per-request completion is measured from the shared start, after that request '
            'produced its final token ID on the host (the engine syncs every tick, the '
            'baseline syncs at the end of each request).',
            'Repeat order alternates engine/sequential; medians over repeats are reported.',
            'The pool holds every request at once, so the engine never preempts or waits for pages.',
            'Token IDs are compared with the first sequential run; top-two logit margins at a '
            'first divergent token come from an untimed engine rerun and a standalone forward.',
        ],
    )


def run_benchmark(args, weights, config, tokenizer):
    report = dict(metadata=_metadata(args), results=[], skipped=[])
    for concurrency in sorted(args.concurrency):
        texts = PROMPTS[:concurrency]
        try:
            report['results'].append(benchmark_concurrency(
                weights, config, tokenizer, texts, max_new_tokens=args.max_new_tokens,
                repeats=args.repeats, attention_backend=args.attention_backend,
                page_size=args.page_size))
        except torch.cuda.OutOfMemoryError as error:
            report['skipped'].append(dict(concurrency=concurrency, reason=repr(error)))
            break
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + '\n', encoding='utf-8')
    print(f'Saved report: {args.output}')
    for result in report['results']:
        print(f"concurrency {result['concurrency']}: engine "
              f"{result['engine']['tokens_per_s']:.1f} tok/s vs sequential "
              f"{result['sequential']['tokens_per_s']:.1f} tok/s; IDs match: "
              f"{result['token_ids_match']}")


if __name__ == '__main__':
    main()

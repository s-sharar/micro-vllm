"""Isolated FP16 GQA decode: direct custom CUDA versus forced fused Flash.

Run python -m benchmarks.decode_attention --help from the repository root.
Synchronized wall time is authoritative. CUDA-event samples are retained
separately because event clocks have shown discrepancies under WSL.
Graph measurements replay complete captured attention operations, including
each backend's full operator sequence; they are not single-kernel timings.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter

import torch

from micro_vllm.attention import causal_attention
from micro_vllm.attention_backend import flash_attention
from micro_vllm.cuda_attention import decode_attention_gqa


@dataclass(frozen=True)
class DecodeInputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    capacity: int


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", nargs="+", type=positive_int, default=[32, 128, 512, 1024])
    parser.add_argument("--query-heads", type=positive_int, default=14)
    parser.add_argument("--kv-heads", type=positive_int, default=2)
    parser.add_argument("--head-dim", type=positive_int, default=64)
    parser.add_argument("--spare-capacity", type=positive_int, default=19)
    parser.add_argument("--seed", type=int, default=932)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--calls-per-sample", type=positive_int, default=100)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--graph-calls", type=positive_int, default=32)
    parser.add_argument("--graph-replays", type=positive_int, default=10)
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/decode_attention.json"))
    args = parser.parse_args(argv)
    if args.query_heads % args.kv_heads:
        parser.error("query heads must be divisible by KV heads")
    if args.head_dim > 256 or args.head_dim % 8:
        parser.error("Flash head dimension must be a multiple of 8 and at most 256")
    return args


def make_inputs(length, *, query_heads, kv_heads, head_dim, spare_capacity, seed, device):
    """Keep original grouped KV prefix views; unused backing storage is NaN."""
    generator = torch.Generator(device=device).manual_seed(seed)
    options = dict(device=device, dtype=torch.float16)
    query = torch.randn((1, query_heads, 1, head_dim), generator=generator, **options)
    capacity = length + spare_capacity
    storage_shape = (1, kv_heads, capacity, head_dim)
    storage = [torch.full(storage_shape, float("nan"), **options) for _ in range(2)]
    for tensor in storage:
        tensor[:, :, :length].copy_(torch.randn(
            (1, kv_heads, length, head_dim), generator=generator, **options))
    return DecodeInputs(query, storage[0][:, :, :length], storage[1][:, :, :length], capacity)


def make_backends(inputs, include_torch=False):
    query, key, value = inputs.query, inputs.key, inputs.value
    past = key.shape[2] - 1
    functions = {
        "custom": lambda: decode_attention_gqa(query, key, value),
        "flash": lambda: flash_attention(query, key, value, past_length=past),
    }
    if include_torch:
        functions["torch"] = lambda: causal_attention(query, key, value, past_length=past)
    return functions


def check_outputs(functions, expected):
    """Fail before any profiling or timing if an implementation is incorrect."""
    errors = {}
    for name, function in functions.items():
        actual = function()
        try:
            torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-3)
        except AssertionError as error:
            raise RuntimeError(f"{name} correctness check failed: {error}") from error
        errors[name] = float((actual.float() - expected.float()).abs().max().item())
    return errors


def verify_profile(name, profile):
    operators, kernels = profile["operators"], profile["cuda_kernels"]
    if name == "flash":
        fused = any("_scaled_dot_product_flash_attention" in op for op in operators)
        math = any("_scaled_dot_product_attention_math" in op for op in operators)
        flash_kernel = any("flash" in kernel.lower() for kernel in kernels)
        if not fused or math or not flash_kernel:
            raise RuntimeError(f"Flash must use its fused CUDA operator without math fallback: {profile}")
    if name == "custom" and not any("decode_attention_online_kernel" in kernel for kernel in kernels):
        raise RuntimeError(f"Custom GQA CUDA kernel was not observed: {profile}")


def profile_backends(functions):
    """Untimed CPU and CUDA activity proves the actual dispatched operators."""
    profiles = {}
    for name, function in functions.items():
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], acc_events=True) as trace:
            function()
            torch.cuda.synchronize()
        events = trace.events()
        profiles[name] = {
            "operators": sorted({event.name for event in events
                                 if event.device_type == torch.autograd.DeviceType.CPU}),
            "cuda_kernels": sorted({event.name for event in events
                                    if event.device_type == torch.autograd.DeviceType.CUDA}),
        }
        verify_profile(name, profiles[name])
    return profiles


def warm_backends(functions, calls):
    """Warm every backend on a side stream before capturing any backend."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for function in functions.values():
            for _ in range(calls):
                function()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def capture_graphs(functions, calls):
    graphs = {}
    for name, function in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [function() for _ in range(calls)]
        graphs[name] = (graph, outputs)  # Retain outputs and their graph-pool storage.
        graph.replay()
        torch.cuda.synchronize()
    return graphs


def wrapper_wall_ms(function, calls):
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(calls):
        function()
    torch.cuda.synchronize()
    return (perf_counter() - start) * 1000


def cuda_event_ms(function, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    torch.cuda.synchronize()
    start.record()
    for _ in range(calls):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def rotated_names(names, repeat):
    offset = repeat % len(names)
    return names[offset:] + names[:offset]


def measure_all(functions, args):
    warm_backends(functions, args.warmup)
    graphs = capture_graphs(functions, args.graph_calls)
    samples = []
    for repeat in range(args.repeats):
        order = rotated_names(list(functions), repeat)
        for name in order:
            wall = wrapper_wall_ms(functions[name], args.calls_per_sample)
            event = cuda_event_ms(functions[name], args.calls_per_sample)
            graph = wrapper_wall_ms(graphs[name][0].replay, args.graph_replays)
            samples.append({
                "repeat": repeat, "order": order, "backend": name,
                "wrapper_wall_ms": wall, "cuda_event_ms": event,
                "graph_replay_wall_ms": graph,
                "wrapper_wall_us": wall * 1000 / args.calls_per_sample,
                "cuda_event_us": event * 1000 / args.calls_per_sample,
                "graph_replay_wall_us": graph * 1000 / (args.graph_calls * args.graph_replays),
            })
    return summarize_samples(samples)


def summarize_samples(samples):
    metrics = ("wrapper_wall_us", "cuda_event_us", "graph_replay_wall_us")
    timings = {}
    for name in dict.fromkeys(row["backend"] for row in samples):
        timings[name] = {}
        for metric in metrics:
            values = [row[metric] for row in samples if row["backend"] == name]
            timings[name][metric] = {"samples": values, "median": statistics.median(values)}
    speedup = {}
    if "flash" in timings and "custom" in timings:
        speedup = {metric: timings["flash"][metric]["median"] / timings["custom"][metric]["median"]
                   for metric in metrics}
    return {"samples": samples, "timings": timings, "custom_speedup_vs_flash": speedup}


@torch.inference_mode()
def benchmark_case(args, length, device="cuda"):
    inputs = make_inputs(
        length, query_heads=args.query_heads, kv_heads=args.kv_heads,
        head_dim=args.head_dim, spare_capacity=args.spare_capacity, seed=args.seed, device=device)
    functions = make_backends(inputs, args.include_torch)
    expected = causal_attention(inputs.query, inputs.key, inputs.value, past_length=length - 1)
    errors = check_outputs(functions, expected)
    profiles = profile_backends(functions)
    result = measure_all(functions, args)
    result.update({
        "context_length": length, "capacity": inputs.capacity,
        "shapes": {name: list(getattr(inputs, name).shape) for name in ("query", "key", "value")},
        "strides": {name: list(getattr(inputs, name).stride()) for name in ("query", "key", "value")},
        "max_abs_error": errors, "profiles": profiles,
    })
    return result


def environment():
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "pytorch": torch.__version__, "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "driver": driver,
        "gpu": torch.cuda.get_device_name(), "gpu_capability": list(torch.cuda.get_device_capability()),
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
        "dtype": "torch.float16",
    }


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU and the custom CUDA extension toolchain")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(), "config": config,
        "measurement_notes": {
            "authoritative_metric": "wrapper_wall_us",
            "wrapper_wall_us": "Synchronized wall time of full Python attention calls, divided by calls.",
            "cuda_event_us": "Separate CUDA-event loop; retain raw clocks, WSL event clocks may differ from wall.",
            "graph_replay_wall_us": (
                "Synchronized wall time per full attention operation in a CUDA graph replay; "
                "each replay executes graph_calls operations, averaged over graph_replays. "
                "Not a raw single-kernel measurement; allocations and Python dispatch occur at capture."),
            "speedup": "Flash median divided by custom median; greater than 1 favors custom.",
            "inputs": "Identical seeded Q/K/V objects; original KV prefix strides; unused capacity is NaN.",
        },
        "cases": [],
    }
    for length in args.contexts:
        case = benchmark_case(args, length)
        report["cases"].append(case)
        print(f"context={length} custom/Flash wall medians (us): "
              f"{case['timings']['custom']['wrapper_wall_us']['median']:.3f} / "
              f"{case['timings']['flash']['wrapper_wall_us']['median']:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {args.output}")
    return report


if __name__ == "__main__":
    main()

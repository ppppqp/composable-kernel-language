#!/usr/bin/env python3
"""End-to-end fixed-batch autoregressive decoding with identical CKL kernels."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

import numpy as np
import ckl
from reference import Config, MUTABLE, EPSILON, expected, make_inputs
from workloads import decode

TAIL = 256
SENTINEL = -12345


def timed(action):
    start = time.perf_counter_ns()
    value = action()
    return value, (time.perf_counter_ns() - start) / 1e6


def summary(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values), samples=values)


def plan_stats(ir):
    topology = {}
    for line in ir.splitlines():
        if '"ckl_exec.kernel"' not in line:
            continue
        node = int(re.search(r'\bid = (\d+)', line).group(1))
        match = re.search(r'dependencies = array<i64(?:: ([\d, ]+))?>', line).group(1)
        topology[node] = [int(value) for value in match.split(",")] if match else []
    heaps = [line for line in ir.splitlines() if '"ckl_exec.heap"' in line]
    temporaries = [line for line in ir.splitlines() if '"ckl_exec.resource"' in line and 'kind = "temporary"' in line]
    return dict(nodes=len(topology), edges=sum(map(len, topology.values())), topology=topology,
                temporary_count=len(temporaries),
                temporary_heap_bytes=sum(int(re.search(r'\bbytes = (\d+)', line).group(1)) for line in heaps))


class State:
    def __init__(self, device, arrays):
        self.initial = arrays
        self.owners, self.buffers, self.host = {}, {}, {}
        for name, values in arrays.items():
            host = np.full(values.size + TAIL, SENTINEL, dtype=values.dtype)
            host[:values.size] = values.ravel()
            dtype = ckl.i32 if values.dtype == np.int32 else ckl.f32
            owner = device.empty(ckl.BufferSpec(host.shape, dtype))
            owner.copy_from_host(host)
            self.owners[name] = owner
            self.host[name] = host
            self.buffers[name] = device.import_buffer(
                owner.address, ckl.BufferSpec(values.shape, dtype), owner=owner)

    def reset(self):
        for name in MUTABLE:
            self.owners[name].copy_from_host(self.host[name])

    def verify(self, reference, config):
        errors = {}
        start, end = int(self.initial["position"][0]), int(reference["position"][0])
        for name, owner in self.owners.items():
            actual = np.empty_like(self.host[name])
            owner.copy_to_host(actual)
            size = self.initial[name].size
            if not np.all(actual[size:] == SENTINEL):
                raise AssertionError(f"{name}: tail overwritten")
            actual = actual[:size].reshape(self.initial[name].shape)
            if name not in MUTABLE:
                if not np.array_equal(actual, self.initial[name]):
                    raise AssertionError(f"{name}: model weights/input metadata changed")
            elif actual.dtype == np.int32:
                if not np.array_equal(actual, reference[name]):
                    raise AssertionError(f"{name}: token/history/position mismatch\n{actual}\n{reference[name]}")
            else:
                if not np.all(np.isfinite(actual)) or not np.allclose(actual, reference[name], atol=2e-5, rtol=3e-4):
                    raise AssertionError(f"{name}: numerical mismatch, max error {np.max(np.abs(actual-reference[name]))}")
                errors[name] = float(np.max(np.abs(actual-reference[name])))
                if name in ("kc", "vc"):
                    if not np.array_equal(actual[:, :, :, :start], self.initial[name][:, :, :, :start]):
                        raise AssertionError(f"{name}: prefilled cache changed")
                    if not np.array_equal(actual[:, :, :, end:], self.initial[name][:, :, :, end:]):
                        raise AssertionError(f"{name}: unused cache suffix changed")
        return max(errors.values(), default=0.)


def run_case(device, target, config, args):
    initial, prompts = make_inputs(config, args.seed)
    references = {count: expected(initial, config, count) for count in (1, config.steps)}
    state = State(device, initial)
    signature = {name: buffer.spec for name, buffer in state.buffers.items()}
    signature.update(epsilon=ckl.f32, attention_scale=ckl.f32)
    lowered, lower_ms = timed(lambda: decode.lower(signature=signature))
    programs, invocations, stats = {}, {}, {}
    arguments = [state.buffers[p.name] if p.name in state.buffers else
                 float(EPSILON) if p.name == "epsilon" else config.head_dim ** -0.5
                 for p in lowered.signature.runtime]
    for reuse in (False, True):
        program, compile_ms = timed(lambda: lowered.compile(
            target=target, fuse_pointwise=False, reuse_storage=reuse))
        bound, bind_ms = timed(lambda: program.bind(*arguments))
        programs[reuse], invocations[reuse] = program, bound
        stats[reuse] = dict(compile_ms=compile_ms, bind_ms=bind_ms, **plan_stats(program.executable_ir))
        if bound.node_count != 21 * config.layers + 6:
            raise AssertionError("unexpected decoder dispatch count")
    if programs[False].prepared_ir != programs[True].prepared_ir:
        raise AssertionError("storage ablation changed the prepared device kernels")
    if stats[True]["temporary_heap_bytes"] > stats[False]["temporary_heap_bytes"]:
        raise AssertionError("storage reuse increased temporary memory")

    # Same code and same buffers. Only dependency topology, storage placement,
    # and number of complete steps per graph submission change.
    variants = [("sequential", False, "ordinary", 1),
                ("capture", False, "capture", 1),
                ("inferred", False, "graph", 1),
                ("inferred_reuse", True, "graph", 1),
                ("batched_capture", True, "capture", config.steps),
                ("batched_inferred", True, "graph", config.steps)]
    metrics = {name: {key: [] for key in ("stream_us", "wall_us", "generation_ms")}
               for name, *_ in variants}
    setup, errors = {}, {}
    # All setup and warmup precedes paired measurement. Capture records work
    # without executing it; the benchmark then launches it once.
    for name, reuse, mode, group in variants:
        bound = invocations[reuse]
        state.reset()
        first = bound.benchmark(iterations=1, mode=mode, batch_size=group)
        setup[name] = first["graph_setup_ms"]
        state.verify(references[group], config)
        state.reset()
        bound.benchmark(iterations=config.steps // group, mode=mode, batch_size=group)
        errors[name] = state.verify(references[config.steps], config)
        for _ in range(args.warmup):
            state.reset()
            bound.benchmark(iterations=config.steps // group, mode=mode, batch_size=group)
    for sample in range(args.samples):
        offset = sample % len(variants)
        order = variants[offset:] + variants[:offset]
        if sample % 2:
            order = list(reversed(order))
        for name, reuse, mode, group in order:
            bound = invocations[reuse]
            state.reset()
            timing = bound.benchmark(iterations=config.steps // group, mode=mode, batch_size=group)
            metrics[name]["stream_us"].append(timing["stream_us"])
            metrics[name]["wall_us"].append(timing["wall_us"])
            errors[name] = max(errors[name], state.verify(references[config.steps], config))
            state.reset()
            begin = time.perf_counter_ns()
            for _ in range(config.steps // group):
                bound.run(mode=mode, batch_size=group)
            metrics[name]["generation_ms"].append((time.perf_counter_ns() - begin) / 1e6)
            errors[name] = max(errors[name], state.verify(references[config.steps], config))
    results = []
    for name, reuse, mode, group in variants:
        row = dict(variant=name, reuse_storage=reuse, mode=mode, steps_per_submission=group,
                   correct=True, max_abs_error=errors[name], graph_setup_ms=setup[name],
                   timings={key: summary(value) for key, value in metrics[name].items()})
        results.append(row)
        print(f"B={config.batch:2} context={config.context:3} {name:18} "
              f"stream={row['timings']['stream_us']['median']:8.2f} us/step "
              f"generation={row['timings']['generation_ms']['median']:7.3f} ms "
              f"heap={stats[reuse]['temporary_heap_bytes']} correct=true", flush=True)
    history = references[config.steps]["history"][config.context - 1:config.context - 1 + config.steps]
    return dict(config=asdict(config), lower_ms=lower_ms, plans={str(k): v for k, v in stats.items()},
                kernel_ir_sha256=hashlib.sha256(programs[False].prepared_ir.encode()).hexdigest(),
                prompts=prompts.tolist(), generated_tokens=history.tolist(),
                persistent_and_weight_bytes=sum(v.nbytes for v in initial.values()), results=results)


def render(data):
    if not data.get("complete") or len(data["cases"]) != len(data["configuration"]["cases"]):
        raise ValueError("incomplete decoder matrix")
    lines = ["# CKL autoregressive decoder measurements", "",
             f"GPU: {data['device']}. Native build: {data['build_type']}. "
             f"Python {data['python']}, NumPy {data['numpy']}.", "",
             "All variants execute identical unfused FP32 device kernels. The model has random "
             "weights and is an orchestration experiment, not a trained language model or a "
             "competitive attention implementation. Prompt prefill, transfers, resets, compilation, "
             "and graph setup are outside steady decode timing.", "",
             "Native stream/wall times are per complete decode step (one token per request) and "
             "include native submission gaps. Each sample averages a growing-context generation "
             "sequence, rather than repeatedly decoding the same token/position. Python generation latency covers all generated "
             "steps, with one completion wait per submission; batched variants submit all steps "
             "at once. That metric measures generation completion, not streamed token latency.", "",
             "| Batch | Initial context | Variant | Stream µs/step | Native wall µs/step | "
             "Python generation ms | Temp heap bytes | Setup ms |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    expected_labels = {"sequential", "capture", "inferred", "inferred_reuse", "batched_capture", "batched_inferred"}
    for case in data["cases"]:
        if {r["variant"] for r in case["results"]} != expected_labels or len(case["results"]) != 6:
            raise ValueError("incomplete variant matrix")
        for row in case["results"]:
            if not row["correct"] or any(len(metric["samples"]) != data["configuration"]["samples"]
                                        or not all(np.isfinite(v) and v > 0 for v in metric["samples"])
                                        for metric in row["timings"].values()):
                raise ValueError("invalid decoder results")
            c, t = case["config"], row["timings"]
            heap = case["plans"][str(row["reuse_storage"])]["temporary_heap_bytes"]
            lines.append(f"| {c['batch']} | {c['context']} | {row['variant']} | "
                         f"{t['stream_us']['median']:.2f} | {t['wall_us']['median']:.2f} | "
                         f"{t['generation_ms']['median']:.3f} | {heap:,} | {row['graph_setup_ms']:.3f} |")
    lines += ["", "| Batch | Context | Sequential/inferred | Capture/inferred | "
              "Sequential/batched inferred | Heap without → with reuse |", "|---:|---:|---:|---:|---:|---:|"]
    for case in data["cases"]:
        rows = {r["variant"]: r["timings"]["stream_us"]["median"] for r in case["results"]}
        lines.append(f"| {case['config']['batch']} | {case['config']['context']} | "
                     f"{rows['sequential']/rows['inferred']:.2f}x | {rows['capture']/rows['inferred']:.2f}x | "
                     f"{rows['sequential']/rows['batched_inferred']:.2f}x | "
                     f"{case['plans']['False']['temporary_heap_bytes']:,} → "
                     f"{case['plans']['True']['temporary_heap_bytes']:,} |")
    lines += ["", "Every timed sample and Python generation sequence passed exact token/history/position "
              "checks and numerical logits/KV-cache checks against an independent NumPy decoder. "
              "Prefilled cache, unused cache suffix, model weights, and allocation tail sentinels "
              "were checked for unintended writes. Raw JSON retains samples, topology, generated "
              "tokens, setup costs, source hashes, and maximum errors.", "",
              "Fixed batches, uniform context lengths, contiguous MHA cache, greedy decoding, "
              "small random model, and unlocked GPU clocks. No serving arrival scheduler, continuous "
              "batching, paged cache, early EOS, or production-serving speedup is claimed. Batched "
              "graphs delay host observation until the group completes.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--chip", default="sm_120")
    parser.add_argument("--cases", nargs="+", default=["1x1", "1x8", "8x8", "1x64", "8x64", "32x64"],
                        help="batch x initial context")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ffn", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 1:
        parser.error("samples and warmup must be positive")
    cases = []
    for shape in args.cases:
        b, context = map(int, shape.split("x"))
        cases.append(Config(b, args.hidden, args.heads, args.layers, args.ffn,
                            args.vocab, context, args.steps))
    device = ckl.Device()
    target = ckl.NVIDIATarget(args.chip, "+ptx87", args.toolkit)
    lib = Path(device._lib._name).resolve()
    cache = lib.parents[2] / "CMakeCache.txt"
    build = next((line.split("=", 1)[1] for line in cache.read_text().splitlines()
                  if line.startswith("CMAKE_BUILD_TYPE:")), "unknown") if cache.exists() else "unknown"
    files = [Path(__file__), Path(__file__).with_name("workloads.py"), Path(__file__).with_name("reference.py")]
    data = dict(complete=False, timestamp=datetime.now(timezone.utc).isoformat(), device=device.name,
                build_type=build, native_library=str(lib), native_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
                source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                target=target.cache_identity(), configuration={k: v for k, v in vars(args).items()
                if k not in ("toolkit", "output")}, cases=[])
    info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    data["nvidia_smi"] = info.stdout if not info.returncode else info.stderr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for config in cases:
        print(f"Compiling and checking {config}...", flush=True)
        data["cases"].append(run_case(device, target, config, args))
        args.output.write_text(json.dumps(data, indent=2) + "\n")
    data["complete"] = True
    report = render(data)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(report)
    print(f"Saved {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()

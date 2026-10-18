#!/usr/bin/env python3
"""Independent-request measurements and warmed, correctness-checked profiler ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import time

import ckl
from ckl.frontend.emitter import identity
from ckl.frontend.source import Definition
import workloads
from reference import EPSILON, expected, make_inputs
from run import State, extended_cases, plan_stats, summary


def prepare(device, target, config, requests, seed, reuse):
    initial = [make_inputs(config, seed, prompt_seed=seed + i)[0] for i in range(requests)]
    states = []
    for arrays in initial:
        states.append(State(device, arrays, shared=states[0] if states else None))
    signature = {name: buffer.spec for name, buffer in states[0].buffers.items()}
    signature.update(epsilon=ckl.f32, attention_scale=ckl.f32)
    lowered = workloads.decode.lower(signature=signature)
    program = lowered.compile(target=target, fuse_pointwise=False, reuse_storage=reuse)
    invocations = []
    for state in states:
        arguments = [state.buffers[p.name] if p.name in state.buffers else
                     float(EPSILON) if p.name == "epsilon" else config.head_dim ** -0.5
                     for p in lowered.signature.runtime]
        invocations.append(program.bind(*arguments))
    symbols = {}
    definitions = [value for value in vars(workloads).values()
                   if isinstance(value, Definition) and value.kind == "kernel"]
    for spec in lowered.specializations:
        matches = [d.name for d in definitions if identity(d, spec.signature) == spec.identity]
        if len(matches) != 1:
            raise AssertionError(f"cannot name kernel {spec.symbol}")
        symbols[spec.symbol] = matches[0]
    return states, invocations, dict(
        config=asdict(config), requests=requests, reuse_storage=reuse, device=device.name,
        shared_readonly_weights=True, independent_prompt_seeds=list(range(seed, seed + requests)),
        target=target.cache_identity(), symbols=symbols, plan=plan_stats(program.executable_ir),
        kernel_ir_sha256=hashlib.sha256(program.prepared_ir.encode()).hexdigest(),
        native_sha256=hashlib.sha256(Path(device._lib._name).read_bytes()).hexdigest(),
        source_hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ("experiment.py", "run.py", "workloads.py", "reference.py")})


def burst(invocations, config, mode, group, parallel, executor):
    def execute(bound):
        return bound.benchmark(iterations=config.steps // group, mode=mode, batch_size=group)
    if parallel:
        # Each invocation owns a stream, all mutable state, and its scratch heap.
        # CDLL releases the GIL. No Python work occurs between native step submissions.
        barrier = threading.Barrier(len(invocations) + 1)
        def worker(bound):
            barrier.wait(timeout=30)
            return execute(bound)
        futures = [executor.submit(worker, bound) for bound in invocations]
        start = time.perf_counter_ns()
        barrier.wait(timeout=30)
        timings = [future.result() for future in futures]
    else:
        start = time.perf_counter_ns()
        timings = [execute(bound) for bound in invocations]
    return (time.perf_counter_ns() - start) / 1e6, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--chip", default="sm_120")
    parser.add_argument("--case", choices=[n for n, _ in extended_cases(16)], default="small")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--profile", action="store_true", help="one CUDA profiler API range")
    parser.add_argument("--mode", choices=("ordinary", "capture", "graph"), default="graph")
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.requests, args.group, args.samples, args.warmup) < 1 or args.steps % args.group:
        parser.error("positive counts and steps divisible by group are required")
    config = dict(extended_cases(args.steps))[args.case]
    device = ckl.Device()
    target = ckl.NVIDIATarget(args.chip, "+ptx87", args.toolkit)
    states, invocations, data = prepare(device, target, config, args.requests, args.seed, args.reuse)
    references = [expected(state.initial, config, config.steps) for state in states]
    def reset():
        for state in states:
            state.reset()
    def verify():
        return max(state.verify(ref, config) for state, ref in zip(states, references))
    choices = [(args.mode, args.group, args.parallel)] if args.profile else [
        (mode, group, parallel) for mode, group in (
            ("ordinary", 1), ("capture", 1), ("graph", 1), ("graph", args.steps))
        for parallel in (False, True)]
    data.update(complete=False, case=args.case, samples=args.samples, warmup=args.warmup, results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.requests) as executor:
        # Construct all graphs and warm workers outside both capture and timing.
        for mode, group, parallel in choices:
            for _ in range(args.warmup):
                reset()
                burst(invocations, config, mode, group, parallel, executor)
                verify()
        if args.profile:
            reset()
            driver = ctypes.CDLL("libcuda.so.1")
            for name in ("cuProfilerStart", "cuProfilerStop"):
                getattr(driver, name).argtypes = []
                getattr(driver, name).restype = ctypes.c_int
            if driver.cuProfilerStart() != 0:
                raise RuntimeError("cuProfilerStart failed")
            try:
                elapsed, timings = burst(invocations, config, args.mode, args.group, args.parallel, executor)
            finally:
                if driver.cuProfilerStop() != 0:
                    raise RuntimeError("cuProfilerStop failed")
            data.update(mode=args.mode, group=args.group, parallel=args.parallel,
                        wall_ms=elapsed, native_timings=timings, max_abs_error=verify())
        else:
            rows = {choice: dict(wall_ms=[], native_stream_us=[], max_abs_error=0.) for choice in choices}
            for sample in range(args.samples):
                offset = sample % len(choices)
                order = choices[offset:] + choices[:offset]
                for choice in order if sample % 2 == 0 else reversed(order):
                    reset()
                    elapsed, timings = burst(invocations, config, *choice, executor)
                    row = rows[choice]
                    row["wall_ms"].append(elapsed)
                    row["native_stream_us"].append([t["stream_us"] for t in timings])
                    row["max_abs_error"] = max(row["max_abs_error"], verify())
            for (mode, group, parallel), row in rows.items():
                row.update(mode=mode, group=group, parallel=parallel, correct=True,
                           wall_ms=summary(row["wall_ms"]))
                row["tokens_per_second"] = config.batch * args.requests * config.steps * 1000 / row["wall_ms"]["median"]
                data["results"].append(row)
                print(f"{mode=} {group=} {parallel=} {row['wall_ms']['median']:.3f} ms "
                      f"{row['tokens_per_second']:.0f} tokens/s", flush=True)
    data["complete"] = True
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

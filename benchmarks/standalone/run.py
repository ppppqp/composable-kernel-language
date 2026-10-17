#!/usr/bin/env python3
"""Reproducible, correctness-checked benchmarks for CKL source programs."""
from array import array
import argparse
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import time

import ckl
from workloads import CASES


def f32(value):
    return ctypes.c_float(value).value


def affine(values, scale, bias):
    return array("f", (f32(f32(value * scale) + bias) for value in values))


def reference(name, src, other):
    if name == "single_affine":
        return affine(src, 0.875, 0.03125)
    if name in ("chain_six", "fused_chain_six"):
        result = src
        for _ in range(6):
            result = affine(result, 0.875, 0.03125)
        return result
    if name == "fork_join":
        a, b = affine(src, 0.5, 0.125), affine(other, -0.25, 0.5)
        c = array("f", (x + y for x, y in zip(a, b)))
        d, e = affine(c, 0.75, -0.0625), affine(c, 1.25, 0.03125)
        return array("f", (x + y for x, y in zip(d, e)))
    result = src
    for _ in range(2):
        result = array("f", [result[0]] + [
            f32(f32(f32(result[i-1] * 0.25) + f32(result[i] * 0.5)) +
                f32(result[i+1] * 0.25)) for i in range(1, len(result)-1)
        ] + [result[-1]])
    return result


def plan_stats(text):
    kernels = [line for line in text.splitlines() if '"ckl_exec.kernel"' in line]
    dependencies = []
    topology = {}
    for line in kernels:
        node = int(re.search(r'\bid = (\d+)', line).group(1))
        match = re.search(r'dependencies = array<i64(?:: ([\d, ]+))?>', line)
        deps = [int(x) for x in match.group(1).split(",")] if match.group(1) else []
        dependencies.append(deps)
        topology[node] = deps
    heaps = [line for line in text.splitlines() if '"ckl_exec.heap"' in line]
    temporaries = [line for line in text.splitlines()
                   if '"ckl_exec.resource"' in line and 'kind = "temporary"' in line]
    baseline = 0
    for line in temporaries:
        size = int(re.search(r'\bbytes = (\d+)', line).group(1))
        alignment = int(re.search(r'\balignment = (\d+)', line).group(1))
        baseline += (size + alignment - 1) // alignment * alignment
    return dict(nodes=len(kernels), edges=sum(map(len, dependencies)),
                roots=sum(not deps for deps in dependencies), topology=topology,
                temporary_count=len(temporaries), baseline_temp_bytes=baseline,
                planned_temp_bytes=sum(int(re.search(r'\bbytes = (\d+)', h).group(1)) for h in heaps))


def summary(samples):
    return dict(median=statistics.median(samples), minimum=min(samples), maximum=max(samples),
                samples=samples)


def timed(action):
    start = time.perf_counter_ns()
    value = action()
    return value, (time.perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--chip", default="sm_120")
    parser.add_argument("--sizes", type=int, nargs="+", default=[257, 4096, 65536, 1048576])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--latency-iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 3 or min(args.iterations, args.samples, args.warmup, args.latency_iterations) < 1:
        parser.error("sizes must be >= 3 and measurement counts must be positive")
    target = ckl.NVIDIATarget(args.chip, "+ptx87", args.toolkit)
    device, initialize_ms = timed(ckl.Device)
    library = Path(device._lib._name).resolve()
    cache = library.parents[2] / "CMakeCache.txt"
    build = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            for key in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "MLIR_DIR"):
                if line.startswith(key + ":"):
                    build[key] = line.split("=", 1)[1]
    results = dict(complete=False, timestamp=datetime.now(timezone.utc).isoformat(), device=device.name,
                   python=platform.python_version(), platform=platform.platform(),
                   native_library=str(library), native_build=build,
                   workloads_sha256=hashlib.sha256(Path(__file__).with_name("workloads.py").read_bytes()).hexdigest(),
                   target=target.cache_identity(), device_initialize_ms=initialize_ms,
                   configuration={key: value for key, value in vars(args).items()
                                  if key not in ("toolkit", "output")}, results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        inputs = array("f", ((index % 97) / 128 for index in range(size)))
        other = array("f", ((index % 113) / 256 for index in range(size)))
        for name in args.cases:
            spec = ckl.BufferSpec((size,), ckl.f32)
            signature = {"src": spec, "dst": spec}
            if name == "fork_join":
                signature["other"] = spec
            lowered, lower_ms = timed(lambda: CASES[name].lower(signature=signature))
            program, compile_ms = timed(lambda: lowered.compile(target=target))
            stats = plan_stats(program.executable_ir)
            assert stats["nodes"] == {"single_affine": 1, "chain_six": 6,
                   "fused_chain_six": 1, "fork_join": 6, "stencil_two": 2}[name]
            if name == "fork_join":
                assert stats["roots"] == 2, "independent roots were serialized"
            expected = reference(name, inputs, other)
            # Keep a guarded tail outside the logical buffer's specialized shape.
            full_spec = ckl.BufferSpec((size + 256,), ckl.f32)
            input_owner, other_owner, output_owner = [device.empty(full_spec) for _ in range(3)]
            full_input = array("f", inputs)
            full_input.extend([-12345.] * 256)
            full_other = array("f", other)
            full_other.extend([-12345.] * 256)
            input_owner.copy_from_host(full_input)
            other_owner.copy_from_host(full_other)
            src = device.import_buffer(input_owner.address, spec, owner=input_owner)
            second = device.import_buffer(other_owner.address, spec, owner=other_owner)
            dst = device.import_buffer(output_owner.address, spec, owner=output_owner)
            arguments = (src, second, dst) if name == "fork_join" else (src, dst)
            bound, bind_ms = timed(lambda: program.bind(*arguments))
            probe, rebind_ms = timed(lambda: program.bind(*arguments))
            del probe
            gc.collect()
            actual = array("f", [0.] * (size + 256))
            poison = array("f", [-12345.] * (size + 256))

            def verify():
                output_owner.copy_to_host(actual)
                error = 0.
                for index, (value, ref) in enumerate(zip(actual, expected)):
                    error = max(error, abs(value - ref))
                    if not math.isfinite(value) or abs(value - ref) > 2e-6 + 2e-5 * abs(ref):
                        raise AssertionError(f"{name}/{size} at {index}: {value} != {ref}")
                assert actual[size:] == poison[size:], "partial-block tail guard overwritten"
                input_owner.copy_to_host(actual)
                assert actual == full_input, "source buffer changed"
                other_owner.copy_to_host(actual)
                assert actual == full_other, "second source buffer changed"
                return error

            # Check both paths independently with a poisoned destination.
            errors = []
            for graph in (False, True):
                output_owner.copy_from_host(poison)
                if graph:
                    first_graph = bound.benchmark(iterations=1, graph=True)
                else:
                    bound.run()
                errors.append(verify())
                for _ in range(args.warmup):
                    bound.run(graph=graph)
            measurements = {mode: dict(stream_us=[], wall_us=[], python_latency_us=[])
                            for mode in ("ordinary", "graph")}
            for sample in range(args.samples):
                # Alternate ordering to reduce systematic clock/temperature bias.
                modes = (False, True) if sample % 2 == 0 else (True, False)
                for graph in modes:
                    mode = "graph" if graph else "ordinary"
                    burst = bound.benchmark(iterations=args.iterations, graph=graph)
                    for key in ("stream_us", "wall_us"):
                        measurements[mode][key].append(burst[key])
                    start = time.perf_counter_ns()
                    for _ in range(args.latency_iterations):
                        bound.run(graph=graph)
                    measurements[mode]["python_latency_us"].append(
                        (time.perf_counter_ns() - start) / args.latency_iterations / 1000)
            # Keep CPU reference scans outside timing samples to avoid long
            # host-induced idle gaps between the paired GPU measurements.
            for graph in (False, True):
                output_owner.copy_from_host(poison)
                bound.run(graph=graph)
                errors.append(verify())
            row = dict(case=name, elements=size, lower_ms=lower_ms, compile_ms=compile_ms,
                       first_bind_ms=bind_ms, rebind_ms=rebind_ms,
                       graph_setup_ms=first_graph["graph_setup_ms"], first_graph=first_graph,
                       max_abs_error=max(errors), correct=True, **stats)
            row["timings"] = {mode: {metric: summary(samples) for metric, samples in values.items()}
                              for mode, values in measurements.items()}
            row["stream_speedup"] = (statistics.median(measurements["ordinary"]["stream_us"]) /
                                     statistics.median(measurements["graph"]["stream_us"]))
            results["results"].append(row)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(f"{name:17s} N={size:8d} ordinary={row['timings']['ordinary']['stream_us']['median']:8.3f} us "
                  f"graph={row['timings']['graph']['stream_us']['median']:8.3f} us "
                  f"speedup={row['stream_speedup']:.2f}x temp={stats['planned_temp_bytes']} correct=true",
                  flush=True)
            del bound, program, lowered, src, second, dst, arguments
            del input_owner, other_owner, output_owner
            gc.collect()
    results["complete"] = True
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

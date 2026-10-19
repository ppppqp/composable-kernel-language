#!/usr/bin/env python3
"""Measure repeated compile/bind setup separately from steady-state decode."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import ckl
import numpy as np
from reference import EPSILON, make_inputs
from run import State, extended_cases, plan_stats, summary, timed
from workloads import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--chip", default="sm_120")
    parser.add_argument("--cases", nargs="+", choices=[n for n, _ in extended_cases(16)],
                        default=["small", "deep"])
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--binds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.warmup, args.binds) < 1:
        parser.error("samples, warmup, and binds must be positive")
    target = ckl.NVIDIATarget(args.chip, "+ptx87", args.toolkit)
    with ckl.Device() as device:
        data = dict(complete=False, device=device.name, python=platform.python_version(),
                    numpy=np.__version__, target=target.cache_identity(),
                    native_sha256=hashlib.sha256(device.native_library.read_bytes()).hexdigest(),
                    settings=dict(cases=args.cases, samples=args.samples, warmup=args.warmup,
                                  binds=args.binds), cases=[])
        for name in args.cases:
            config = dict(extended_cases(16))[name]
            state = State(device, make_inputs(config, 123)[0])
            signature = {n: buffer.spec for n, buffer in state.buffers.items()}
            signature.update(epsilon=ckl.f32, attention_scale=ckl.f32)
            lowered = decode.lower(signature=signature)
            arguments = [state.buffers[p.name] if p.name in state.buffers else
                         float(EPSILON) if p.name == "epsilon" else config.head_dim ** -0.5
                         for p in lowered.signature.runtime]
            for reuse in (False, True):
                metrics = {key: [] for key in ("compile_ms", "first_bind_ms", "rebind_ms")}
                invariant = None
                for sample in range(args.warmup + args.samples):
                    program, compile_ms = timed(lambda: lowered.compile(
                        target=target, fuse_pointwise=False, reuse_storage=reuse))
                    with program:
                        current = dict(plan=plan_stats(program.executable_ir),
                                       kernel_ir_sha256=hashlib.sha256(
                                           program.prepared_ir.encode()).hexdigest())
                        if invariant is not None and current != invariant:
                            raise AssertionError("compilation changed kernel IR or plan")
                        invariant = current
                        invocation, first_bind_ms = timed(lambda: program.bind(*arguments))
                        invocation.close()
                        rebinds = []
                        for _ in range(args.binds):
                            invocation, bind_ms = timed(lambda: program.bind(*arguments))
                            invocation.close()
                            rebinds.append(bind_ms)
                    if sample >= args.warmup:
                        metrics["compile_ms"].append(compile_ms)
                        metrics["first_bind_ms"].append(first_bind_ms)
                        metrics["rebind_ms"].extend(rebinds)
                row = dict(name=name, reuse_storage=reuse, **invariant,
                           timings={key: summary(values) for key, values in metrics.items()})
                data["cases"].append(row)
                print(name, reuse, {key: round(value["median"], 4)
                                    for key, value in row["timings"].items()}, flush=True)
        data["complete"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()

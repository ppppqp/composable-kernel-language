#!/usr/bin/env python3
"""Collect independent-request timings and representative Nsight Systems traces.

Run with the same MLIR PYTHONPATH/CKL_NATIVE_LIBRARY extension path as run.py. Measurements and
profiles execute serially; Nsight instrumentation never supplies speedup numbers.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


PROFILES = [
    ("small_sequential", "small", "ordinary", 1, 1, False, False),
    ("small_capture", "small", "capture", 1, 1, False, False),
    ("small_inferred", "small", "graph", 1, 1, False, False),
    ("small_reuse", "small", "graph", 1, 1, True, False),
    ("small_group16", "small", "graph", 16, 1, True, False),
    ("long_capture", "long_context", "capture", 1, 1, False, False),
    ("long_inferred", "long_context", "graph", 1, 1, False, False),
    ("wide_capture", "wide", "capture", 1, 1, False, False),
    ("wide_inferred", "wide", "graph", 1, 1, False, False),
    ("vocab_inferred", "large_vocab", "graph", 1, 1, False, False),
    ("requests_serial", "small", "graph", 1, 4, True, False),
    ("requests_parallel", "small", "graph", 1, 4, True, True),
    ("requests_group16", "small", "graph", 16, 4, True, True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--chip", default="sm_120")
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-requests", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="run the extended single-invocation matrix first")
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--resume", action="store_true", help="retain already exported profiles")
    parser.add_argument("--profiles", nargs="*", choices=[p[0] for p in PROFILES])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = Path(__file__).with_name("experiment.py")
    command = [sys.executable, str(experiment), "--toolkit", str(args.toolkit), "--chip", args.chip]
    if args.matrix:
        subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")), "--toolkit", str(args.toolkit),
                        "--chip", args.chip, "--suite", "extended", "--output",
                        str(args.output / "matrix.json")], check=True)
    if not args.skip_requests:
        for case in ("small", "wide"):
            for requests in (2, 4, 8):
                subprocess.run(command + ["--case", case, "--requests", str(requests), "--reuse",
                                          "--output", str(args.output / f"requests-{case}-{requests}.json")], check=True)
        for case in ("batch4", "wide_batch4"):
            subprocess.run(command + ["--case", case, "--requests", "1", "--reuse",
                                      "--output", str(args.output / f"requests-{case}-1.json")], check=True)
    if args.skip_profiles:
        return
    traces = args.output / "traces"
    traces.mkdir(exist_ok=True)
    manifest = dict(complete=False, profiler=subprocess.check_output([args.nsys, "--version"], text=True).strip(),
                    profiles=[])
    manifest_path = traces / "manifest.json"
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, case, mode, group, requests, reuse, parallel in PROFILES:
        if args.profiles is not None and name not in args.profiles:
            continue
        if args.resume and any(p["name"] == name for p in manifest["profiles"]):
            continue
        prefix = traces / name
        capture = command + ["--profile", "--case", case, "--mode", mode, "--group", str(group),
                             "--requests", str(requests), "--output", str(prefix.with_suffix(".json"))]
        if reuse:
            capture.append("--reuse")
        if parallel:
            capture.append("--parallel")
        profile = [args.nsys, "profile", "--trace=cuda", "--cuda-graph-trace=node", "--sample=none",
                   "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                   "--force-overwrite=true", "--output", str(prefix), *capture]
        print(f"Profiling {name}...", flush=True)
        with prefix.with_suffix(".log").open("w") as log:
            subprocess.run(profile, check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([args.nsys, "export", "--type=sqlite", "--force-overwrite=true",
                        "--quiet=true",
                        "--output", str(prefix.with_suffix(".sqlite")), str(prefix.with_suffix(".nsys-rep"))], check=True)
        manifest["profiles"].append(dict(name=name, command=profile))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

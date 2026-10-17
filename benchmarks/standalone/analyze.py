#!/usr/bin/env python3
"""Render retained benchmark measurements as a compact comparison."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if not data.get("complete") or len(data["results"]) != (
        len(data["configuration"]["sizes"]) * len(data["configuration"]["cases"])
    ):
        raise ValueError("incomplete benchmark matrix; rerun failed cases before reporting")
    lines = ["# Standalone CKL measurements", "",
             f"GPU: {data['device']}. Python {data['python']}. "
             f"Native build: {data['native_build'].get('CMAKE_BUILD_TYPE', 'unknown')}.", "",
             "Times below are medians in microseconds per complete program. CUDA-event stream "
             "time includes native host submission gaps. These are warm repeated-buffer measurements, "
             "not isolated kernel execution or cold-memory bandwidth measurements.", "",
             "| Case | Elements | Ordinary stream µs | Graph stream µs | Graph speedup | "
             "Python ordinary µs | Python graph µs |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in data["results"]:
        t = row["timings"]
        ordinary, graph = t["ordinary"], t["graph"]
        lines.append(f"| {row['case']} | {row['elements']:,} | "
                     f"{ordinary['stream_us']['median']:.2f} | {graph['stream_us']['median']:.2f} | "
                     f"{row['stream_speedup']:.2f}x | {ordinary['python_latency_us']['median']:.2f} | "
                     f"{graph['python_latency_us']['median']:.2f} |")
    lines += ["", "| Case | Elements | Lower ms | Compile/JIT ms | First bind ms | "
              "Rebind ms | Graph instantiate ms | Estimated graph break-even repeats | "
              "Temporary bytes, baseline → planned |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["results"]:
        t = row["timings"]
        saved = t["ordinary"]["wall_us"]["median"] - t["graph"]["wall_us"]["median"]
        first_excess = max(0., row["first_graph"]["wall_us"] - t["graph"]["wall_us"]["median"])
        upfront = row["graph_setup_ms"] * 1000 + first_excess
        breakeven = str(math.ceil(upfront / saved)) if saved > 0 else "no steady saving"
        lines.append(f"| {row['case']} | {row['elements']:,} | {row['lower_ms']:.2f} | "
                     f"{row['compile_ms']:.2f} | {row['first_bind_ms']:.3f} | "
                     f"{row['rebind_ms']:.3f} | {row['graph_setup_ms']:.3f} | {breakeven} | "
                     f"{row['baseline_temp_bytes']:,} → {row['planned_temp_bytes']:,} |")
    lines += ["", "Estimated break-even includes measured graph instantiation and excess first-launch "
              "wall time, amortized against native burst wall-time savings. It excludes compilation and "
              "binding, which both execution modes share. It is an estimate from separate measurements.", "",
              f"All {len(data['results'])} cases passed full output, tail-sentinel, and unchanged-input checks. "
              "Raw JSON retains individual samples, ranges, topology, setup costs, and maximum errors.", ""]
    rows = {(r["case"], r["elements"]): r for r in data["results"]}
    lines += ["| Elements | Chain graph / manually fused graph |", "|---|---:|"]
    for size in data["configuration"]["sizes"]:
        if ("chain_six", size) in rows and ("fused_chain_six", size) in rows:
            chain = rows["chain_six", size]["timings"]["graph"]["stream_us"]["median"]
            fused = rows["fused_chain_six", size]["timings"]["graph"]["stream_us"]["median"]
            lines.append(f"| {size:,} | {chain / fused:.2f}x |")
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

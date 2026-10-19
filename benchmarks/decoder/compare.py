#!/usr/bin/env python3
"""Compare two complete decoder matrices, including correctness and plan invariants."""

import argparse
import json
import statistics
from pathlib import Path


def compare(before, after):
    if not before.get("complete") or not after.get("complete"):
        raise ValueError("both decoder matrices must be complete")
    for key in ("device", "target", "configuration", "numpy", "python"):
        if before[key] != after[key]:
            raise ValueError(f"incompatible benchmark {key}")
    old_cases = {case["name"]: case for case in before["cases"]}
    new_cases = {case["name"]: case for case in after["cases"]}
    if old_cases.keys() != new_cases.keys():
        raise ValueError("case sets differ")
    rows = []
    for name, old in old_cases.items():
        new = new_cases[name]
        for key in ("config", "prompts", "generated_tokens"):
            if old[key] != new[key]:
                raise ValueError(f"{name}: {key} differ")
        for reuse in ("False", "True"):
            for key in ("nodes", "edges", "topology", "temporary_count", "temporary_heap_bytes"):
                if old["plans"][reuse][key] != new["plans"][reuse][key]:
                    raise ValueError(f"{name}: reuse={reuse} {key} differ")
        old_rows = {row["variant"]: row for row in old["results"]}
        new_rows = {row["variant"]: row for row in new["results"]}
        if old_rows.keys() != new_rows.keys():
            raise ValueError(f"{name}: variant sets differ")
        for variant, old_row in old_rows.items():
            new_row = new_rows[variant]
            if not old_row["correct"] or not new_row["correct"]:
                raise ValueError(f"{name}/{variant}: correctness failed")
            for key in ("mode", "reuse_storage", "steps_per_submission"):
                if old_row[key] != new_row[key]:
                    raise ValueError(f"{name}/{variant}: {key} differs")
            delta = {
                key: 100
                * (new_row["timings"][key]["median"] / old_row["timings"][key]["median"] - 1)
                for key in ("stream_us", "wall_us", "generation_ms")
            }
            rows.append((name, variant, old_row, new_row, delta))
    lines = [
        "# Decoder binding migration comparison",
        "",
        (
            "Positive changes mean slower execution. Comparisons use per-variant medians; "
            "they are descriptive measurements, not statistical equivalence tests."
        ),
        "",
        (
            f"All {len(rows)} case/variant pairs passed both runners' reference checks. "
            "Prompts, generated tokens, dependency topologies, dispatch counts, temporary counts, "
            "and temporary heap bytes match exactly."
        ),
        "",
        "| Metric | Minimum change | Median change | Maximum change |",
        "|---|---:|---:|---:|",
    ]
    for key in ("stream_us", "wall_us", "generation_ms"):
        values = [row[4][key] for row in rows]
        lines.append(
            f"| {key} | {min(values):+.2f}% | {statistics.median(values):+.2f}% | "
            f"{max(values):+.2f}% |"
        )
    lines += [
        "",
        "## Frontend lowering setup",
        "",
        "Lowering is excluded from steady decode timing.",
        "",
        "| Case | Before ms | After ms |",
        "|---|---:|---:|",
    ]
    for name, old in old_cases.items():
        lines.append(f"| {name} | {old['lower_ms']:.2f} | {new_cases[name]['lower_ms']:.2f} |")
    lines += [
        "",
        "## Per-variant medians",
        "",
        "| Case | Variant | Before stream µs/step | After stream µs/step | Stream change | Generation change |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, variant, old, new, delta in rows:
        lines.append(
            f"| {name} | {variant} | {old['timings']['stream_us']['median']:.2f} | "
            f"{new['timings']['stream_us']['median']:.2f} | {delta['stream_us']:+.2f}% | "
            f"{delta['generation_ms']:+.2f}% |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FlyDSL Stage 1 samples")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with args.csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            grouped[(row["metric"], row["mode"])].append(float(row["value_us"]))

    print("metric,mode,median_us,p10_us,p90_us,over_flydsl_eager")
    for (metric, mode), values in sorted(grouped.items()):
        if metric not in {"gpu_replay_us", "cpu_submit_us"}:
            continue
        baseline = grouped[(metric, "flydsl_eager")]
        median = statistics.median(values)
        ratio = median / statistics.median(baseline)
        print(
            f"{metric},{mode},{median:.6f},{percentile(values, 0.1):.6f},"
            f"{percentile(values, 0.9):.6f},{ratio:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

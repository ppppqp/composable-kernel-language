#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_MODES = frozenset({"sequential", "capture", "streams", "explicit"})


@dataclass(frozen=True)
class WorkloadResult:
    workload: str
    capture_ms: float
    explicit_ms: float
    streams_ms: float
    speedup: float
    correct: bool
    complete: bool


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            rows.extend(csv.DictReader(line for line in source if not line.startswith("#")))
    return rows


def summarize(rows: Iterable[dict[str, str]]) -> list[WorkloadResult]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["workload"], {})[row["mode"]] = row

    results: list[WorkloadResult] = []
    for workload, modes in sorted(grouped.items()):
        complete = REQUIRED_MODES.issubset(modes)
        correct = complete and all(modes[mode]["correct"].lower() == "true" for mode in REQUIRED_MODES)
        capture_ms = float(modes["capture"]["median_ms"]) if "capture" in modes else float("nan")
        explicit_ms = float(modes["explicit"]["median_ms"]) if "explicit" in modes else float("nan")
        streams_ms = float(modes["streams"]["median_ms"]) if "streams" in modes else float("nan")
        speedup = capture_ms / explicit_ms if complete and explicit_ms > 0.0 else 0.0
        results.append(
            WorkloadResult(
                workload,
                capture_ms,
                explicit_ms,
                streams_ms,
                speedup,
                correct,
                complete,
            )
        )
    return results


def select_primary(
    results: Iterable[WorkloadResult], minimum_speedup: float = 1.10
) -> WorkloadResult | None:
    candidates = [
        result
        for result in results
        if result.complete and result.correct and result.speedup >= minimum_speedup
    ]
    return max(candidates, key=lambda result: result.speedup, default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze CKL Milestone 0 benchmark CSV files")
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=1.10)
    arguments = parser.parse_args()

    results = summarize(read_rows(arguments.csv))
    print("workload,capture_ms,explicit_ms,streams_ms,capture_over_explicit,correct,complete")
    for result in results:
        print(
            f"{result.workload},{result.capture_ms:.6f},{result.explicit_ms:.6f},"
            f"{result.streams_ms:.6f},{result.speedup:.4f},"
            f"{str(result.correct).lower()},{str(result.complete).lower()}"
        )

    selected = select_primary(results, arguments.minimum_speedup)
    if selected is None:
        print(
            f"No primary workload selected: none is correct, complete, and at least "
            f"{arguments.minimum_speedup:.2f}x faster than capture."
        )
        return 2
    print(f"Selected primary workload: {selected.workload} ({selected.speedup:.2f}x over capture)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

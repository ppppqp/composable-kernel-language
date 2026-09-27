#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


CSV_HEADER = "device,workload,mode,elements,cycles,warmup,repeats,median_ms,p95_ms,checksum,correct"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CKL Milestone 0 CUDA benchmarks")
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--elements", type=int, default=1 << 11)
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    arguments = parser.parse_args()

    command = [
        str(arguments.executable.resolve()),
        "--elements",
        str(arguments.elements),
        "--cycles",
        str(arguments.cycles),
        "--warmup",
        str(arguments.warmup),
        "--repeats",
        str(arguments.repeats),
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode:
        raise SystemExit(process.stderr or f"benchmark exited with status {process.returncode}")
    output = process.stdout.strip()
    if not output.startswith(CSV_HEADER):
        raise SystemExit("benchmark did not emit the expected CSV header")

    arguments.output.write_text(output + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


CSV_HEADER = (
    "device,batch,in_flight,nodes,temp_before_bytes,temp_after_bytes,setup_ms,"
    "median_ms,logical_iterations,checksum,correct,selected"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CKL Milestone 4 measured-plan search")
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--cubin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--elements", type=int, default=2048)
    parser.add_argument("--cycles", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    arguments = parser.parse_args()

    command = [
        str(arguments.executable.resolve()),
        str(arguments.cubin.resolve()),
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
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(output + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

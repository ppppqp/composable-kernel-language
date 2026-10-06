#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


HEADER = "device,architecture,mode,nodes,elements,metric,sample,value_us,correct"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CKL HIP runtime-overhead benchmark")
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--hsaco", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--elements", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--nodes", default="2,4,8,16")
    args = parser.parse_args()

    command = [
        str(args.executable.resolve()),
        str(args.hsaco.resolve()),
        "--device", str(args.device),
        "--elements", str(args.elements),
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--samples", str(args.samples),
        "--nodes", args.nodes,
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode:
        raise SystemExit(process.stderr or f"benchmark exited with status {process.returncode}")
    output = process.stdout.strip()
    if not output.startswith(HEADER):
        raise SystemExit("benchmark did not emit the expected CSV header")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from pathlib import Path
import sys

from ckl import extract_gpu_objects


def main() -> int:
    source = Path(sys.argv[1]).read_text()
    objects = extract_gpu_objects(source)
    if len(objects) != 1:
        raise RuntimeError(f"expected one GPU object, found {len(objects)}")
    Path(sys.argv[2]).write_bytes(objects[0].data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

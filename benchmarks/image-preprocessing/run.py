#!/usr/bin/env python3
"""Benchmark RGB normalization; excludes resize, crop, decode, and conversion."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application_common import main

if __name__ == "__main__":
    main("image-preprocessing", Path(__file__).with_name("workloads.py"))

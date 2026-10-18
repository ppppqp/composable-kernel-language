#!/usr/bin/env python3
"""Run with a CUDA PyTorch + NumPy environment and PYTHONPATH=python."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application_common import main

if __name__ == "__main__":
    main("momentum-sgd", Path(__file__).with_name("workloads.py"))

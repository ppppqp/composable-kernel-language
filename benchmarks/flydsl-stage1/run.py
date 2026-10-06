#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import flydsl.compiler as flyc
from kernels.norm.rmsnorm_bwd_kernel import build_rmsnorm_bwd_two_stage_module

from ckl import import_flydsl_artifact


@dataclass
class Mode:
    name: str
    launch: object


class Bridge:
    def __init__(self, library: Path):
        self.library = ctypes.CDLL(str(library.resolve()))
        self.library.ckl_flydsl_stage1_last_error.restype = ctypes.c_char_p
        self.library.ckl_flydsl_stage1_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_size_t,
            *([ctypes.c_uint64] * 7),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self.library.ckl_flydsl_stage1_create.restype = ctypes.c_void_p
        self.library.ckl_flydsl_stage1_launch_ordinary.argtypes = [ctypes.c_void_p]
        self.library.ckl_flydsl_stage1_launch_ordinary.restype = ctypes.c_int
        self.library.ckl_flydsl_stage1_launch_graph.argtypes = [ctypes.c_void_p]
        self.library.ckl_flydsl_stage1_launch_graph.restype = ctypes.c_int
        self.library.ckl_flydsl_stage1_destroy.argtypes = [ctypes.c_void_p]
        self.handle: int | None = None

    def error(self) -> str:
        message = self.library.ckl_flydsl_stage1_last_error()
        return message.decode() if message else "unknown CKL bridge error"

    def create(self, bundle: Path, tensors: tuple[torch.Tensor, ...], rows: int,
               columns: int, programs: int, stream: torch.cuda.Stream) -> None:
        pointers = [tensor.data_ptr() for tensor in tensors]
        self.handle = self.library.ckl_flydsl_stage1_create(
            str(bundle.resolve()).encode(),
            torch.cuda.current_device(),
            stream.cuda_stream,
            *pointers,
            rows,
            columns,
            programs,
        )
        if not self.handle:
            raise RuntimeError(self.error())

    def _launch(self, function) -> None:
        if function(self.handle):
            raise RuntimeError(self.error())

    def ordinary(self) -> None:
        self._launch(self.library.ckl_flydsl_stage1_launch_ordinary)

    def graph(self) -> None:
        self._launch(self.library.ckl_flydsl_stage1_launch_graph)

    def close(self) -> None:
        if self.handle:
            torch.cuda.synchronize()
            self.library.ckl_flydsl_stage1_destroy(self.handle)
            self.handle = None


def repository_revision(source: Path) -> str:
    for directory in (source, *source.parents):
        if (directory / ".git").exists():
            return subprocess.check_output(
                ("git", "-C", str(directory), "describe", "--always", "--dirty"),
                text=True,
            ).strip()
    return "unknown"


def gpu_time_us(launch, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def cpu_time_us(launch, iterations: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        launch()
    elapsed = time.perf_counter_ns() - start
    torch.cuda.synchronize()
    return elapsed / 1000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare native FlyDSL and CKL execution")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=11)
    args = parser.parse_args()

    metadata = json.loads((args.bundle / "metadata.json").read_text(encoding="utf-8"))
    columns = int(metadata["columns"])
    programs = int(metadata["programs"])
    rows = args.rows
    if rows != int(metadata["rows"]):
        raise ValueError(
            f"bundle is specialized for {metadata['rows']} rows, but --rows requested {rows}; "
            "rerun prepare.py for the requested shape"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    # Explicitly use one non-default stream for all four modes. HIP does not permit beginning
    # thread-local capture on the legacy default stream.
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    torch.manual_seed(42)

    source = torch.randn((rows, columns), device=device, dtype=torch.float32)
    gamma = torch.rand((columns,), device=device, dtype=torch.float32)
    dy = torch.randn_like(source)
    rstd = torch.rsqrt(source.square().mean(1) + 1e-5)
    dx = torch.empty_like(source)
    dweight = torch.empty_like(gamma)
    partial = torch.empty((programs * columns,), device=device, dtype=torch.float32)
    tensors = (source, gamma, dy, rstd, dx, dweight, partial)

    launcher = build_rmsnorm_bwd_two_stage_module(columns, "f32", programs)
    compile_start = time.perf_counter_ns()
    compiled = flyc.compile(
        launcher, source, gamma, dy, rstd, dx, dweight, partial, rows, stream
    )
    torch.cuda.synchronize()
    compile_us = (time.perf_counter_ns() - compile_start) / 1000.0
    artifact = import_flydsl_artifact(compiled.artifact.export_for_orchestration())
    if artifact.identity != metadata["identity"]:
        raise RuntimeError(
            f"prepared artifact {metadata['identity']} does not match runtime artifact "
            f"{artifact.identity}"
        )

    def native_launch() -> None:
        compiled(source, gamma, dy, rstd, dx, dweight, partial, rows, stream)

    native_launch()
    torch.cuda.synchronize()
    reference = (dx.clone(), dweight.clone(), partial.clone())

    bridge = Bridge(args.bridge)
    create_start = time.perf_counter_ns()
    bridge.create(args.bundle, tensors, rows, columns, programs, stream)
    create_us = (time.perf_counter_ns() - create_start) / 1000.0

    graph_capture_start = time.perf_counter_ns()
    torch_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_graph, stream=stream):
        native_launch()
    torch.cuda.synchronize()
    graph_capture_us = (time.perf_counter_ns() - graph_capture_start) / 1000.0

    modes = [
        Mode("flydsl_eager", native_launch),
        Mode("torch_graph", torch_graph.replay),
        Mode("ckl_ordinary", bridge.ordinary),
        Mode("ckl_graph", bridge.graph),
    ]

    for mode in modes:
        dx.zero_()
        dweight.zero_()
        partial.zero_()
        mode.launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            dx, reference[0], rtol=1e-5, atol=1e-5, msg=lambda message: f"{mode.name} dx: {message}"
        )
        torch.testing.assert_close(
            dweight,
            reference[1],
            rtol=1e-5,
            atol=1e-5,
            msg=lambda message: f"{mode.name} dweight: {message}",
        )
        torch.testing.assert_close(
            partial,
            reference[2],
            rtol=1e-5,
            atol=1e-5,
            msg=lambda message: f"{mode.name} partial: {message}",
        )
        for _ in range(args.warmup):
            mode.launch()
        torch.cuda.synchronize()

    rows_out: list[dict[str, object]] = []
    launch_geometry = json.dumps(
        [
            {
                "grid": list(launch.grid),
                "block": list(launch.block),
                "shared_memory": launch.shared_memory,
            }
            for launch in artifact.launch_plan.launches
        ],
        separators=(",", ":"),
    )
    ckl_revision = repository_revision(Path(__file__).resolve())
    flydsl_revision = repository_revision(Path(flyc.__file__).resolve())

    def append(mode: str, metric: str, sample: int, value_us: float) -> None:
        rows_out.append(
            {
                "device": torch.cuda.get_device_name(device),
                "architecture": torch.cuda.get_device_properties(device).gcnArchName,
                "artifact": artifact.identity,
                "ckl_revision": ckl_revision,
                "flydsl_revision": flydsl_revision,
                "torch_version": torch.__version__,
                "hip_version": torch.version.hip,
                "mode": mode,
                "rows": rows,
                "columns": columns,
                "programs": programs,
                "dtype": "f32",
                "launch_count": len(artifact.launch_plan.launches),
                "launch_geometry": launch_geometry,
                "workspace_bytes": partial.numel() * partial.element_size(),
                "warmup": args.warmup,
                "iterations": args.iterations,
                "samples": args.samples,
                "arguments_rotated": "false",
                "metric": metric,
                "sample": sample,
                "value_us": f"{value_us:.9f}",
                "correct": "true",
            }
        )

    append("flydsl_eager", "compile_us", 0, compile_us)
    append("ckl_graph", "create_resolve_instantiate_us", 0, create_us)
    append("torch_graph", "capture_us", 0, graph_capture_us)
    for sample in range(args.samples):
        for offset in range(len(modes)):
            mode = modes[(sample + offset) % len(modes)]
            append(mode.name, "gpu_replay_us", sample, gpu_time_us(mode.launch, args.iterations))
            append(mode.name, "cpu_submit_us", sample, cpu_time_us(mode.launch, args.iterations))

    bridge.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"wrote {args.output}")
    for metric in ("gpu_replay_us", "cpu_submit_us"):
        medians = {
            mode.name: statistics.median(
                float(row["value_us"])
                for row in rows_out
                if row["metric"] == metric and row["mode"] == mode.name
            )
            for mode in modes
        }
        print(metric + " " + " ".join(f"{name}={value:.3f}" for name, value in medians.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

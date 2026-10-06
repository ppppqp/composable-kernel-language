#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

import flydsl.compiler as flyc
from kernels.norm.rmsnorm_bwd_kernel import build_rmsnorm_bwd_two_stage_module

from ckl import emit_flydsl_cpp_bundle, import_flydsl_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile the Stage 1 FlyDSL RMSNorm artifact")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ckl-import-manifest", required=True, type=Path)
    parser.add_argument("--ckl-hostgen", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--columns", type=int, default=256)
    parser.add_argument("--programs", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("FlyDSL Stage 1 requires an accessible ROCm device")
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(device)
    rows, columns, programs = args.rows, args.columns, args.programs

    source = torch.randn((rows, columns), device=device, dtype=torch.float32)
    weight = torch.rand((columns,), device=device, dtype=torch.float32)
    output_gradient = torch.randn_like(source)
    reciprocal_std = torch.rsqrt(source.square().mean(1) + 1e-5)
    input_gradient = torch.empty_like(source)
    weight_gradient = torch.empty_like(weight)
    partial = torch.empty((programs * columns,), device=device, dtype=torch.float32)

    launcher = build_rmsnorm_bwd_two_stage_module(columns, "f32", programs)
    compiled = flyc.compile(
        launcher,
        source,
        weight,
        output_gradient,
        reciprocal_std,
        input_gradient,
        weight_gradient,
        partial,
        rows,
        stream,
    )
    torch.cuda.synchronize(device)

    artifact = import_flydsl_artifact(compiled.artifact.export_for_orchestration())
    print(
        "device_objects="
        + repr([(gpu_object.target, len(gpu_object.data)) for gpu_object in artifact.gpu_objects]),
        flush=True,
    )
    bundle = emit_flydsl_cpp_bundle(
        artifact,
        args.output,
        resource_layouts={
            "Input": struct.pack("<iiq", rows, columns, columns),
            "Gamma": struct.pack("<i", columns),
            "DY": struct.pack("<iiq", rows, columns, columns),
            "Rstd": struct.pack("<i", rows),
            "DX": struct.pack("<iiq", rows, columns, columns),
            "DWeight": struct.pack("<i", columns),
            "DWeightPartial": struct.pack("<i", programs * columns),
        },
        manifest_importer=args.ckl_import_manifest,
        host_generator=args.ckl_hostgen,
    )
    metadata = {
        "identity": artifact.identity,
        "target": artifact.target,
        "host_entry": artifact.host_entry,
        "plan_builder": bundle.plan_builder,
        "artifact_loader": bundle.artifact_loader,
        "launches": len(artifact.launch_plan.launches) if artifact.launch_plan else 0,
        "rows": rows,
        "columns": columns,
        "programs": programs,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

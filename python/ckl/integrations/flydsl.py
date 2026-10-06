from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ..compiler import GPUObject


class FlyDSLDeviceObject(Protocol):
    data: bytes
    format: int
    target: str


class FlyDSLOrchestrationArtifact(Protocol):
    compiled_ir: str
    source_ir: str | None
    host_entry: str
    backend: str
    target: str
    kernel_abi: str
    device_objects: Sequence[FlyDSLDeviceObject]
    launch_plan: object | None
    launch_plan_error: str | None


@dataclass(frozen=True)
class FlyDSLArgument:
    logical_index: int
    kind: str
    source_type: str
    binding: str | None
    value: int | float | None


@dataclass(frozen=True)
class FlyDSLLaunch:
    id: int
    kernel: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_memory: int
    arguments: tuple[FlyDSLArgument, ...]
    dependencies: tuple[int, ...]


@dataclass(frozen=True)
class FlyDSLLaunchPlan:
    host_entry: str
    launches: tuple[FlyDSLLaunch, ...]


@dataclass(frozen=True)
class FlyDSLArtifact:
    """FlyDSL compiler output normalized for CKL without adopting its runtime."""

    identity: str
    source_ir: str
    compiled_ir: str
    host_entry: str
    backend: str
    target: str
    kernel_abi: str
    gpu_objects: tuple[GPUObject, ...]
    launch_plan: FlyDSLLaunchPlan | None
    launch_plan_error: str | None


@dataclass(frozen=True)
class FlyDSLCppBundle:
    """Files needed to compile and load one FlyDSL plan through CKL."""

    directory: Path
    manifest: Path
    device_object: Path
    host_source: Path
    plan_builder: str
    artifact_loader: str


class FlyDSLBridgeError(RuntimeError):
    pass


def _import_launch_plan(plan: object | None) -> FlyDSLLaunchPlan | None:
    if plan is None:
        return None
    launches = tuple(
        FlyDSLLaunch(
            id=int(launch.id),
            kernel=str(launch.kernel),
            grid=tuple(int(value) for value in launch.grid),
            block=tuple(int(value) for value in launch.block),
            shared_memory=int(launch.shared_memory),
            arguments=tuple(
                FlyDSLArgument(
                    logical_index=int(argument.logical_index),
                    kind=str(argument.kind),
                    source_type=str(argument.source_type),
                    binding=None if argument.binding is None else str(argument.binding),
                    value=argument.value,
                )
                for argument in launch.arguments
            ),
            dependencies=tuple(int(value) for value in launch.dependencies),
        )
        for launch in plan.launches
    )
    return FlyDSLLaunchPlan(host_entry=str(plan.host_entry), launches=launches)


def import_flydsl_artifact(exported: FlyDSLOrchestrationArtifact) -> FlyDSLArtifact:
    """Import FlyDSL's public orchestration export.

    This imports compiler artifacts and a producer-verified launch plan when present. It does not
    treat FlyDSL's host-wrapper ABI as a device ABI or reconstruct a plan from source text.
    """

    required = {
        "compiled_ir": exported.compiled_ir,
        "source_ir": exported.source_ir,
        "host_entry": exported.host_entry,
        "backend": exported.backend,
        "target": exported.target,
        "kernel_abi": exported.kernel_abi,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"FlyDSL orchestration export is missing: {', '.join(missing)}")

    objects = tuple(
        GPUObject(data=bytes(obj.data), format=int(obj.format), target=str(obj.target))
        for obj in exported.device_objects
    )
    digest = hashlib.sha256()
    digest.update(exported.backend.encode("utf-8"))
    digest.update(b"\0")
    digest.update(exported.target.encode("utf-8"))
    digest.update(b"\0")
    digest.update(exported.kernel_abi.encode("utf-8"))
    for gpu_object in objects:
        digest.update(gpu_object.data)
    if not objects:
        digest.update(exported.compiled_ir.encode("utf-8"))
    identity = f"flydsl.{exported.backend}.{digest.hexdigest()[:24]}"

    return FlyDSLArtifact(
        identity=identity,
        source_ir=exported.source_ir,
        compiled_ir=exported.compiled_ir,
        host_entry=exported.host_entry,
        backend=exported.backend,
        target=exported.target,
        kernel_abi=exported.kernel_abi,
        gpu_objects=objects,
        launch_plan=_import_launch_plan(getattr(exported, "launch_plan", None)),
        launch_plan_error=getattr(exported, "launch_plan_error", None),
    )


def flydsl_executable_manifest(
    artifact: FlyDSLArtifact,
    *,
    device: int = 0,
    resource_layouts: Mapping[str, bytes] | None = None,
) -> dict:
    """Convert a verified FlyDSL launch plan to CKL executable-manifest v1.

    FlyDSL memrefs are logical launch operands. For its ROCm bare-pointer ABI,
    each one becomes a pointer followed by a packed, by-value layout descriptor.
    The descriptor bytes are specialization data supplied by the caller that
    owns the concrete tensor shapes and strides.
    """
    if artifact.launch_plan is None:
        reason = artifact.launch_plan_error or "the producer did not export a launch plan"
        raise ValueError(f"FlyDSL artifact is not executable through CKL: {reason}")
    if artifact.kernel_abi != "rocm.bare_ptr":
        raise ValueError(f"unsupported FlyDSL kernel ABI: {artifact.kernel_abi}")

    resource_names = {
        argument.binding
        for launch in artifact.launch_plan.launches
        for argument in launch.arguments
        if argument.kind in {"resource", "memref"}
    }
    if None in resource_names:
        raise ValueError("FlyDSL resource argument has no host binding")
    device_name = f"{artifact.backend}:{device}"

    layouts = {} if resource_layouts is None else dict(resource_layouts)
    kernels = []
    for launch in artifact.launch_plan.launches:
        arguments = []
        slot = 0
        for argument in launch.arguments:
            descriptor = {
                "slot": slot,
                "logical_index": argument.logical_index,
                "kind": argument.kind,
                "type": "memref<?xi8>" if argument.kind == "resource" else argument.source_type,
            }
            if argument.kind == "resource":
                descriptor.update(resource=argument.binding, packing="bare_pointer")
                arguments.append(descriptor)
                slot += 1
                continue
            elif argument.kind == "memref":
                if argument.binding is None:
                    raise ValueError("FlyDSL memref argument has no host binding")
                try:
                    layout = bytes(layouts[argument.binding])
                except KeyError as error:
                    raise ValueError(
                        f"FlyDSL memref {argument.binding!r} requires packed layout descriptor bytes"
                    ) from error
                if not layout:
                    raise ValueError(
                        f"FlyDSL memref {argument.binding!r} has an empty layout descriptor"
                    )
                descriptor.update(
                    kind="resource",
                    type="memref<?xi8>",
                    resource=argument.binding,
                    packing="bare_pointer",
                )
                arguments.append(descriptor)
                slot += 1
                arguments.append(
                    {
                        "slot": slot,
                        "logical_index": argument.logical_index,
                        "kind": "bytes",
                        "type": f"vector<{len(layout)}xi8>",
                        "value": list(layout),
                    }
                )
                slot += 1
                continue
            elif argument.kind == "scalar":
                descriptor["name"] = argument.binding
            elif argument.kind == "constant":
                descriptor["value"] = argument.value
            else:
                raise ValueError(f"unsupported FlyDSL launch argument kind: {argument.kind}")
            arguments.append(descriptor)
            slot += 1
        kernels.append(
            {
                "id": launch.id,
                "kernel": launch.kernel,
                "implementation": f"{artifact.identity}.launch.{launch.id}",
                "artifact": artifact.identity,
                "abi": artifact.kernel_abi,
                "device": device_name,
                "grid": list(launch.grid),
                "block": list(launch.block),
                "shared_memory": launch.shared_memory,
                "arguments": arguments,
                "dependencies": list(launch.dependencies),
            }
        )

    return {
        "schema_version": 1,
        "plan": {
            "source": f"@{artifact.launch_plan.host_entry}",
            "name": f"flydsl.{artifact.launch_plan.host_entry}",
            "backend": artifact.backend,
            "heaps": [],
            "resources": [
                {
                    "name": name,
                    "kind": "external",
                    "device": device_name,
                    "address_space": "global",
                    "alignment": 1,
                }
                for name in sorted(resource_names)
            ],
            "kernels": kernels,
        },
    }


def _resolve_tool(explicit: Path | None, environment: str, relative: str) -> Path:
    candidate = explicit
    if candidate is None and (configured := os.environ.get(environment)):
        candidate = Path(configured)
    if candidate is None:
        candidate = Path(__file__).resolve().parents[3] / "build" / "tools" / relative / relative
    candidate = Path(candidate)
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{relative} was not found at {candidate}; set {environment} or pass its path"
        )
    return candidate.resolve()


def _run_bridge_tool(command: Path, source: str) -> str:
    process = subprocess.run(
        (str(command), "-"), input=source, text=True, capture_output=True, check=False
    )
    if process.returncode:
        raise FlyDSLBridgeError(f"{command.name} failed:\n{process.stderr}")
    return process.stdout


def _cpp_identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9]", "_", value)
    return f"_{result}" if not result or result[0].isdigit() else result


def emit_flydsl_cpp_bundle(
    artifact: FlyDSLArtifact,
    output_directory: Path | str,
    *,
    device: int = 0,
    resource_layouts: Mapping[str, bytes] | None = None,
    manifest_importer: Path | None = None,
    host_generator: Path | None = None,
) -> FlyDSLCppBundle:
    """Emit a HSACO and C++ builders from a verified FlyDSL artifact.

    The generated source contains both the ExecutionPlan builder and a small ArtifactRegistry
    loader. Keeping the HSACO as a neighboring file avoids embedding a potentially large binary
    in generated C++.
    """

    manifest = flydsl_executable_manifest(
        artifact, device=device, resource_layouts=resource_layouts
    )
    if artifact.backend != "rocm":
        raise ValueError(f"unsupported FlyDSL bundle backend: {artifact.backend}")
    unique_objects = []
    seen_payloads = set()
    for gpu_object in artifact.gpu_objects:
        digest = hashlib.sha256(gpu_object.data).digest()
        if digest not in seen_payloads:
            seen_payloads.add(digest)
            unique_objects.append(gpu_object)
    if len(unique_objects) != 1:
        raise ValueError(
            "FlyDSL C++ bundle emission currently requires exactly one unique GPU object, "
            f"got {len(unique_objects)} unique payloads from {len(artifact.gpu_objects)} objects"
        )
    device_object = unique_objects[0]
    if not device_object.data.startswith(b"\x7fELF"):
        raise ValueError("FlyDSL ROCm GPU object is not an ELF HSACO")

    importer = _resolve_tool(
        manifest_importer, "CKL_IMPORT_MANIFEST", "ckl-import-manifest"
    )
    hostgen = _resolve_tool(host_generator, "CKL_HOSTGEN", "ckl-hostgen")
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    executable_mlir = _run_bridge_tool(importer, manifest_text)
    generated_host = _run_bridge_tool(hostgen, executable_mlir)

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "plan.json"
    object_path = directory / "device.hsaco"
    host_path = directory / "host.cpp.inc"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    object_path.write_bytes(device_object.data)

    name = _cpp_identifier(manifest["plan"]["name"])
    plan_builder = f"build_{name}"
    artifact_loader = f"load_{name}_artifacts"
    registry_loader = f'''\n#include <filesystem>

namespace ckl_generated {{

inline mlir::ckl::runtime::ArtifactRegistry {artifact_loader}(
    const std::filesystem::path &bundle_directory) {{
  mlir::ckl::runtime::ArtifactRegistry result;
  result.add(mlir::ckl::runtime::KernelArtifact::readFile(
      {json.dumps(artifact.identity)}, "rocm.hsaco", {json.dumps(artifact.target)},
      bundle_directory / "device.hsaco"));
  return result;
}}

}} // namespace ckl_generated
'''
    host_path.write_text(generated_host + registry_loader, encoding="utf-8")
    return FlyDSLCppBundle(
        directory, manifest_path, object_path, host_path, plan_builder, artifact_loader
    )

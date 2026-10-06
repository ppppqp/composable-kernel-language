from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, Sequence

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


def flydsl_executable_manifest(artifact: FlyDSLArtifact, *, device: int = 0) -> dict:
    """Convert a verified FlyDSL launch plan to CKL executable-manifest v1."""
    if artifact.launch_plan is None:
        reason = artifact.launch_plan_error or "the producer did not export a launch plan"
        raise ValueError(f"FlyDSL artifact is not executable through CKL: {reason}")
    if artifact.kernel_abi != "rocm.bare_ptr":
        raise ValueError(f"unsupported FlyDSL kernel ABI: {artifact.kernel_abi}")

    resource_names = {
        argument.binding
        for launch in artifact.launch_plan.launches
        for argument in launch.arguments
        if argument.kind == "resource"
    }
    if None in resource_names:
        raise ValueError("FlyDSL resource argument has no host binding")
    device_name = f"{artifact.backend}:{device}"

    kernels = []
    for launch in artifact.launch_plan.launches:
        arguments = []
        for slot, argument in enumerate(launch.arguments):
            descriptor = {
                "slot": slot,
                "logical_index": argument.logical_index,
                "kind": argument.kind,
                "type": "memref<?xi8>" if argument.kind == "resource" else argument.source_type,
            }
            if argument.kind == "resource":
                descriptor.update(resource=argument.binding, packing="bare_pointer")
            elif argument.kind == "scalar":
                descriptor["name"] = argument.binding
            elif argument.kind == "constant":
                descriptor["value"] = argument.value
            else:
                raise ValueError(f"unsupported FlyDSL launch argument kind: {argument.kind}")
            arguments.append(descriptor)
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
            "name": artifact.identity,
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

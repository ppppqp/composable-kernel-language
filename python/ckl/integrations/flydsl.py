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


def import_flydsl_artifact(exported: FlyDSLOrchestrationArtifact) -> FlyDSLArtifact:
    """Import FlyDSL's public orchestration export.

    This imports compiler artifacts only. It does not treat FlyDSL's host-wrapper ABI as a
    device ABI and does not infer an execution plan from the source program.
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
    )

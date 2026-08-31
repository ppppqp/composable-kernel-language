from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class CompilationError(RuntimeError):
    def __init__(self, command: Sequence[str], stderr: str):
        self.command = tuple(command)
        self.stderr = stderr
        super().__init__(f"CKL compilation failed: {' '.join(command)}\n{stderr}")


@dataclass(frozen=True)
class NVIDIATarget:
    chip: str
    features: str
    toolkit_root: Path | None = None
    binary_format: str = "bin"

    def pipeline(self) -> str:
        return (
            "--gpu-lower-to-nvvm-pipeline="
            f"cubin-chip={self.chip} cubin-features={self.features} "
            f"cubin-format={self.binary_format}"
        )

    def environment(self) -> dict[str, str]:
        if self.toolkit_root is None:
            return {}
        root = str(Path(self.toolkit_root).resolve())
        return {
            "CUDA_HOME": root,
            "CUDA_PATH": root,
            "CUDAToolkit_ROOT": root,
            "PATH": f"{root}/bin{os.pathsep}{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": (
                f"{root}/lib64{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}"
            ),
        }

    def cache_identity(self) -> str:
        toolkit = (
            str(Path(self.toolkit_root).resolve())
            if self.toolkit_root is not None
            else "environment"
        )
        return f"nvidia:{self.chip}:{self.features}:{self.binary_format}:{toolkit}"


@dataclass(frozen=True)
class CompilerOptions:
    ckl_opt: Path | None = None
    passes: tuple[str, ...] = ()
    target: NVIDIATarget | None = None

    def resolve_ckl_opt(self) -> Path:
        if self.ckl_opt is not None:
            candidate = Path(self.ckl_opt)
        elif value := os.environ.get("CKL_OPT"):
            candidate = Path(value)
        else:
            root = Path(__file__).resolve().parents[2]
            candidate = root / "build" / "tools" / "ckl-opt" / "ckl-opt"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"ckl-opt was not found at {candidate}; set CKL_OPT or pass CompilerOptions"
            )
        return candidate.resolve()

    def command(self) -> tuple[str, ...]:
        target_pipeline = (self.target.pipeline(),) if self.target is not None else ()
        return (str(self.resolve_ckl_opt()), *self.passes, *target_pipeline)

    def environment(self) -> Mapping[str, str]:
        environment = os.environ.copy()
        if self.target is not None:
            environment.update(self.target.environment())
        return environment

    def cache_command(self) -> tuple[str, ...]:
        identity = self.target.cache_identity() if self.target is not None else "no-target"
        return (*self.command(), identity)


@dataclass(frozen=True)
class GPUObject:
    data: bytes
    format: int
    target: str


@dataclass(frozen=True)
class CompiledModule:
    source: str
    mlir: str
    command: tuple[str, ...]
    cache_key: str
    gpu_objects: tuple[GPUObject, ...] = ()


def compilation_key(source: str, command: Sequence[str]) -> str:
    return hashlib.sha256(
        source.encode("utf-8") + b"\0" + "\0".join(command).encode("utf-8")
    ).hexdigest()


def _extract_gpu_objects(source: str) -> tuple[GPUObject, ...]:
    if "gpu.binary" not in source:
        return ()
    from mlir import ir
    from mlir.dialects import gpu, nvvm  # noqa: F401 - registers dialect attributes

    context = ir.Context()
    context.allow_unregistered_dialects = True
    with context:
        module = ir.Module.parse(source)
        objects: list[GPUObject] = []

        def visit(operation: ir.Operation) -> None:
            if operation.name == "gpu.binary":
                for attribute in ir.ArrayAttr(operation.attributes["objects"]):
                    object_attribute = gpu.ObjectAttr(attribute)
                    objects.append(
                        GPUObject(
                            data=object_attribute.object,
                            format=object_attribute.format,
                            target=str(object_attribute.target),
                        )
                    )
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        visit(child)

        visit(module.operation)
        return tuple(objects)


def compile_module(source: str, options: CompilerOptions) -> CompiledModule:
    command = options.command()
    cache_key = compilation_key(source, options.cache_command())
    process = subprocess.run(
        command,
        input=source,
        text=True,
        capture_output=True,
        check=False,
        env=options.environment(),
    )
    if process.returncode:
        raise CompilationError(command, process.stderr)
    return CompiledModule(
        source, process.stdout, command, cache_key, _extract_gpu_objects(process.stdout)
    )

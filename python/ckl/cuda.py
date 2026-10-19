"""In-process CUDA compilation, host JIT, and execution."""

from __future__ import annotations

import importlib.util
import math
import os
import re
import struct
import sys
from contextlib import AbstractContextManager
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

from .compiler import NVIDIATarget
from .types import BufferSpec, f32, f64, i32, i64

_library = None
_SCALARS = {f32: ("f", 4), f64: ("d", 8), i32: ("i", 4), i64: ("q", 8)}


def _byte_count(spec):
    if not isinstance(spec, BufferSpec) or spec.dtype not in _SCALARS:
        raise TypeError("device storage requires a numeric BufferSpec")
    size = math.prod(spec.shape) * _SCALARS[spec.dtype][1]
    if not 0 < size < (1 << 64):
        raise ValueError("device storage byte count must fit a positive 64-bit size")
    return size


def _native():
    global _library
    if _library is not None:
        return _library
    configured = os.environ.get("CKL_NATIVE_LIBRARY")
    root = Path(__file__).resolve().parents[2]
    candidates = (
        [Path(configured)]
        if configured
        else [
            directory / ("_ckl_native" + suffix)
            for directory in (Path(__file__).parent, root / "build/python/ckl")
            for suffix in EXTENSION_SUFFIXES
        ]
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise RuntimeError(
            "CKL nanobind extension is not built; enable CKL_ENABLE_MLIR "
            "or set CKL_NATIVE_LIBRARY to the built _ckl_native extension"
        )
    name = "ckl._ckl_native"
    spec = importlib.util.spec_from_file_location(name, path.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _library = module
    return module


class _ContextManager(AbstractContextManager):
    """Explicit scopes for resources whose storage is owned by native C++ objects."""

    _resource = None

    @property
    def _native_resource(self):
        if self._resource is None:
            raise RuntimeError(f"{type(self).__name__} is closed")
        return self._resource

    @property
    def closed(self):
        return self._resource is None

    def close(self):
        # Bound invocations retain their native dependencies independently.
        self._resource = None

    def __enter__(self):
        _ = self._native_resource
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class Device(_ContextManager):
    """Own CUDA device 0's runtime. Construction initializes CUDA; compilation does not."""

    def __init__(self):
        self._module = _native()
        if not self._module.has_cuda_runtime:
            raise RuntimeError("rebuild CKL with CKL_ENABLE_CUDA_RUNTIME=ON to execute programs")
        self._resource = self._module.Device()

    @property
    def native_library(self):
        return Path(self._module.__file__).resolve()

    @property
    def name(self):
        return self._native_resource.name

    def empty(self, spec: BufferSpec) -> DeviceBuffer:
        size = _byte_count(spec)
        resource = self._native_resource.empty(size)
        return DeviceBuffer(self, resource, spec)

    def import_buffer(self, address: int, spec: BufferSpec, *, owner) -> DeviceBuffer:
        """Borrow device storage, retaining its Python owner for the buffer's lifetime."""
        size = _byte_count(spec)
        if type(address) is not int or not 0 < address < (1 << 64) or owner is None:
            raise ValueError("imported storage requires a valid address and a live owner")
        native_owner = owner._native_resource if isinstance(owner, DeviceBuffer) else owner
        resource = self._native_resource.import_buffer(address, size, native_owner)
        return DeviceBuffer(self, resource, spec, owner)


class DeviceBuffer(_ContextManager):
    def __init__(self, device, resource, spec, owner=None):
        self.device, self.spec, self._owner = device, spec, owner
        self._module, self._resource = device._module, resource

    def close(self):
        super().close()
        self.device = self._owner = None

    @property
    def address(self):
        return self._native_resource.address

    @property
    def nbytes(self):
        return self._native_resource.nbytes

    def copy_from_host(self, source):
        view = memoryview(source).cast("B")
        if view.nbytes != self.nbytes:
            raise ValueError("host buffer byte count does not match device storage")
        self._native_resource.copy_from_host(view)

    def copy_to_host(self, destination):
        view = memoryview(destination).cast("B")
        if view.readonly or view.nbytes != self.nbytes:
            raise ValueError("destination must be writable and match device storage byte count")
        self._native_resource.copy_to_host(view)


class CompiledProgram(_ContextManager):
    """Own specialized device code and its JIT-compiled host entry point."""

    def __init__(self, program, target: NVIDIATarget, *, fuse_pointwise=True, reuse_storage=True):
        if type(fuse_pointwise) is not bool:
            raise TypeError("fuse_pointwise must be bool")
        if type(reuse_storage) is not bool:
            raise TypeError("reuse_storage must be bool")
        if target.binary_format != "bin":
            raise ValueError("CKL CUDA compilation requires a binary target")
        if not re.fullmatch(r"sm_[0-9]+[a-z]?", target.chip) or not re.fullmatch(
            r"[+\-a-zA-Z0-9_,]+", target.features
        ):
            raise ValueError("invalid CUDA chip or features")
        self.signature, self.target = program.signature, target
        self._module = _native()
        toolkit = str(Path(target.toolkit_root).resolve()) if target.toolkit_root else ""
        self._resource = self._module.compile(
            program.mlir, target.chip, target.features, toolkit, fuse_pointwise, reuse_storage
        )

    @property
    def prepared_ir(self):
        return self._native_resource.ir(0)

    @property
    def executable_ir(self):
        return self._native_resource.ir(1)

    @property
    def device_ir(self):
        return self._native_resource.ir(2)

    @property
    def host_ir(self):
        return self._native_resource.ir(3)

    def bind(self, *arguments, device: Device | None = None) -> Invocation:
        parameters = self.signature.runtime
        if len(arguments) != len(parameters):
            raise TypeError(f"expected {len(parameters)} runtime arguments, got {len(arguments)}")
        buffers = [a for a in arguments if isinstance(a, DeviceBuffer)]
        device = device or (buffers[0].device if buffers else None)
        if device is None:
            raise ValueError("pass a Device for programs without buffer arguments")
        if device._module is not self._module:
            raise ValueError("program and device must use the same native library")
        packed = []
        for index, (parameter, value) in enumerate(zip(parameters, arguments)):
            if isinstance(parameter.dtype, BufferSpec):
                if not isinstance(value, DeviceBuffer) or value.spec != parameter.dtype:
                    raise TypeError(f"{parameter.name} requires {parameter.dtype}")
                if value.device is not device:
                    raise ValueError("all buffers must belong to the selected Device")
                packed.append(value._native_resource)
            else:
                scalar, _ = _SCALARS[parameter.dtype]
                if parameter.dtype in (i32, i64):
                    width = _SCALARS[parameter.dtype][1] * 8
                    if type(value) is not int or not -(1 << (width - 1)) <= value < (
                        1 << (width - 1)
                    ):
                        raise TypeError(f"{parameter.name} requires a {width}-bit integer")
                elif type(value) not in (int, float) or not math.isfinite(float(value)):
                    raise TypeError(f"{parameter.name} requires a finite numeric value")
                if parameter.dtype in (f32, f64):
                    try:
                        rounded = struct.unpack(scalar, struct.pack(scalar, value))[0]
                    except (OverflowError, struct.error):
                        raise ValueError(
                            f"{parameter.name} overflows {parameter.dtype.name}"
                        ) from None
                    if not math.isfinite(rounded):
                        raise ValueError(f"{parameter.name} overflows {parameter.dtype.name}")
                packed.append(value)
        resource = self._module.bind(self._native_resource, device._native_resource, packed)
        return Invocation(self, device, arguments, resource)

    def run(self, *arguments, graph=False, device=None) -> Invocation:
        return self.bind(*arguments, device=device).run(graph=graph)


class Invocation(_ContextManager):
    """A bound program, owning temporaries and a lazily instantiated CUDA graph."""

    def __init__(self, program, device, arguments, resource):
        self.program, self.device, self.arguments = program, device, arguments
        self._module, self._resource = program._module, resource

    def close(self):
        super().close()
        self.program = self.device = self.arguments = None

    @staticmethod
    def _execution(graph, mode, batch_size):
        if type(batch_size) is not int or not 0 < batch_size < (1 << 64):
            raise ValueError("batch_size must be a positive 64-bit count")
        if mode is not None and graph:
            raise ValueError("specify either graph or mode")
        choices = {"ordinary": 0, "graph": 1, "capture": 2}
        if mode is not None and mode not in choices:
            raise ValueError("mode must be ordinary, graph, or capture")
        return choices[mode] if mode is not None else int(bool(graph))

    def run(self, *, graph=False, mode=None, batch_size=1):
        # Synchronous execution keeps Python owners and temporary heaps alive
        # through completion. Reusing an invocation reuses its graph and heaps.
        execution = self._execution(graph, mode, batch_size)
        self._native_resource.run(execution, batch_size)
        return self

    @property
    def node_count(self):
        """Actual bound topology, including any overlap-triggered fallback."""
        return self._native_resource.node_count

    def benchmark(self, *, iterations=200, graph=False, mode=None, batch_size=1):
        """Measure iterations groups of batch_size programs; report time per program.

        mode='capture' uses actual single-stream CUDA capture. mode='graph'
        uses inferred dependencies. Batched graphs serialize complete iterations,
        sharing the bound invocation's storage. All execution is synchronous.
        """
        if type(iterations) is not int or not 0 < iterations < (1 << 64):
            raise ValueError("iterations must be a positive 64-bit count")
        execution = self._execution(graph, mode, batch_size)
        result = self._native_resource.benchmark(execution, iterations, batch_size)
        return {"stream_us": result[0], "wall_us": result[1], "graph_setup_ms": result[2]}

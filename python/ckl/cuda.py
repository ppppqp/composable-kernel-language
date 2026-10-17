"""In-process CUDA compilation, host JIT, and execution."""
from __future__ import annotations

import ctypes as ct
import math
import os
from pathlib import Path
import re

from .compiler import NVIDIATarget
from .types import BufferSpec, f32, f64, i32, i64

_library = None
_SCALARS = {f32: (ct.c_float, 4), f64: (ct.c_double, 8),
            i32: (ct.c_int32, 4), i64: (ct.c_int64, 8)}


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
    candidates = [Path(configured)] if configured else [
        Path(__file__).with_name("libCKLCompiler.so"),
        root / "build/python/ckl/libCKLCompiler.so",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise RuntimeError("CKL native compiler is not built; enable CKL_ENABLE_MLIR "
                           "or set CKL_NATIVE_LIBRARY to libCKLCompiler.so")
    lib = ct.CDLL(str(path.resolve()))
    pointer, size = ct.c_void_p, ct.c_uint64

    def declare(name, result, *arguments):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, arguments

    declare("ckl_error", ct.c_char_p)
    declare("ckl_compile", pointer, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_int)
    declare("ckl_program_free", None, pointer)
    declare("ckl_program_ir", ct.c_char_p, pointer, ct.c_uint)
    declare("ckl_has_cuda_runtime", ct.c_int)
    if lib.ckl_has_cuda_runtime():
        declare("ckl_device_create", pointer)
        declare("ckl_device_free", None, pointer)
        declare("ckl_device_name", ct.c_char_p, pointer)
        declare("ckl_buffer_create", pointer, pointer, size)
        declare("ckl_buffer_import", pointer, pointer, size, size)
        declare("ckl_buffer_free", None, pointer)
        declare("ckl_buffer_address", size, pointer)
        declare("ckl_buffer_size", size, pointer)
        declare("ckl_buffer_copy", ct.c_int, pointer, pointer, size, ct.c_int)
        declare("ckl_bind", pointer, pointer, pointer, pointer, size)
        declare("ckl_invoke", ct.c_int, pointer, ct.c_int)
        declare("ckl_invocation_free", None, pointer)
        declare("ckl_invocation_nodes", size, pointer)
        declare("ckl_benchmark", ct.c_int, pointer, ct.c_int, size, ct.POINTER(ct.c_double))
    _library = lib
    return lib


def _checked(lib, result):
    if not result:
        raise RuntimeError(lib.ckl_error().decode() or "CKL native operation failed")
    return result


class _Owned:
    _handle = None

    def __del__(self):
        if self._handle is not None:
            getattr(self._lib, self._release)(self._handle)
            self._handle = None


class Device(_Owned):
    """Own CUDA device 0's runtime. Construction initializes CUDA; compilation does not."""
    _release = "ckl_device_free"

    def __init__(self):
        self._lib = _native()
        if not self._lib.ckl_has_cuda_runtime():
            raise RuntimeError("rebuild CKL with CKL_ENABLE_CUDA_RUNTIME=ON to execute programs")
        self._handle = _checked(self._lib, self._lib.ckl_device_create())

    @property
    def name(self):
        return _checked(self._lib, self._lib.ckl_device_name(self._handle)).decode()

    def empty(self, spec: BufferSpec) -> DeviceBuffer:
        size = _byte_count(spec)
        handle = _checked(self._lib, self._lib.ckl_buffer_create(self._handle, size))
        return DeviceBuffer(self, handle, spec)

    def import_buffer(self, address: int, spec: BufferSpec, *, owner) -> DeviceBuffer:
        """Borrow device storage, retaining its Python owner for the buffer's lifetime."""
        size = _byte_count(spec)
        if type(address) is not int or not 0 < address < (1 << 64) or owner is None:
            raise ValueError("imported storage requires a valid address and a live owner")
        handle = _checked(self._lib, self._lib.ckl_buffer_import(self._handle, address, size))
        return DeviceBuffer(self, handle, spec, owner)


class DeviceBuffer(_Owned):
    _release = "ckl_buffer_free"

    def __init__(self, device, handle, spec, owner=None):
        self.device, self.spec, self._owner = device, spec, owner
        self._lib, self._handle = device._lib, handle

    @property
    def address(self):
        return self._lib.ckl_buffer_address(self._handle)

    @property
    def nbytes(self):
        return self._lib.ckl_buffer_size(self._handle)

    def copy_from_host(self, source):
        view = memoryview(source).cast("B")
        if view.nbytes != self.nbytes:
            raise ValueError("host buffer byte count does not match device storage")
        storage = ct.create_string_buffer(view.tobytes())
        _checked(self._lib, self._lib.ckl_buffer_copy(self._handle, storage, self.nbytes, 1))

    def copy_to_host(self, destination):
        view = memoryview(destination).cast("B")
        if view.readonly or view.nbytes != self.nbytes:
            raise ValueError("destination must be writable and match device storage byte count")
        storage = (ct.c_byte * view.nbytes).from_buffer(view)
        _checked(self._lib, self._lib.ckl_buffer_copy(self._handle, storage, self.nbytes, 0))


class CompiledProgram(_Owned):
    """Own specialized device code and its JIT-compiled host entry point."""
    _release = "ckl_program_free"

    def __init__(self, program, target: NVIDIATarget, *, fuse_pointwise=True):
        if type(fuse_pointwise) is not bool:
            raise TypeError("fuse_pointwise must be bool")
        if target.binary_format != "bin":
            raise ValueError("CKL CUDA compilation requires a binary target")
        if not re.fullmatch(r"sm_[0-9]+[a-z]?", target.chip) or not re.fullmatch(
            r"[+\-a-zA-Z0-9_,]+", target.features
        ):
            raise ValueError("invalid CUDA chip or features")
        self.signature, self.target = program.signature, target
        self._lib = _native()
        toolkit = str(Path(target.toolkit_root).resolve()) if target.toolkit_root else ""
        self._handle = _checked(self._lib, self._lib.ckl_compile(
            program.mlir.encode(), target.chip.encode(), target.features.encode(), toolkit.encode(),
            fuse_pointwise))

    @property
    def prepared_ir(self):
        return self._lib.ckl_program_ir(self._handle, 0).decode()

    @property
    def executable_ir(self):
        return self._lib.ckl_program_ir(self._handle, 1).decode()

    @property
    def device_ir(self):
        return self._lib.ckl_program_ir(self._handle, 2).decode()

    @property
    def host_ir(self):
        return self._lib.ckl_program_ir(self._handle, 3).decode()

    def bind(self, *arguments, device: Device | None = None) -> Invocation:
        parameters = self.signature.runtime
        if len(arguments) != len(parameters):
            raise TypeError(f"expected {len(parameters)} runtime arguments, got {len(arguments)}")
        buffers = [a for a in arguments if isinstance(a, DeviceBuffer)]
        device = device or (buffers[0].device if buffers else None)
        if device is None:
            raise ValueError("pass a Device for programs without buffer arguments")
        if device._lib is not self._lib:
            raise ValueError("program and device must use the same native library")
        storage, packed = [], (ct.c_void_p * len(arguments))()
        for index, (parameter, value) in enumerate(zip(parameters, arguments)):
            if isinstance(parameter.dtype, BufferSpec):
                if not isinstance(value, DeviceBuffer) or value.spec != parameter.dtype:
                    raise TypeError(f"{parameter.name} requires {parameter.dtype}")
                if value.device is not device:
                    raise ValueError("all buffers must belong to the selected Device")
                packed[index] = value._handle
            else:
                scalar, _ = _SCALARS[parameter.dtype]
                if parameter.dtype in (i32, i64):
                    width = ct.sizeof(scalar) * 8
                    if type(value) is not int or not -(1 << (width - 1)) <= value < (1 << (width - 1)):
                        raise TypeError(f"{parameter.name} requires a {width}-bit integer")
                elif type(value) not in (int, float) or not math.isfinite(float(value)):
                    raise TypeError(f"{parameter.name} requires a finite numeric value")
                holder = scalar(value)
                if parameter.dtype in (f32, f64) and not math.isfinite(holder.value):
                    raise ValueError(f"{parameter.name} overflows {parameter.dtype.name}")
                storage.append(holder)
                packed[index] = ct.addressof(holder)
        handle = _checked(self._lib, self._lib.ckl_bind(self._handle, device._handle,
                                                      packed, len(arguments)))
        return Invocation(self, device, arguments, handle)

    def run(self, *arguments, graph=False, device=None) -> Invocation:
        return self.bind(*arguments, device=device).run(graph=graph)


class Invocation(_Owned):
    """A bound program, owning temporaries and a lazily instantiated CUDA graph."""
    _release = "ckl_invocation_free"

    def __init__(self, program, device, arguments, handle):
        self.program, self.device, self.arguments = program, device, arguments
        self._lib, self._handle = program._lib, handle

    def run(self, *, graph=False):
        # Synchronous execution keeps Python owners and temporary heaps alive
        # through completion. Reusing an invocation reuses its graph and heaps.
        _checked(self._lib, self._lib.ckl_invoke(self._handle, bool(graph)))
        return self

    @property
    def node_count(self):
        """Actual bound topology, including any overlap-triggered fallback."""
        return self._lib.ckl_invocation_nodes(self._handle)

    def benchmark(self, *, iterations=200, graph=False):
        """Measure a native submission burst; CUDA stream time includes submission gaps."""
        if type(iterations) is not int or not 0 < iterations < (1 << 64):
            raise ValueError("iterations must be a positive 64-bit count")
        result = (ct.c_double * 3)()
        _checked(self._lib, self._lib.ckl_benchmark(self._handle, bool(graph), iterations, result))
        return dict(stream_us=result[0], wall_us=result[1], graph_setup_ms=result[2])

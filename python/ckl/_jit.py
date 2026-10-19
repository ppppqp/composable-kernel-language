"""MLIR pass pipelines and ExecutionEngine ownership for the CUDA frontend."""

from __future__ import annotations

import ctypes
import json
import re
from pathlib import Path

from mlir import ir
from mlir.dialects import func, gpu
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager

from .types import BufferSpec


def _clone(module):
    result = ir.Module.create(loc=module.operation.location)
    for name in module.operation.attributes:
        result.operation.attributes[name] = module.operation.attributes[name]
    for operation in module.body.operations:
        operation.operation.clone(ip=ir.InsertionPoint(result.body))
    return result


def _pipeline(module, text):
    try:
        PassManager.parse(f"builtin.module({text})").run(module.operation)
    except ir.MLIRError as error:
        raise RuntimeError(f"CKL compilation failed: {error}") from error


def _device_pipeline(target):
    # Expand the upstream pipeline, then configure its serializer explicitly.
    # This retains upstream pass ordering without changing CUDA_HOME or PATH.
    manager = PassManager.parse(
        "builtin.module(ckl-lower-kernels-to-gpu,gpu-lower-to-nvvm-pipeline{"
        f"cubin-chip={target.chip} cubin-features={target.features} "
        "cubin-format=bin index-bitwidth=64 opt-level=3 "
        "kernel-bare-ptr-calling-convention=true})"
    )
    if target.toolkit_root:
        serializer = (
            "gpu-module-to-binary{format=bin toolkit="
            + json.dumps(str(Path(target.toolkit_root).resolve())) + "}"
        )
        pipeline, count = re.subn(r"gpu-module-to-binary\{[^}]*\}",
                                  lambda _: serializer, str(manager))
        if count != 1:
            raise RuntimeError("expected one CUDA binary serializer in the upstream pipeline")
        manager = PassManager.parse(pipeline)
    return manager


class CompiledState:
    """Retain all modules and the engine independently of public wrapper scopes."""

    def ir(self, stage):
        with self.context:
            if stage == 0:
                return self.prepared_ir
            if stage == 1:
                return self.executable_ir
            if stage == 2:
                return self.device_ir
            if stage == 3:
                return self.extension.translate_llvm(self.host._CAPIPtr)
            if stage == 4:
                return str(self.host)
        raise ValueError("invalid IR stage")


def compile_program(program, target, extension, *, fusion, reuse_storage):
    from .cuda import _byte_count
    from .frontend.emitter import mlir_type

    state = CompiledState()
    state.extension = extension
    state.context = ir.Context()
    extension.register_context(state.context._CAPIPtr)
    with state.context, ir.Location.unknown():
        try:
            state.source = ir.Module.parse(program.mlir)
        except ir.MLIRError as error:
            raise RuntimeError(f"invalid CKL source module: {error}") from error
        if "ckl.fusion.requires_disjoint" in state.source.operation.attributes:
            raise RuntimeError("compile original source, not pre-fused guarded IR")
        if fusion:
            _pipeline(state.source, "ckl-fuse-pointwise")
        needs_fallback = "ckl.fusion.requires_disjoint" in state.source.operation.attributes
        _pipeline(state.source, f"ckl-prepare-cuda{{chip={target.chip} features={target.features}}}")
        state.prepared_ir = str(state.source)
        state.executable = _clone(state.source)
        _pipeline(state.executable,
                  f"ckl-build-graph{{reuse-storage={str(reuse_storage).lower()}}},"
                  "ckl-lower-graph-to-exec")
        state.executable_ir = str(state.executable)
        plans = [op for op in state.executable.body.operations
                 if op.operation.name == "ckl_exec.plan"]
        if len(plans) != 1:
            raise RuntimeError("compilation requires exactly one host program")
        plan = plans[0]
        source_name = ir.FlatSymbolRefAttr(plan.attributes["source"]).value
        source = ir.SymbolTable(state.executable.operation)[source_name]
        source_type = ir.FunctionType(ir.TypeAttr(source.attributes["function_type"]).value)
        if tuple(source_type.inputs) != tuple(mlir_type(p.dtype) for p in program.signature.runtime):
            raise RuntimeError("compiled source signature does not match the Python signature")

        state.device = _clone(state.source)
        try:
            _device_pipeline(target).run(state.device.operation)
        except ir.MLIRError as error:
            raise RuntimeError(f"CKL compilation failed: {error}") from error
        state.device_ir = str(state.device)
        objects = []
        for op in state.device.body.operations:
            if op.operation.name == "gpu.binary":
                objects.extend(gpu.ObjectAttr(attr) for attr in ir.ArrayAttr(op.attributes["objects"]))
        if (len(objects) != 1 or objects[0].format != gpu.CompilationTarget.Binary
                or not str(objects[0].target).startswith("#nvvm.target")
                or not objects[0].object.startswith(b"\x7fELF")):
            raise RuntimeError("expected exactly one NVIDIA ELF binary object")

        state.host = ir.Module.create()
        with ir.InsertionPoint(state.host.body):
            func.FuncOp(source_name, source_type, visibility="private")
            plan.operation.clone()
        state.host.operation.attributes["ckl_exec.binary"] = ir.StringAttr.get(objects[0].object)
        _pipeline(state.host, "ckl-lower-exec-to-llvm")

        state.fallback = (compile_program(program, target, extension, fusion=False,
                                         reuse_storage=reuse_storage) if needs_fallback else None)
        state.engine = None
        state.native = None
        if extension.has_cuda_runtime:
            # This entry constructs a plan once per bind; it is outside the
            # execution hot path. Avoid spending setup time optimizing it.
            state.engine = ExecutionEngine(state.host, opt_level=0)
            for name, address in extension.runtime_symbols().items():
                pointer = ctypes.c_void_p(address)
                # ExecutionEngine's native callback references survive until
                # after its JIT is destroyed. Keep its IR/context alive there.
                pointer._ckl_owner = (state.context, state.host, extension)
                state.engine.raw_register_runtime(name, pointer)
            state.engine.initialize()
            entry = state.engine.raw_lookup("ckl_entry")
            parameters = program.signature.runtime
            sizes = [_byte_count(p.dtype) if isinstance(p.dtype, BufferSpec) else 0
                     for p in parameters]
            types = ["" if isinstance(p.dtype, BufferSpec) else p.dtype.name for p in parameters]
            state.native = extension.make_program(
                entry, sizes, types, state.engine, state.fallback.native if state.fallback else None)
    return state

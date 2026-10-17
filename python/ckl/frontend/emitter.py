"""Typed AST lowering to deterministic, source-associated CKL MLIR."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path

from ..compiler import CompilerOptions, NVIDIATarget, compile_module
from ..types import BufferSpec, Dimension, ScalarType, bool_, f32, f64, i32, i64, index
from .checker import BoundSignature, bind, constant, resolve
from .source import Definition, Intrinsic, Namespace, UnsupportedBinding, fingerprint


@dataclass(frozen=True)
class Value:
    name: str
    dtype: ScalarType | BufferSpec


@dataclass(frozen=True)
class Specialization:
    symbol: str
    identity: str
    signature: BoundSignature


@dataclass(frozen=True)
class LoweredProgram:
    mlir: str
    signature: BoundSignature
    specializations: tuple[Specialization, ...]
    cache_key: str
    diagnostics: tuple[str, ...] = ()

    def analyze(self, *passes: str, ckl_opt: Path | None = None):
        """Verify or analyze with ckl-opt; never run a device compiler or a kernel."""
        return compile_module(self.mlir, CompilerOptions(ckl_opt=ckl_opt, passes=passes))

    def verify(self, *, ckl_opt: Path | None = None) -> LoweredProgram:
        self.analyze(ckl_opt=ckl_opt)
        return self

    def compile(self, *, target: NVIDIATarget, fuse_pointwise=True):
        """Compile device code and JIT its host entry point in this process."""
        from ..cuda import CompiledProgram
        return CompiledProgram(self, target, fuse_pointwise=fuse_pointwise)


def identity(definition: Definition, signature: BoundSignature) -> str:
    payload = [definition.identity, [
        [p.name, fingerprint(p.dtype) if isinstance(p.dtype, ScalarType) else
         [p.dtype.shape, p.dtype.dtype.name, p.dtype.layout], p.value, p.is_constant]
        for p in signature.parameters
    ]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def quoted(text: str) -> str:
    # MLIR string escapes use hexadecimal bytes, rather than Python/JSON unicode escapes.
    return '"' + ''.join(chr(byte) if 32 <= byte < 127 and byte not in (34, 92)
                         else f"\\{byte:02X}" for byte in text.encode("utf-8")) + '"'


class ModuleEmitter:
    def __init__(self):
        self.kernels: dict[str, tuple[Specialization, str]] = {}

    def specialize(self, definition: Definition, signature: BoundSignature) -> Specialization:
        key = identity(definition, signature)
        if key not in self.kernels:
            specialization = Specialization("kernel_" + key[:24], key, signature)
            body = FunctionEmitter(self, definition, signature).build(specialization.symbol)
            self.kernels[key] = specialization, body
        return self.kernels[key][0]


class FunctionEmitter:
    def __init__(self, module: ModuleEmitter, definition: Definition, signature: BoundSignature):
        self.module = module
        self.definition = definition
        self.signature = signature
        self.env: dict[str, object] = {}
        self.lines: list[str] = []
        self.indent = 2
        self.counter = 0
        self.temporaries: list[Value] = []
        self.local_names = {p.name for p in signature.parameters} | {
            node.id for node in ast.walk(definition.tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.dimensions = dict(signature.dimensions)
        for position, parameter in enumerate(signature.runtime):
            self.env[parameter.name] = Value(f"%arg{position}", parameter.dtype)
        self.env.update({p.name: p.value for p in signature.parameters if p.is_constant})

    def fail(self, node: ast.AST, message: str):
        raise self.definition.error(node, message)

    def location(self, node: ast.AST) -> str:
        line = self.definition.first_line + getattr(node, "lineno", 1) - 1
        column = self.definition.indentation + getattr(node, "col_offset", 0) + 1
        return f"loc({quoted(self.definition.filename)}:{line}:{column})"

    def line(self, text: str, node: ast.AST | None = None):
        self.lines.append("  " * self.indent + text + (" " + self.location(node) if node else ""))

    def emit(self, text: str, dtype: ScalarType | BufferSpec, node: ast.AST) -> Value:
        value = Value(f"%v{self.counter}", dtype)
        self.counter += 1
        self.line(f"{value.name} = {text}", node)
        return value

    def build(self, symbol: str) -> str:
        args = ", ".join(f"%arg{i}: {p.dtype.mlir}" for i, p in enumerate(self.signature.runtime))
        marker = " attributes {ckl.kernel}" if self.definition.kind == "kernel" else ""
        self.lines.append(f"  func.func @{symbol}({args}){marker} {{")
        self.statements(self.definition.tree.body, top=True)
        for temporary in reversed(self.temporaries):
            self.line(f'"ckl.free"({temporary.name}) : ({temporary.dtype.mlir}) -> ()', self.definition.tree)
        self.line("func.return", self.definition.tree)
        self.lines.append("  } " + self.location(self.definition.tree))
        return "\n".join(self.lines)

    def typed(self, value: object, dtype: ScalarType, node: ast.AST) -> Value:
        if isinstance(value, Value):
            if value.dtype != dtype:
                self.fail(node, f"expected {dtype.name}, got {value.dtype.mlir}; use ckl.cast explicitly")
            return value
        value = constant(value, dtype, self.definition, node)
        literal = ("true" if value else "false") if dtype == bool_ else (
            format(value, ".17e") if dtype in (f32, f64) else str(value)
        )
        return self.emit(f"arith.constant {literal} : {dtype.mlir}", dtype, node)

    def buffer(self, node: ast.AST) -> Value:
        value = self.expression(node)
        if not isinstance(value, Value) or not isinstance(value.dtype, BufferSpec):
            self.fail(node, "expected a typed buffer")
        return value

    def indices(self, buffer: Value, node: ast.Subscript) -> list[Value]:
        nodes = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(nodes) != len(buffer.dtype.shape):
            self.fail(node, "requires one index per buffer dimension")
        result = []
        for axis, item in enumerate(nodes):
            value = self.expression(item)
            if type(value) is int and not 0 <= value < buffer.dtype.shape[axis]:
                self.fail(item, f"constant index is out of bounds for dimension {axis}")
            result.append(self.typed(value, index, item))
        return result

    def expression(self, node: ast.AST) -> object:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            if type(node.value) is float and not math.isfinite(node.value):
                self.fail(node, "non-finite source constants are unsupported")
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            if node.id in self.local_names:
                self.fail(node, f"local '{node.id}' is not defined on this path")
            if node.id not in self.definition.bindings:
                self.fail(node, f"unknown name '{node.id}'")
            captured = self.definition.bindings[node.id]
            if isinstance(captured, UnsupportedBinding):
                self.fail(node, f"unsupported captured binding '{node.id}'")
            if isinstance(captured, Dimension):
                if captured.name not in self.dimensions:
                    self.fail(node, f"unbound dimension '{captured.name}'")
                return self.dimensions[captured.name]
            return captured
        if isinstance(node, ast.Tuple):
            return tuple(self.expression(item) for item in node.elts)
        if isinstance(node, ast.Attribute):
            base = self.expression(node.value)
            if isinstance(base, Namespace) and node.attr in base.values:
                return base.values[node.attr]
            if isinstance(base, Value) and isinstance(base.dtype, BufferSpec):
                if node.attr == "shape":
                    return base.dtype.shape
                if node.attr == "dtype":
                    return base.dtype.dtype
            self.fail(node, f"unsupported attribute '{node.attr}'")
        if isinstance(node, ast.Subscript):
            base = self.expression(node.value)
            if isinstance(base, Value) and isinstance(base.dtype, BufferSpec):
                if self.definition.kind != "kernel":
                    self.fail(node, "buffer contents can only be accessed inside a kernel")
                indices = self.indices(base, node)
                operands = [base, *indices]
                return self.emit(
                    f'"ckl.load_element"({", ".join(v.name for v in operands)}) : '
                    f'({", ".join(v.dtype.mlir for v in operands)}) -> {base.dtype.dtype.mlir}',
                    base.dtype.dtype, node,
                )
            if type(base) is tuple:
                position = self.expression(node.slice)
                if type(position) is not int or not 0 <= position < len(base):
                    self.fail(node, "tuple/shape axes require an in-range nonnegative compile-time integer")
                return base[position]
            self.fail(node, "unsupported subscript; slices and negative indexing are not supported")
        if isinstance(node, ast.BinOp):
            return self.binary(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.expression(node.operand)
            if type(value) in (int, float):
                return -value if isinstance(node.op, ast.USub) else value
            if not isinstance(value, Value) or not isinstance(value.dtype, ScalarType) or value.dtype == bool_:
                self.fail(node, "unary arithmetic requires a numeric scalar")
            self.require_kernel_arithmetic(node)
            if isinstance(node.op, ast.UAdd):
                return value
            if value.dtype in (f32, f64):
                return self.emit(f"arith.negf {value.name} : {value.dtype.mlir}", value.dtype, node)
            zero = self.typed(0, value.dtype, node)
            return self.emit(f"arith.subi {zero.name}, {value.name} : {value.dtype.mlir}", value.dtype, node)
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                self.fail(node, "chained comparisons are not supported")
            return self.binary(node, comparison=True)
        if isinstance(node, ast.Call):
            return self.call(node)
        self.fail(node, f"unsupported expression: {type(node).__name__}")

    def require_kernel_arithmetic(self, node: ast.AST):
        if self.definition.kind != "kernel":
            self.fail(node, "computed runtime host scalars are unsupported; forward the parameter unchanged")

    def binary(self, node: ast.AST, comparison=False) -> object:
        left = self.expression(node.left)
        right = self.expression(node.comparators[0] if comparison else node.right)
        op = node.ops[0] if comparison else node.op
        static_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                      ast.Div: operator.truediv, ast.Lt: operator.lt, ast.LtE: operator.le,
                      ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq,
                      ast.NotEq: operator.ne}
        if type(op) not in static_ops:
            self.fail(node, f"unsupported numeric operator: {type(op).__name__}")
        if not isinstance(left, Value) and not isinstance(right, Value):
            allowed = (int, float, bool) if comparison else (int, float)
            if type(left) not in allowed or type(right) not in allowed:
                self.fail(node, "numeric expressions require scalar operands")
            try:
                result = static_ops[type(op)](left, right)
                if type(result) is float and not math.isfinite(result):
                    self.fail(node, "non-finite constant expression")
                return result
            except (ZeroDivisionError, OverflowError) as error:
                self.fail(node, f"invalid constant expression: {error}")
        self.require_kernel_arithmetic(node)
        dynamic = left if isinstance(left, Value) else right
        dtype = dynamic.dtype
        if not isinstance(dtype, ScalarType) or (dtype == bool_ and type(op) not in (ast.Eq, ast.NotEq)):
            self.fail(node, "numeric expressions require compatible scalar types")
        left, right = self.typed(left, dtype, node), self.typed(right, dtype, node)
        floating = dtype in (f32, f64)
        if comparison:
            predicates = {ast.Lt: "olt" if floating else "slt", ast.LtE: "ole" if floating else "sle",
                          ast.Gt: "ogt" if floating else "sgt", ast.GtE: "oge" if floating else "sge",
                          ast.Eq: "oeq" if floating else "eq", ast.NotEq: "une" if floating else "ne"}
            text = f"arith.cmp{'f' if floating else 'i'} {predicates[type(op)]}, {left.name}, {right.name} : {dtype.mlir}"
            return self.emit(text, bool_, node)
        if isinstance(op, ast.Div) and not floating:
            self.fail(node, "division requires floating operands")
        mnemonic = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}[type(op)]
        return self.emit(f"arith.{mnemonic}{'f' if floating else 'i'} {left.name}, {right.name} : {dtype.mlir}", dtype, node)

    def call(self, node: ast.Call) -> object:
        callee = self.expression(node.func)
        if isinstance(callee, Definition):
            self.fail(node, "kernel definitions are not ordinary calls; use ckl.launch")
        if not isinstance(callee, Intrinsic):
            self.fail(node, "calling arbitrary Python functions is unsupported")
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(k.arg is None for k in node.keywords):
            self.fail(node, "argument expansion is unsupported")
        name = callee.name
        if name == "launch":
            return self.launch(node)
        args = [self.expression(argument) for argument in node.args]
        keywords = {k.arg: self.expression(k.value) for k in node.keywords}
        if len(keywords) != len(node.keywords):
            self.fail(node, "duplicate keyword arguments")
        if name in ("global_id", "thread_id", "block_id", "block_dim", "grid_dim"):
            if self.definition.kind != "kernel":
                self.fail(node, "device coordinates are only valid in kernels")
            if len(args) != 1 or keywords or type(args[0]) is not int or not 0 <= args[0] <= 2:
                self.fail(node, "device coordinates require one constant axis 0, 1, or 2")
            return self.emit(f'"ckl.{name}"() {{axis = {args[0]} : i32}} : () -> index', index, node)
        if name == "cdiv":
            if len(args) != 2 or keywords or any(type(v) is not int for v in args) or args[0] < 0 or args[1] <= 0:
                self.fail(node, "cdiv requires a nonnegative constant extent and positive constant divisor")
            return (args[0] + args[1] - 1) // args[1]
        if name in ("empty", "empty_like"):
            if self.definition.kind != "program":
                self.fail(node, "program temporaries cannot be allocated inside kernels")
            if name == "empty_like":
                if len(args) != 1 or keywords or not isinstance(args[0], Value) or not isinstance(args[0].dtype, BufferSpec):
                    self.fail(node, "empty_like requires one buffer")
                spec = args[0].dtype
            else:
                if len(args) != 1 or set(keywords) != {"dtype"}:
                    self.fail(node, "use empty(shape_tuple, dtype=element_type)")
                try:
                    spec = BufferSpec(args[0], keywords["dtype"])
                except (ValueError, TypeError) as error:
                    self.fail(node, str(error))
            value = self.emit(f'"ckl.alloc"() : () -> {spec.mlir}', spec, node)
            self.temporaries.append(value)
            return value
        if name == "cast":
            if len(args) != 2 or keywords or not isinstance(args[1], ScalarType):
                self.fail(node, "use cast(value, scalar_type)")
            return self.cast(args[0], args[1], node)
        self.fail(node, f"unsupported intrinsic: {name}")

    def cast(self, value: object, dtype: ScalarType, node: ast.AST) -> object:
        if not isinstance(value, Value):
            if dtype in (i32, i64, index) and type(value) is float and math.isfinite(value):
                value = int(value)
            return constant(value, dtype, self.definition, node)
        self.require_kernel_arithmetic(node)
        source = value.dtype
        if source == dtype:
            return value
        if not isinstance(source, ScalarType) or bool_ in (source, dtype):
            self.fail(node, "boolean/buffer casts are unsupported")
        floats = (f32, f64)
        integers = (i32, i64)
        if source in floats and dtype in floats:
            operation = "extf" if dtype == f64 else "truncf"
        elif source in integers and dtype in integers:
            operation = "extsi" if dtype == i64 else "trunci"
        elif source in floats and dtype in integers:
            operation = "fptosi"
        elif source in integers and dtype in floats:
            operation = "sitofp"
        elif index in (source, dtype) and (source in integers or dtype in integers):
            operation = "index_cast"
        else:
            self.fail(node, "cast through an integer type to convert between index and float")
        return self.emit(f"arith.{operation} {value.name} : {source.mlir} to {dtype.mlir}", dtype, node)

    def launch(self, node: ast.Call):
        if self.definition.kind != "program":
            self.fail(node, "kernels cannot launch other kernels")
        if not node.args:
            self.fail(node, "launch requires a kernel definition")
        definition = self.expression(node.args[0])
        if not isinstance(definition, Definition) or definition.kind != "kernel":
            self.fail(node, "launch requires a @ckl.kernel definition")
        parameters = definition.tree.args.args
        if len(node.args) - 1 != len(parameters):
            self.fail(node, "launch argument count does not match kernel signature")
        keywords = {k.arg: self.expression(k.value) for k in node.keywords}
        if set(keywords) - {"grid", "block", "shared_memory"} or not {"grid", "block"} <= set(keywords):
            self.fail(node, "launch requires grid and block; only shared_memory is optional")
        geometry = []
        for name in ("grid", "block"):
            dimensions = keywords[name]
            if type(dimensions) is not tuple or not 1 <= len(dimensions) <= 3 or any(
                type(d) is not int or not 0 < d < (1 << 63) for d in dimensions
            ):
                self.fail(node, f"{name} requires a tuple of one to three positive compile-time integers")
            geometry.append(dimensions + (1,) * (3 - len(dimensions)))
        shared = keywords.get("shared_memory", 0)
        if type(shared) is not int or not 0 <= shared < (1 << 63):
            self.fail(node, "shared_memory requires nonnegative compile-time bytes")
        signature, constants, operands = {}, {}, []
        from ..types import ConstexprType
        for parameter, argument in zip(parameters, node.args[1:]):
            annotation = definition.annotations.get(parameter.arg)
            if isinstance(annotation, str) or annotation is None:
                if parameter.annotation is None:
                    self.fail(node, "kernel parameters require annotations")
                annotation = resolve(parameter.annotation, definition)
            actual = self.expression(argument)
            if isinstance(annotation, ConstexprType):
                if isinstance(actual, Value):
                    self.fail(argument, "constexpr launch arguments must be compile-time constants")
                constants[parameter.arg] = actual
                continue
            if isinstance(annotation, ScalarType):
                actual = self.typed(actual, annotation, argument)
            if not isinstance(actual, Value):
                self.fail(argument, "kernel arguments must be typed buffers or scalars")
            signature[parameter.arg] = actual.dtype
            operands.append(actual)
        bound = bind(definition, signature, constants)
        specialization = self.module.specialize(definition, bound)
        grid, block = geometry
        grid_text, block_text = ", ".join(map(str, grid)), ", ".join(map(str, block))
        text = (f'"ckl.dispatch"({", ".join(v.name for v in operands)}) {{'
                f'kernel = @{specialization.symbol}, grid = array<i64: {grid_text}>, '
                f'block = array<i64: {block_text}>, shared_memory = {shared} : i64, '
                f'device = "unassigned:0", capabilities = [], implementation = "{specialization.identity}", '
                f'operandSegmentSizes = array<i32: {len(operands)}, 0>}} : '
                f'({", ".join(v.dtype.mlir for v in operands)}) -> !ckl.token')
        # Completion values are emitted, but memory dependencies are inferred rather than chained.
        self.emit(text, ScalarType("!ckl.token"), node)
        return None

    def statements(self, nodes: list[ast.stmt], *, top=False, branch_outer=None):
        for position, node in enumerate(nodes):
            if isinstance(node, ast.Return):
                if not top or position != len(nodes) - 1 or node.value is not None:
                    self.fail(node, "only a final bare return is supported; buffers carry outputs")
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    self.fail(node, "chained assignment is unsupported")
                target = node.targets[0]
                value = self.expression(node.value)
                if isinstance(target, ast.Name):
                    if value is None:
                        self.fail(node, "launches do not return source values")
                    if branch_outer is not None and target.id in branch_outer:
                        self.fail(target, "runtime branch results are deferred; write buffers or use branch-local variables")
                    old = self.env.get(target.id)
                    if isinstance(old, Value) and isinstance(old.dtype, ScalarType):
                        value = self.typed(value, old.dtype, node.value)
                    self.env[target.id] = value
                elif isinstance(target, ast.Subscript):
                    if self.definition.kind != "kernel":
                        self.fail(target, "buffer writes are only valid in kernels")
                    buffer = self.buffer(target.value)
                    indices = self.indices(buffer, target)
                    value = self.typed(value, buffer.dtype.dtype, node.value)
                    operands = [value, buffer, *indices]
                    self.line(f'"ckl.store_element"({", ".join(v.name for v in operands)}) : '
                              f'({", ".join(v.dtype.mlir for v in operands)}) -> ()', node)
                else:
                    self.fail(target, "assignment requires a local name or indexed buffer element")
            elif isinstance(node, ast.If):
                predicate = self.expression(node.test)
                if type(predicate) is bool:
                    self.statements(node.body if predicate else node.orelse, branch_outer=branch_outer)
                    continue
                if self.definition.kind != "kernel" or not isinstance(predicate, Value) or predicate.dtype != bool_:
                    self.fail(node.test, "if requires a boolean predicate; program control flow must be compile-time")
                outer = self.env.copy()
                self.line(f"scf.if {predicate.name} {{")
                self.indent += 1
                self.statements(node.body, branch_outer=set(outer))
                self.line("scf.yield")
                self.indent -= 1
                self.env = outer.copy()
                if node.orelse:
                    self.line("} else {")
                    self.indent += 1
                    self.statements(node.orelse, branch_outer=set(outer))
                    self.line("scf.yield")
                    self.indent -= 1
                self.line("}", node)
                self.env = outer
            elif isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and position == 0 and top:
                    continue  # source docstring
                if not isinstance(node.value, ast.Call) or self.expression(node.value) is not None:
                    self.fail(node, "expression statements must be kernel launches")
            elif isinstance(node, ast.Pass):
                continue
            else:
                self.fail(node, f"unsupported statement: {type(node).__name__}; loops and richer control flow are deferred")


def lower_program(definition: Definition, signature, constants, *, verify=False, ckl_opt=None):
    bound = bind(definition, signature, constants)
    module = ModuleEmitter()
    program_identity = identity(definition, bound)
    host = FunctionEmitter(module, definition, bound).build("program_" + program_identity[:24])
    kernels = [body for _, body in module.kernels.values()]
    mlir = "module {\n" + "\n\n".join([*kernels, host]) + "\n}\n"
    semantic = re.sub(r' loc\("(?:[^"\\]|\\.)*":\d+:\d+\)', "", mlir)
    key = hashlib.sha256((program_identity + semantic).encode()).hexdigest()
    result = LoweredProgram(mlir, bound, tuple(s for s, _ in module.kernels.values()), key)
    if verify:
        result.verify(ckl_opt=ckl_opt)
    return result

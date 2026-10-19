"""Typed AST lowering to deterministic, source-associated CKL MLIR."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
from dataclasses import dataclass
from pathlib import Path

from mlir import ir
from mlir.dialects import arith, func

from ..compiler import CompilerOptions, NVIDIATarget, compile_module
from ..types import BufferSpec, Dimension, ScalarType, bool_, f32, f64, i32, i64, index
from .checker import BoundSignature, bind, constant, resolve
from .source import Definition, Intrinsic, Namespace, UnsupportedBinding, fingerprint


@dataclass(frozen=True)
class Value:
    ir: ir.Value
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

    def compile(self, *, target: NVIDIATarget, fuse_pointwise=True, reuse_storage=True):
        """Compile device code and JIT its host entry point in this process."""
        from ..cuda import CompiledProgram

        return CompiledProgram(
            self, target, fuse_pointwise=fuse_pointwise, reuse_storage=reuse_storage
        )


def identity(definition: Definition, signature: BoundSignature) -> str:
    payload = [
        definition.identity,
        [
            [
                p.name,
                fingerprint(p.dtype)
                if isinstance(p.dtype, ScalarType)
                else [p.dtype.shape, p.dtype.dtype.name, p.dtype.layout],
                p.value,
                p.is_constant,
            ]
            for p in signature.parameters
        ],
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def mlir_type(dtype: ScalarType | BufferSpec) -> ir.Type:
    if isinstance(dtype, BufferSpec):
        return ir.MemRefType.get(dtype.shape, mlir_type(dtype.dtype))
    if dtype == index:
        return ir.IndexType.get()
    if dtype in (f32, f64):
        return (ir.F32Type if dtype == f32 else ir.F64Type).get()
    if dtype.name == "!ckl.token":
        return ir.OpaqueType.get("ckl", "token")
    return ir.IntegerType.get_signless({bool_: 1, i32: 32, i64: 64}[dtype])


class IRBuilder:
    """Build operations with SSA operands at the current MLIR insertion point."""

    @staticmethod
    def create(name, *, operands=(), results=(), attributes=None, regions=0, loc=None):
        return ir.Operation.create(
            name,
            operands=[v.ir for v in operands],
            results=[mlir_type(t) for t in results],
            attributes=attributes,
            regions=regions,
            loc=loc,
        )


class ModuleEmitter:
    def __init__(self):
        self.ir = ir.Module.create()
        self.kernels: dict[str, tuple[Specialization, func.FuncOp]] = {}

    def specialize(self, definition: Definition, signature: BoundSignature) -> Specialization:
        key = identity(definition, signature)
        if key not in self.kernels:
            # if the kernel has not been specialized yet, emit its MLIR body and record the specialization
            specialization = Specialization("kernel_" + key[:24], key, signature)
            body = FunctionEmitter(self, definition, signature).build(specialization.symbol)
            self.kernels[key] = specialization, body
        return self.kernels[key][0]


class FunctionEmitter:
    """Lower a typed AST directly to MLIR operations, regions, and SSA values."""

    def __init__(self, module: ModuleEmitter, definition: Definition, signature: BoundSignature):
        self.module = module
        self.definition = definition
        self.signature = signature
        self.env: dict[str, object] = {}
        self.builder = IRBuilder()
        self.temporaries: list[Value] = []
        self.local_names = {p.name for p in signature.parameters} | {
            node.id
            for node in ast.walk(definition.tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.dimensions = dict(signature.dimensions)
        self.env.update({p.name: p.value for p in signature.parameters if p.is_constant})

    def fail(self, node: ast.AST, message: str):
        raise self.definition.error(node, message)

    def location(self, node: ast.AST) -> ir.Location:
        line = self.definition.first_line + getattr(node, "lineno", 1) - 1
        column = self.definition.indentation + getattr(node, "col_offset", 0) + 1
        return ir.Location.file(self.definition.filename, line, column)

    def operation(self, name, node, *, operands=(), results=(), attributes=None, regions=0):
        return self.builder.create(
            name,
            operands=operands,
            results=results,
            attributes=attributes,
            regions=regions,
            loc=self.location(node),
        )

    def emit(self, name, dtype, node, *, operands=(), attributes=None) -> Value:
        operation = self.operation(
            name, node, operands=operands, results=[dtype], attributes=attributes
        )
        return Value(operation.result, dtype)

    def build(self, symbol: str) -> func.FuncOp:
        with ir.InsertionPoint(self.module.ir.body):
            function = func.FuncOp(
                symbol,
                ([mlir_type(p.dtype) for p in self.signature.runtime], []),
                loc=self.location(self.definition.tree),
            )
        if self.definition.kind == "kernel":
            function.attributes["ckl.kernel"] = ir.UnitAttr.get()
        block = function.add_entry_block()
        for parameter, argument in zip(self.signature.runtime, block.arguments):
            self.env[parameter.name] = Value(argument, parameter.dtype)
        with ir.InsertionPoint(block):
            self.statements(self.definition.tree.body, top=True)
            for temporary in reversed(self.temporaries):
                self.operation("ckl.free", self.definition.tree, operands=[temporary])
            self.operation("func.return", self.definition.tree)
        return function

    def typed(self, value: object, dtype: ScalarType, node: ast.AST) -> Value:
        if isinstance(value, Value):
            if value.dtype != dtype:
                self.fail(
                    node, f"expected {dtype.name}, got {value.dtype.mlir}; use ckl.cast explicitly"
                )
            return value
        value = constant(value, dtype, self.definition, node)
        ty = mlir_type(dtype)
        attribute = (
            ir.FloatAttr.get(ty, value)
            if dtype in (f32, f64)
            else ir.IntegerAttr.get(ty, int(value))
        )
        return self.emit("arith.constant", dtype, node, attributes={"value": attribute})

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
                return self.emit("ckl.load_element", base.dtype.dtype, node, operands=operands)
            if type(base) is tuple:
                position = self.expression(node.slice)
                if type(position) is not int or not 0 <= position < len(base):
                    self.fail(
                        node,
                        "tuple/shape axes require an in-range nonnegative compile-time integer",
                    )
                return base[position]
            self.fail(node, "unsupported subscript; slices and negative indexing are not supported")
        if isinstance(node, ast.BinOp):
            return self.binary(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.expression(node.operand)
            if type(value) in (int, float):
                return -value if isinstance(node.op, ast.USub) else value
            if (
                not isinstance(value, Value)
                or not isinstance(value.dtype, ScalarType)
                or value.dtype == bool_
            ):
                self.fail(node, "unary arithmetic requires a numeric scalar")
            self.require_kernel_arithmetic(node)
            if isinstance(node.op, ast.UAdd):
                return value
            if value.dtype in (f32, f64):
                return self.emit("arith.negf", value.dtype, node, operands=[value])
            zero = self.typed(0, value.dtype, node)
            return self.emit("arith.subi", value.dtype, node, operands=[zero, value])
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                self.fail(node, "chained comparisons are not supported")
            return self.binary(node, comparison=True)
        if isinstance(node, ast.Call):
            return self.call(node)
        self.fail(node, f"unsupported expression: {type(node).__name__}")

    def require_kernel_arithmetic(self, node: ast.AST):
        if self.definition.kind != "kernel":
            self.fail(
                node,
                "computed runtime host scalars are unsupported; forward the parameter unchanged",
            )

    def binary(self, node: ast.AST, comparison=False) -> object:
        left = self.expression(node.left)
        right = self.expression(node.comparators[0] if comparison else node.right)
        op = node.ops[0] if comparison else node.op
        static_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
        }
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
        if not isinstance(dtype, ScalarType) or (
            dtype == bool_ and type(op) not in (ast.Eq, ast.NotEq)
        ):
            self.fail(node, "numeric expressions require compatible scalar types")
        left, right = self.typed(left, dtype, node), self.typed(right, dtype, node)
        floating = dtype in (f32, f64)
        if comparison:
            predicates = {
                ast.Lt: "olt" if floating else "slt",
                ast.LtE: "ole" if floating else "sle",
                ast.Gt: "ogt" if floating else "sgt",
                ast.GtE: "oge" if floating else "sge",
                ast.Eq: "oeq" if floating else "eq",
                ast.NotEq: "une" if floating else "ne",
            }
            predicate = getattr(
                arith.CmpFPredicate if floating else arith.CmpIPredicate,
                predicates[type(op)].upper() if floating else predicates[type(op)],
            )
            return self.emit(
                "arith.cmpf" if floating else "arith.cmpi",
                bool_,
                node,
                operands=[left, right],
                attributes={
                    "predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(predicate))
                },
            )
        if isinstance(op, ast.Div) and not floating:
            self.fail(node, "division requires floating operands")
        mnemonic = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}[type(op)]
        return self.emit(
            f"arith.{mnemonic}{'f' if floating else 'i'}", dtype, node, operands=[left, right]
        )

    def call(self, node: ast.Call) -> object:
        callee = self.expression(node.func)
        if isinstance(callee, Definition):
            self.fail(node, "kernel definitions are not ordinary calls; use ckl.launch")
        if not isinstance(callee, Intrinsic):
            self.fail(node, "calling arbitrary Python functions is unsupported")
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            k.arg is None for k in node.keywords
        ):
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
            return self.emit(
                f"ckl.{name}",
                index,
                node,
                attributes={"axis": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), args[0])},
            )
        if name == "cdiv":
            if (
                len(args) != 2
                or keywords
                or any(type(v) is not int for v in args)
                or args[0] < 0
                or args[1] <= 0
            ):
                self.fail(
                    node,
                    "cdiv requires a nonnegative constant extent and positive constant divisor",
                )
            return (args[0] + args[1] - 1) // args[1]
        if name in ("empty", "empty_like"):
            if self.definition.kind != "program":
                self.fail(node, "program temporaries cannot be allocated inside kernels")
            if name == "empty_like":
                if (
                    len(args) != 1
                    or keywords
                    or not isinstance(args[0], Value)
                    or not isinstance(args[0].dtype, BufferSpec)
                ):
                    self.fail(node, "empty_like requires one buffer")
                spec = args[0].dtype
            else:
                if len(args) != 1 or set(keywords) != {"dtype"}:
                    self.fail(node, "use empty(shape_tuple, dtype=element_type)")
                try:
                    spec = BufferSpec(args[0], keywords["dtype"])
                except (ValueError, TypeError) as error:
                    self.fail(node, str(error))
            value = self.emit("ckl.alloc", spec, node)
            self.temporaries.append(value)
            return value
        if name == "cast":
            if len(args) != 2 or keywords or not isinstance(args[1], ScalarType):
                self.fail(node, "use cast(value, scalar_type)")
            return self.cast(args[0], args[1], node)
        if name in ("exp", "rsqrt"):
            self.require_kernel_arithmetic(node)
            if (
                len(args) != 1
                or keywords
                or not isinstance(args[0], Value)
                or args[0].dtype not in (f32, f64)
            ):
                self.fail(node, f"{name} requires one typed floating-point scalar")
            return self.emit(f"math.{name}", args[0].dtype, node, operands=[args[0]])
        if name == "select":
            self.require_kernel_arithmetic(node)
            if len(args) != 3 or keywords:
                self.fail(node, "use select(predicate, true_value, false_value)")
            condition = self.typed(args[0], bool_, node)
            dynamic = next((v for v in args[1:] if isinstance(v, Value)), None)
            if dynamic is None or not isinstance(dynamic.dtype, ScalarType):
                self.fail(node, "select requires a typed scalar alternative")
            lhs, rhs = [self.typed(v, dynamic.dtype, node) for v in args[1:]]
            return self.emit("arith.select", lhs.dtype, node, operands=[condition, lhs, rhs])
        self.fail(node, f"unsupported intrinsic: {name}")

    def for_loop(self, node: ast.For, branch_outer):
        if (
            node.orelse
            or not isinstance(node.target, ast.Name)
            or not isinstance(node.iter, ast.Call)
        ):
            self.fail(node, "for requires a simple induction variable and range without else")
        callee = self.expression(node.iter.func)
        if not isinstance(callee, Intrinsic) or callee.name != "range" or node.iter.keywords:
            self.fail(node, "for requires range(start, stop, positive_constant_step)")
        bounds = [self.expression(arg) for arg in node.iter.args]
        if not 1 <= len(bounds) <= 3:
            self.fail(node, "range requires one to three bounds")
        start, stop, step = (
            (0, bounds[0], 1)
            if len(bounds) == 1
            else (bounds[0], bounds[1], bounds[2] if len(bounds) == 3 else 1)
        )
        if type(step) is not int or step <= 0:
            self.fail(node, "range step must be a positive compile-time integer")
        induction = node.target.id
        outer = self.env.copy()
        if self.definition.kind == "program":
            if type(start) is not int or type(stop) is not int:
                self.fail(node, "program range bounds must be compile-time integers")
            if len(range(start, stop, step)) > 1024:
                self.fail(node, "program unrolling is limited to 1024 iterations")
            for value in range(start, stop, step):
                self.env[induction] = value
                self.statements(node.body, branch_outer=branch_outer)
            if induction in outer:
                self.env[induction] = outer[induction]
            else:
                self.env.pop(induction, None)
            return
        assignments = {
            n.id
            for statement in node.body
            for n in ast.walk(statement)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        if induction in assignments:
            self.fail(node, "cannot assign to the loop induction variable")
        names = sorted(assignments & outer.keys())
        initials = []
        for name in names:
            if branch_outer is not None and name in branch_outer:
                self.fail(node, "runtime branch results are deferred")
            value = outer[name]
            if not isinstance(value, Value):
                dtype = f32 if type(value) is float else i32 if type(value) is int else bool_
                value = self.typed(value, dtype, node)
            if not isinstance(value.dtype, ScalarType):
                self.fail(node, "loop-carried locals must be scalars")
            initials.append(value)
        lower, upper, stride = [self.typed(v, index, node) for v in (start, stop, step)]

        loop = self.operation(
            "scf.for",
            node,
            operands=[lower, upper, stride, *initials],
            results=[v.dtype for v in initials],
            regions=1,
        )
        block = loop.regions[0].blocks.append(
            mlir_type(index), *[mlir_type(v.dtype) for v in initials]
        )
        self.env[induction] = Value(block.arguments[0], index)
        self.env.update(
            (name, Value(argument, initial.dtype))
            for name, argument, initial in zip(names, list(block.arguments)[1:], initials)
        )
        with ir.InsertionPoint(block):
            self.statements(node.body, branch_outer=branch_outer)
            yielded = [
                self.typed(self.env[name], initial.dtype, node)
                for name, initial in zip(names, initials)
            ]
            self.operation("scf.yield", node, operands=yielded)
        self.env = outer
        for name, initial, result in zip(names, initials, loop.results):
            self.env[name] = Value(result, initial.dtype)

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
        return self.emit(f"arith.{operation}", dtype, node, operands=[value])

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
        if set(keywords) - {"grid", "block", "shared_memory"} or not {"grid", "block"} <= set(
            keywords
        ):
            self.fail(node, "launch requires grid and block; only shared_memory is optional")
        geometry = []
        for name in ("grid", "block"):
            dimensions = keywords[name]
            if (
                type(dimensions) is not tuple
                or not 1 <= len(dimensions) <= 3
                or any(type(d) is not int or not 0 < d < (1 << 63) for d in dimensions)
            ):
                self.fail(
                    node, f"{name} requires a tuple of one to three positive compile-time integers"
                )
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
        attributes = {
            "kernel": ir.FlatSymbolRefAttr.get(specialization.symbol),
            "grid": ir.DenseI64ArrayAttr.get(grid),
            "block": ir.DenseI64ArrayAttr.get(block),
            "shared_memory": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), shared),
            "device": ir.StringAttr.get("unassigned:0"),
            "capabilities": ir.ArrayAttr.get([]),
            "implementation": ir.StringAttr.get(specialization.identity),
            "operandSegmentSizes": ir.DenseI32ArrayAttr.get([len(operands), 0]),
        }
        # Dependencies are inferred from effects rather than chaining completion tokens.
        self.emit(
            "ckl.dispatch", ScalarType("!ckl.token"), node, operands=operands, attributes=attributes
        )

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
                        self.fail(
                            target,
                            "runtime branch results are deferred; write buffers or use branch-local variables",
                        )
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
                    self.operation("ckl.store_element", node, operands=operands)
                else:
                    self.fail(target, "assignment requires a local name or indexed buffer element")
            elif isinstance(node, ast.If):
                predicate = self.expression(node.test)
                if type(predicate) is bool:
                    self.statements(
                        node.body if predicate else node.orelse, branch_outer=branch_outer
                    )
                    continue
                if (
                    self.definition.kind != "kernel"
                    or not isinstance(predicate, Value)
                    or predicate.dtype != bool_
                ):
                    self.fail(
                        node.test,
                        "if requires a boolean predicate; program control flow must be compile-time",
                    )
                outer = self.env.copy()
                branch = self.operation("scf.if", node, operands=[predicate], regions=2)
                then_block = branch.regions[0].blocks.append()
                with ir.InsertionPoint(then_block):
                    self.statements(node.body, branch_outer=set(outer))
                    self.operation("scf.yield", node)
                self.env = outer.copy()
                if node.orelse:
                    else_block = branch.regions[1].blocks.append()
                    with ir.InsertionPoint(else_block):
                        self.statements(node.orelse, branch_outer=set(outer))
                        self.operation("scf.yield", node)
                self.env = outer
            elif isinstance(node, ast.For):
                self.for_loop(node, branch_outer)
            elif isinstance(node, ast.Expr):
                if (
                    isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and position == 0
                    and top
                ):
                    continue  # source docstring
                if not isinstance(node.value, ast.Call) or self.expression(node.value) is not None:
                    self.fail(node, "expression statements must be kernel launches")
            elif isinstance(node, ast.Pass):
                continue
            else:
                self.fail(node, f"unsupported statement: {type(node).__name__}")


def lower_program(
    definition: Definition, signature, constants, *, verify=False, ckl_opt=None
) -> LoweredProgram:
    """
    Lower a CKL program definition to MLIR.
    Returns a LoweredProgram object containing the MLIR text, bound signature, kernel symbols, and a unique key.
    """
    bound = bind(definition, signature, constants)
    program_identity = identity(definition, bound)
    with ir.Context() as context, ir.Location.unknown():
        # CKL ops use the generic builder; their native verifier runs in ckl-opt/compiler.
        context.allow_unregistered_dialects = True
        module = ModuleEmitter()
        FunctionEmitter(module, definition, bound).build("program_" + program_identity[:24])
        # Stable printing keeps source locations separate from the semantic cache identity.
        module.ir.operation.verify()
        mlir = module.ir.operation.get_asm(enable_debug_info=True, use_local_scope=True) + "\n"
        semantic = module.ir.operation.get_asm(enable_debug_info=False, use_local_scope=True)
        key = hashlib.sha256((program_identity + semantic).encode()).hexdigest()
        result = LoweredProgram(mlir, bound, tuple(s for s, _ in module.kernels.values()), key)
    if verify:
        result.verify(ckl_opt=ckl_opt)
    return result

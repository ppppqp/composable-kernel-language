from __future__ import annotations

import contextvars
import inspect
from dataclasses import dataclass
from functools import update_wrapper
from typing import Any, Callable, Sequence

from .ir import Alternative, Distribution, IndexType, MemRefType, TileType, Type


_active_trace: contextvars.ContextVar[_Trace | None] = contextvars.ContextVar(
    "ckl_active_trace", default=None
)


def _mlir():
    try:
        from mlir import ir
        from mlir.dialects import func as func_dialect
        from .dialects import ckl as ckl_dialect
    except ImportError as error:
        raise RuntimeError(
            "CKL tracing requires the MLIR Python bindings from the LLVM build"
        ) from error
    return ir, func_dialect, ckl_dialect


@dataclass(frozen=True)
class TracedValue:
    value: Any
    type: Type


class NativeModule:
    """Owns an MLIR context and the module constructed in that context."""

    def __init__(self, context: Any, module: Any):
        self.context = context
        self.operation = module

    def __str__(self) -> str:
        return str(self.operation)

    def verify(self) -> bool:
        return self.operation.operation.verify()


class _Trace:
    def __init__(self, ir: Any, ckl_dialect: Any):
        self.ir = ir
        self.ckl_dialect = ckl_dialect

    def attribute(self, value: Distribution) -> Any:
        return self.ir.Attribute.parse(value.mlir())

    def type(self, value: Type) -> Any:
        return self.ir.Type.parse(value.mlir())


def _trace() -> _Trace:
    trace = _active_trace.get()
    if trace is None:
        raise RuntimeError("CKL operation used outside a @ckl.func trace")
    return trace


def load_tile(
    source: TracedValue,
    offsets: Sequence[TracedValue],
    *,
    distribution: Distribution,
) -> TracedValue:
    trace = _trace()
    if not isinstance(source.type, MemRefType):
        raise TypeError("load_tile source must be a memref")
    if len(offsets) != len(source.type.shape) or any(
        not isinstance(offset.type, IndexType) for offset in offsets
    ):
        raise TypeError("load_tile requires one index offset per memref dimension")
    result_type = TileType(source.type.element_type, distribution.tile)
    operation = trace.ckl_dialect.LoadTileOp(
        trace.type(result_type),
        source.value,
        [offset.value for offset in offsets],
        trace.attribute(distribution),
    )
    return TracedValue(operation.result, result_type)


def store_tile(
    value: TracedValue,
    target: TracedValue,
    offsets: Sequence[TracedValue],
    *,
    distribution: Distribution,
) -> None:
    trace = _trace()
    if not isinstance(value.type, TileType) or not isinstance(target.type, MemRefType):
        raise TypeError("store_tile requires a tile and a memref")
    if value.type.element_type != target.type.element_type or value.type.space != distribution.tile:
        raise TypeError("stored tile must match the target element type and distribution")
    if len(offsets) != len(target.type.shape) or any(
        not isinstance(offset.type, IndexType) for offset in offsets
    ):
        raise TypeError("store_tile requires one index offset per memref dimension")
    trace.ckl_dialect.StoreTileOp(
        value.value,
        target.value,
        [offset.value for offset in offsets],
        trace.attribute(distribution),
    )


class TaskDef:
    def __init__(self, function: Callable[..., Any], alternatives: Sequence[Alternative]):
        self.function = function
        self.signature = inspect.signature(function)
        self.alternatives = tuple(alternatives)
        if not self.alternatives:
            raise ValueError("a task must declare at least one alternative")
        if len({alternative.name for alternative in self.alternatives}) != len(self.alternatives):
            raise ValueError("task alternative names must be unique")
        self.input_types = tuple(
            _require_annotation(parameter.name, parameter.annotation)
            for parameter in self.signature.parameters.values()
        )
        self.result_types = _result_annotations(self.signature.return_annotation)
        update_wrapper(self, function)

    def emit(self, trace: _Trace) -> Any:
        attributes = []
        for alternative in self.alternatives:
            entries = {"name": trace.ir.StringAttr.get(alternative.name)}
            entries.update(
                (key, trace.ir.StringAttr.get(value))
                for key, value in alternative.properties.items()
            )
            attributes.append(trace.ir.DictAttr.get(entries))
        function_type = trace.ir.FunctionType.get(
            [trace.type(type_) for type_ in self.input_types],
            [trace.type(type_) for type_ in self.result_types],
        )
        return trace.ckl_dialect.TaskOp(
            self.function.__name__, trace.ir.TypeAttr.get(function_type), attributes
        )


def task(*, alternatives: Sequence[Alternative]) -> Callable[[Callable[..., Any]], TaskDef]:
    """Declare a logical CKL task contract from Python annotations."""

    def decorate(function: Callable[..., Any]) -> TaskDef:
        return TaskDef(function, alternatives)

    return decorate


def invoke(
    task: TaskDef,
    operands: Sequence[TracedValue],
    *,
    alternative: str | None = None,
) -> TracedValue | tuple[TracedValue, ...]:
    trace = _trace()
    if tuple(operand.type for operand in operands) != task.input_types:
        raise TypeError(f"operands do not match task @{task.function.__name__}")
    if alternative is not None and alternative not in {
        candidate.name for candidate in task.alternatives
    }:
        raise ValueError(f"task @{task.function.__name__} has no alternative {alternative!r}")
    operation = trace.ckl_dialect.InvokeOp(
        [trace.type(type_) for type_ in task.result_types],
        task.function.__name__,
        [operand.value for operand in operands],
        alternative=alternative,
    )
    results = tuple(
        TracedValue(result, type_) for result, type_ in zip(operation.results_, task.result_types)
    )
    return results[0] if len(results) == 1 else results


class Func:
    def __init__(self, function: Callable[..., Any], tasks: Sequence[TaskDef] = ()):
        self.function = function
        self.signature = inspect.signature(function)
        self.tasks = tuple(tasks)
        if len({task.function.__name__ for task in self.tasks}) != len(self.tasks):
            raise ValueError("function task dependencies must have unique names")
        update_wrapper(self, function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)

    def emit(self) -> NativeModule:
        ir, func_dialect, ckl_dialect = _mlir()
        parameters = tuple(self.signature.parameters.values())
        for parameter in parameters:
            if parameter.annotation is inspect.Parameter.empty or not hasattr(
                parameter.annotation, "mlir"
            ):
                raise TypeError(f"parameter {parameter.name!r} needs a CKL type annotation")
        result_types = _result_annotations(self.signature.return_annotation)

        context = ir.Context()
        context.allow_unregistered_dialects = True
        with context, ir.Location.unknown():
            module = ir.Module.create()
            trace = _Trace(ir, ckl_dialect)
            with ir.InsertionPoint(module.body):
                for task_definition in self.tasks:
                    task_definition.emit(trace)
                function = func_dialect.FuncOp(
                    self.function.__name__,
                    ([trace.type(parameter.annotation) for parameter in parameters],
                     [trace.type(result) for result in result_types]),
                )
                block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    token = _active_trace.set(trace)
                    try:
                        arguments = tuple(
                            TracedValue(argument, parameter.annotation)
                            for argument, parameter in zip(block.arguments, parameters)
                        )
                        returned = self.function(*arguments)
                    finally:
                        _active_trace.reset(token)
                    values = _normalize_results(returned, result_types)
                    func_dialect.ReturnOp([value.value for value in values])
        return NativeModule(context, module)


def _result_annotations(annotation: Any) -> tuple[Type, ...]:
    if annotation in {inspect.Signature.empty, None, type(None)}:
        return ()
    if isinstance(annotation, tuple):
        return annotation
    if hasattr(annotation, "mlir"):
        return (annotation,)
    raise TypeError("function return annotation must contain CKL types")


def _require_annotation(name: str, annotation: Any) -> Type:
    if annotation is inspect.Parameter.empty or not hasattr(annotation, "mlir"):
        raise TypeError(f"parameter {name!r} needs a CKL type annotation")
    return annotation


def _normalize_results(returned: Any, expected: tuple[Type, ...]) -> tuple[TracedValue, ...]:
    if not expected:
        if returned is not None:
            raise TypeError("void CKL function returned a value")
        return ()
    values = returned if isinstance(returned, tuple) else (returned,)
    if len(values) != len(expected) or any(
        not isinstance(value, TracedValue) or value.type != type_
        for value, type_ in zip(values, expected)
    ):
        raise TypeError("traced return values do not match the function annotation")
    return values


def func(
    function: Callable[..., Any] | None = None,
    *,
    tasks: Sequence[TaskDef] = (),
) -> Func | Callable[[Callable[..., Any]], Func]:
    """Trace a Python function into an MLIR `func.func` on `emit()`."""
    if function is not None:
        return Func(function, tasks)

    def decorate(value: Callable[..., Any]) -> Func:
        return Func(value, tasks)

    return decorate

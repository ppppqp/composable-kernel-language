"""Resolve restricted annotations and bind concrete shape specializations."""
from __future__ import annotations

import ast
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass

from ..types import (
    Buffer, BufferSpec, BufferType, ConstexprType, Dimension, NUMERIC_TYPES,
    ScalarType, bool_, constexpr, f32, f64, i32, i64, index,
)
from .source import Definition, Namespace, UnsupportedBinding


def resolve(node: ast.AST, definition: Definition) -> object:
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        if isinstance(node.value, str):
            return resolve(ast.parse(node.value, mode="eval").body, definition)
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in definition.bindings:
            raise definition.error(node, f"unknown captured name '{node.id}'")
        value = definition.bindings[node.id]
        if isinstance(value, UnsupportedBinding):
            raise definition.error(node, f"unsupported captured binding '{node.id}'")
        return value
    if isinstance(node, ast.Attribute):
        base = resolve(node.value, definition)
        if isinstance(base, Namespace) and node.attr in base.values:
            return base.values[node.attr]
    if isinstance(node, ast.Tuple):
        return tuple(resolve(item, definition) for item in node.elts)
    if isinstance(node, ast.Subscript):
        base = resolve(node.value, definition)
        arguments = resolve(node.slice, definition)
        try:
            if base is Buffer:
                return Buffer[arguments]
            if base is constexpr:
                return constexpr[arguments]
        except ValueError as error:
            raise definition.error(node, str(error)) from error
    raise definition.error(node, "unsupported annotation; use CKL types and shape declarations")


def constant(value: object, dtype: ScalarType, definition: Definition, node: ast.AST) -> object:
    if dtype == bool_:
        if type(value) is bool:
            return value
    elif dtype in (i32, i64, index):
        width = 32 if dtype == i32 else 64
        if type(value) is int and -(1 << (width - 1)) <= value < (1 << (width - 1)):
            return value
    elif dtype in (f32, f64) and type(value) in (int, float):
        try:
            converted = float(value)
            if dtype == f32:
                converted = struct.unpack("f", struct.pack("f", converted))[0]
            if math.isfinite(converted):
                return converted
        except (OverflowError, struct.error):
            pass
    display = repr(value) if type(value) in (int, float, bool, str, type(None)) else (
        f"<{type(value).__name__}>"
    )
    raise definition.error(node, f"constant {display} does not fit {dtype.name}")


@dataclass(frozen=True)
class BoundParameter:
    name: str
    dtype: ScalarType | BufferSpec
    value: object = None
    is_constant: bool = False


@dataclass(frozen=True)
class BoundSignature:
    parameters: tuple[BoundParameter, ...]
    dimensions: tuple[tuple[str, int], ...]

    @property
    def runtime(self) -> tuple[BoundParameter, ...]:
        return tuple(p for p in self.parameters if not p.is_constant)


def bind(definition: Definition, signature: Mapping, constants: Mapping) -> BoundSignature:
    if not isinstance(signature, Mapping) or not isinstance(constants, Mapping):
        raise definition.error(definition.tree, "signature and constants must be mappings")
    arguments = definition.tree.args
    if arguments.posonlyargs or arguments.vararg or arguments.kwarg or arguments.kwonlyargs or (
        arguments.defaults or arguments.kw_defaults
    ):
        raise definition.error(definition.tree, "CKL definitions require named positional parameters without defaults")
    names = {a.arg for a in arguments.args}
    extra = (set(signature) | set(constants)) - names
    if extra:
        raise definition.error(definition.tree, f"unknown specialization arguments: {sorted(extra)}")
    result = []
    dimensions: dict[str, int] = {}
    for argument in arguments.args:
        annotation = definition.annotations.get(argument.arg)
        if annotation is None or isinstance(annotation, str):
            if argument.annotation is None:
                raise definition.error(argument, "every CKL parameter requires a type annotation")
            annotation = resolve(argument.annotation, definition)
        if isinstance(annotation, ConstexprType):
            if argument.arg not in constants or argument.arg in signature:
                raise definition.error(argument, "constexpr parameters require a constants binding, not a runtime signature")
            result.append(BoundParameter(argument.arg, annotation.dtype,
                constant(constants[argument.arg], annotation.dtype, definition, argument), True))
            continue
        if argument.arg not in signature or argument.arg in constants:
            raise definition.error(argument, f"missing or misplaced runtime signature for '{argument.arg}'")
        actual = signature[argument.arg]
        if isinstance(annotation, BufferType):
            if not isinstance(actual, BufferSpec) or actual.dtype != annotation.dtype or (
                len(actual.shape) != len(annotation.shape)
            ):
                raise definition.error(argument, f"buffer type/rank mismatch for '{argument.arg}'")
            for expected, extent in zip(annotation.shape, actual.shape):
                if isinstance(expected, Dimension):
                    previous = dimensions.setdefault(expected.name, extent)
                    if previous != extent:
                        raise definition.error(argument, f"dimension {expected.name} conflicts: {previous} versus {extent}")
                elif expected != extent:
                    raise definition.error(argument, f"fixed extent {expected} does not match {extent}")
        elif annotation in NUMERIC_TYPES:
            if not isinstance(actual, ScalarType) or actual != annotation:
                raise definition.error(argument, f"scalar type mismatch for '{argument.arg}'")
        else:
            raise definition.error(argument, "runtime parameters require numeric scalars or typed buffers")
        result.append(BoundParameter(argument.arg, actual))
    if definition.tree.returns is not None:
        returned = definition.annotations.get("return")
        if isinstance(returned, str):
            returned = resolve(definition.tree.returns, definition)
        if returned is not None:
            raise definition.error(definition.tree.returns, "CKL entry points have no return values")
    return BoundSignature(tuple(result), tuple(sorted(dimensions.items())))

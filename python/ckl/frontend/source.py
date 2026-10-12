"""Capture source without running DSL bodies or evaluating annotation strings."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Mapping

from ..types import Buffer, BufferType, ConstexprType, Dimension, ScalarType, constexpr


class FrontendError(ValueError):
    """A source-associated frontend diagnostic."""


@dataclass(frozen=True)
class Intrinsic:
    name: str

    def __call__(self, *args, **kwargs):
        raise FrontendError(f"ckl.{self.name} is only valid inside a CKL source definition")


INTRINSICS = {name: Intrinsic(name) for name in (
    "global_id", "thread_id", "block_id", "block_dim", "grid_dim",
    "launch", "empty_like", "empty", "cdiv", "cast",
)}


@dataclass(frozen=True)
class Namespace:
    values: Mapping[str, object]


@dataclass(frozen=True)
class UnsupportedBinding:
    name: str


def immutable(value: object) -> bool:
    return type(value) in (int, float, bool, str, type(None)) or (
        type(value) is tuple and all(immutable(item) for item in value)
    )


def capture(value: object, name: str) -> object:
    if immutable(value) or isinstance(value, (
        ScalarType, BufferType, ConstexprType, Dimension, Definition, Intrinsic,
    )) or value is Buffer or value is constexpr:
        return value
    if isinstance(value, ModuleType) and value.__name__ == "ckl":
        values = {key: item for key, item in vars(value).items() if (
            isinstance(item, (ScalarType, Intrinsic)) or item is Buffer or item is constexpr
        )}
        return Namespace(MappingProxyType(values))
    return UnsupportedBinding(name)


def fingerprint(value: object) -> object:
    if isinstance(value, Definition):
        return ["definition", value.identity]
    if isinstance(value, Namespace):
        return ["namespace", sorted((k, fingerprint(v)) for k, v in value.values.items())]
    if value is Buffer or value is constexpr:
        return ["type_constructor", value.__name__]
    if isinstance(value, (ScalarType, Dimension, Intrinsic, UnsupportedBinding)):
        return [type(value).__name__, value.name]
    if isinstance(value, (BufferType, ConstexprType)):
        return [type(value).__name__, fingerprint(value.dtype),
                fingerprint(value.shape) if isinstance(value, BufferType) else None]
    if type(value) is tuple:
        return ["tuple", [fingerprint(v) for v in value]]
    return [type(value).__name__, repr(value)]


@dataclass(frozen=True)
class Definition:
    kind: str
    name: str
    tree: ast.FunctionDef
    source: str
    filename: str
    first_line: int
    indentation: int
    bindings: Mapping[str, object]
    annotations: Mapping[str, object]
    identity: str

    def error(self, node: ast.AST, message: str) -> FrontendError:
        line = self.first_line + getattr(node, "lineno", 1) - 1
        column = self.indentation + getattr(node, "col_offset", 0) + 1
        lines = self.source.splitlines()
        position = getattr(node, "lineno", 1) - 1
        excerpt = lines[position] if position < len(lines) else ""
        caret = " " * getattr(node, "col_offset", 0) + "^"
        return FrontendError(f"{self.filename}:{line}:{column}: {message}\n{excerpt}\n{caret}")

    def __call__(self, *args, **kwargs):
        hint = "use ckl.launch inside a program" if self.kind == "kernel" else "use .lower(signature=...)"
        raise self.error(self.tree, f"CKL {self.kind} bodies are not Python callables; {hint}")

    def lower(self, *, signature, constants=None, verify=False, ckl_opt=None):
        if self.kind != "program":
            raise self.error(self.tree, "lower a @ckl.program entry point")
        from .emitter import lower_program
        return lower_program(self, signature, {} if constants is None else constants,
                             verify=verify, ckl_opt=ckl_opt)


def decorate(function, kind: str) -> Definition:
    try:
        lines, first_line = inspect.getsourcelines(function)
        filename = inspect.getsourcefile(function) or "<unknown>"
    except (OSError, TypeError) as error:
        raise FrontendError("CKL definitions require retrievable source in a Python file") from error
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if node is None:
        raise FrontendError("CKL requires a synchronous function definition")
    environment = dict(function.__globals__)
    environment.update(inspect.getclosurevars(function).nonlocals)
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    for annotation in function.__annotations__.values():
        if isinstance(annotation, str):
            names.update(n.id for n in ast.walk(ast.parse(annotation, mode="eval"))
                         if isinstance(n, ast.Name))
    bindings = {name: capture(environment[name], name) for name in sorted(names)
                if name in environment}
    annotations = {name: capture(value, name) if not isinstance(value, str) else value
                   for name, value in function.__annotations__.items()}
    payload = ["ckl.frontend.v1", kind, ast.dump(node, include_attributes=False),
               [(name, fingerprint(value)) for name, value in bindings.items()],
               [(name, fingerprint(value)) for name, value in sorted(annotations.items())]]
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    indentation = len(lines[0]) - len(lines[0].lstrip())
    return Definition(kind, function.__name__, node, source, filename, first_line, indentation,
                      MappingProxyType(bindings), MappingProxyType(annotations), identity)


def kernel(function) -> Definition:
    return decorate(function, "kernel")


def program(function) -> Definition:
    return decorate(function, "program")

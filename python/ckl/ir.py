from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence


def _identifier(value: str, kind: str) -> str:
    if not value or not (value[0].isalpha() or value[0] == "_"):
        raise ValueError(f"invalid {kind} {value!r}")
    if not all(character.isalnum() or character in "_.$-" for character in value):
        raise ValueError(f"invalid {kind} {value!r}")
    return value


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Type(Protocol):
    def mlir(self) -> str: ...


@dataclass(frozen=True)
class Symbol:
    name: str

    def __post_init__(self) -> None:
        _identifier(self.name, "symbol")


Extent = int | Symbol


@dataclass(frozen=True)
class Space:
    axes: tuple[tuple[str, Extent], ...]

    def __init__(self, **axes: Extent):
        if not axes:
            raise ValueError("a space must have at least one axis")
        normalized: list[tuple[str, Extent]] = []
        for name, extent in axes.items():
            _identifier(name, "axis name")
            if isinstance(extent, int) and (isinstance(extent, bool) or extent <= 0):
                raise ValueError(f"axis {name!r} must have a positive extent")
            if not isinstance(extent, (int, Symbol)):
                raise TypeError(f"axis {name!r} extent must be int or Symbol")
            normalized.append((name, extent))
        object.__setattr__(self, "axes", tuple(normalized))

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(extent.name for _, extent in self.axes if isinstance(extent, Symbol))

    def mlir(self) -> str:
        axes = ", ".join(
            f"{name} = {extent if isinstance(extent, int) else f'symbol<{_quote(extent.name)}>'}"
            for name, extent in self.axes
        )
        return f"#ckl.space<[{axes}]>"


@dataclass(frozen=True)
class TileType:
    element_type: str
    space: Space

    def __post_init__(self) -> None:
        if not self.element_type:
            raise ValueError("tile element type cannot be empty")

    def mlir(self) -> str:
        return f"!ckl.tile<{self.element_type}, {self.space.mlir()}>"


@dataclass(frozen=True)
class IndexType:
    def mlir(self) -> str:
        return "index"


INDEX = IndexType()


@dataclass(frozen=True)
class Alternative:
    name: str
    properties: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.name, "alternative name")
        for key, value in self.properties.items():
            _identifier(key, "alternative property")
            if not isinstance(value, str):
                raise TypeError("initial alternative properties must be strings")

    def mlir(self) -> str:
        entries = [("name", self.name), *self.properties.items()]
        return "{" + ", ".join(f"{key} = {_quote(value)}" for key, value in entries) + "}"


@dataclass(frozen=True)
class Task:
    name: str
    inputs: tuple[Type, ...]
    outputs: tuple[Type, ...]
    alternatives: tuple[Alternative, ...]

    def mlir(self, indent: str = "") -> str:
        inputs = ", ".join(value.mlir() for value in self.inputs)
        outputs = _result_types(self.outputs)
        alternatives = ", ".join(value.mlir() for value in self.alternatives)
        return (
            f"{indent}ckl.task @{self.name} : ({inputs}) -> {outputs} "
            f"alternatives = [{alternatives}]"
        )


@dataclass(frozen=True)
class Value:
    name: str
    type: Type

    def __post_init__(self) -> None:
        _identifier(self.name, "SSA value")

    @property
    def ref(self) -> str:
        return f"%{self.name}"


def _result_types(types: Sequence[Type]) -> str:
    if len(types) == 1:
        return types[0].mlir()
    return "(" + ", ".join(value.mlir() for value in types) + ")"


class Function:
    def __init__(
        self,
        name: str,
        arguments: Sequence[tuple[str, Type]],
        results: Sequence[Type],
    ):
        self.name = _identifier(name, "function name")
        self.arguments = tuple(Value(name, type_) for name, type_ in arguments)
        if len({argument.name for argument in self.arguments}) != len(self.arguments):
            raise ValueError("function argument names must be unique")
        self.results = tuple(results)
        self._operations: list[str] = []
        self._values = {argument.name: argument for argument in self.arguments}
        self._terminated = False

    def arg(self, name: str) -> Value:
        try:
            return self._values[name]
        except KeyError as error:
            raise ValueError(f"unknown function value {name!r}") from error

    def _result(self, name: str, type_: Type) -> Value:
        if name in self._values:
            raise ValueError(f"SSA value {name!r} is already defined")
        value = Value(name, type_)
        self._values[name] = value
        return value

    def bind_symbols(self, name: str, tile: Value, bindings: Mapping[str, Value]) -> Value:
        if not isinstance(tile.type, TileType):
            raise TypeError("bind_symbols requires a tile value")
        expected = tile.type.space.symbols
        if tuple(bindings) != expected:
            raise ValueError(f"expected symbol bindings {expected}, got {tuple(bindings)}")
        for symbol, value in bindings.items():
            if value.type != INDEX:
                raise TypeError(f"binding for {symbol!r} must have index type")
        result = self._result(name, tile.type)
        operands = ", ".join(value.ref for value in bindings.values())
        symbols = ", ".join(_quote(symbol) for symbol in bindings)
        self._operations.append(
            f"{result.ref} = ckl.bind_symbols {tile.ref} [{operands}] symbols = [{symbols}]\n"
            f"    : {tile.type.mlir()} -> {result.type.mlir()}"
        )
        return result

    def invoke(
        self,
        name: str,
        task: Task,
        operands: Sequence[Value],
        *,
        alternative: str | None = None,
    ) -> Value:
        if len(task.outputs) != 1:
            raise NotImplementedError("the initial frontend supports one-result tasks")
        operand_types = tuple(value.type for value in operands)
        if operand_types != task.inputs:
            raise TypeError(f"task @{task.name} expects {task.inputs}, got {operand_types}")
        if alternative is not None and alternative not in {item.name for item in task.alternatives}:
            raise ValueError(f"task @{task.name} has no alternative {alternative!r}")
        result = self._result(name, task.outputs[0])
        operand_refs = ", ".join(value.ref for value in operands)
        pin = f" alternative = {_quote(alternative)}" if alternative is not None else ""
        input_types = ", ".join(value.mlir() for value in task.inputs)
        self._operations.append(
            f"{result.ref} = ckl.invoke @{task.name}({operand_refs}){pin}\n"
            f"    : ({input_types}) -> {result.type.mlir()}"
        )
        return result

    def return_(self, *values: Value) -> None:
        if self._terminated:
            raise ValueError("function already has a terminator")
        if tuple(value.type for value in values) != self.results:
            raise TypeError("return values do not match the function result types")
        if values:
            refs = ", ".join(value.ref for value in values)
            types = ", ".join(value.type.mlir() for value in values)
            self._operations.append(f"return {refs} : {types}")
        else:
            self._operations.append("return")
        self._terminated = True

    def mlir(self, indent: str = "") -> str:
        if not self._terminated:
            raise ValueError(f"function @{self.name} has no terminator")
        arguments = ", ".join(
            f"{argument.ref}: {argument.type.mlir()}" for argument in self.arguments
        )
        result = f" -> {_result_types(self.results)}" if self.results else ""
        body = "\n".join(
            "\n".join(f"{indent}  {line}" for line in operation.splitlines())
            for operation in self._operations
        )
        return f"{indent}func.func @{self.name}({arguments}){result} {{\n{body}\n{indent}}}"


class Module:
    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._functions: dict[str, Function] = {}

    def task(
        self,
        name: str,
        inputs: Iterable[Type],
        outputs: Iterable[Type],
        alternatives: Iterable[Alternative],
    ) -> Task:
        name = _identifier(name, "task name")
        if name in self._tasks:
            raise ValueError(f"task @{name} is already defined")
        task = Task(name, tuple(inputs), tuple(outputs), tuple(alternatives))
        if not task.alternatives:
            raise ValueError("a task must have at least one alternative")
        if len({item.name for item in task.alternatives}) != len(task.alternatives):
            raise ValueError("task alternative names must be unique")
        self._tasks[name] = task
        return task

    def function(
        self,
        name: str,
        arguments: Sequence[tuple[str, Type]],
        results: Sequence[Type] = (),
    ) -> Function:
        name = _identifier(name, "function name")
        if name in self._functions:
            raise ValueError(f"function @{name} is already defined")
        function = Function(name, arguments, results)
        self._functions[name] = function
        return function

    def mlir(self) -> str:
        operations = [task.mlir("  ") for task in self._tasks.values()]
        operations.extend(function.mlir("  ") for function in self._functions.values())
        return "module {\n" + "\n\n".join(operations) + "\n}\n"

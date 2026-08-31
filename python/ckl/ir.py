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

    def product(self, other: Space) -> Space:
        names = {name for name, _ in self.axes}
        if names.intersection(name for name, _ in other.axes):
            raise ValueError("product-space axis names must be unique")
        result = object.__new__(Space)
        object.__setattr__(result, "axes", self.axes + other.axes)
        return result


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
class MemRefType:
    shape: tuple[int | None, ...]
    element_type: str

    def __init__(self, shape: Iterable[int | None], element_type: str):
        normalized = tuple(shape)
        for extent in normalized:
            if extent is not None and (
                not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0
            ):
                raise ValueError("memref extents must be positive integers or None")
        if not element_type:
            raise ValueError("memref element type cannot be empty")
        object.__setattr__(self, "shape", normalized)
        object.__setattr__(self, "element_type", element_type)

    def mlir(self) -> str:
        dimensions = "x".join("?" if extent is None else str(extent) for extent in self.shape)
        separator = "x" if dimensions else ""
        return f"memref<{dimensions}{separator}{self.element_type}>"


@dataclass(frozen=True)
class IndexExpr:
    operation: str
    operands: tuple[IndexExpr | int, ...]

    def __post_init__(self) -> None:
        if self.operation in {"dim", "const"}:
            if len(self.operands) != 1 or not isinstance(self.operands[0], int):
                raise ValueError(f"{self.operation} expects one integer operand")
            if self.operation == "dim" and self.operands[0] < 0:
                raise ValueError("dimension position must be non-negative")
            return
        if self.operation not in {"add", "mul", "floordiv", "mod", "xor"}:
            raise ValueError(f"unsupported index expression {self.operation!r}")
        if len(self.operands) != 2 or not all(
            isinstance(operand, IndexExpr) for operand in self.operands
        ):
            raise ValueError(f"{self.operation} expects two index expressions")
        if self.operation in {"floordiv", "mod"}:
            rhs = self.operands[1]
            if rhs.operation != "const" or rhs.operands[0] <= 0:
                raise ValueError(f"{self.operation} divisor must be a positive constant")

    def __add__(self, other: IndexExpr | int) -> IndexExpr:
        return IndexExpr("add", (self, _expr(other)))

    def __radd__(self, other: IndexExpr | int) -> IndexExpr:
        return _expr(other) + self

    def __mul__(self, other: IndexExpr | int) -> IndexExpr:
        return IndexExpr("mul", (self, _expr(other)))

    def __rmul__(self, other: IndexExpr | int) -> IndexExpr:
        return _expr(other) * self

    def __floordiv__(self, other: IndexExpr | int) -> IndexExpr:
        return IndexExpr("floordiv", (self, _expr(other)))

    def __mod__(self, other: IndexExpr | int) -> IndexExpr:
        return IndexExpr("mod", (self, _expr(other)))

    def __xor__(self, other: IndexExpr | int) -> IndexExpr:
        return IndexExpr("xor", (self, _expr(other)))

    def dimensions(self) -> tuple[int, ...]:
        if self.operation == "dim":
            return (self.operands[0],)  # type: ignore[return-value]
        result: set[int] = set()
        for operand in self.operands:
            if isinstance(operand, IndexExpr):
                result.update(operand.dimensions())
        return tuple(sorted(result))

    def mlir(self) -> str:
        if self.operation in {"dim", "const"}:
            return f"{self.operation}({self.operands[0]})"
        if self.operation in {"floordiv", "mod"}:
            rhs = self.operands[1]
            return f"{self.operation}({_expr(self.operands[0]).mlir()}, {rhs.operands[0]})"
        return f"{self.operation}({', '.join(_expr(value).mlir() for value in self.operands)})"


def _expr(value: IndexExpr | int) -> IndexExpr:
    if isinstance(value, IndexExpr):
        return value
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("index expressions accept integer constants")
    return IndexExpr("const", (value,))


def dim(position: int) -> IndexExpr:
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise ValueError("dimension position must be a non-negative integer")
    return IndexExpr("dim", (position,))


def const(value: int) -> IndexExpr:
    return _expr(value)


@dataclass(frozen=True)
class IndexMap:
    domain: Space
    codomain: Space
    results: tuple[IndexExpr, ...]

    def __init__(self, domain: Space, codomain: Space, results: Iterable[IndexExpr | int]):
        normalized = tuple(_expr(result) for result in results)
        if len(normalized) != len(codomain.axes):
            raise ValueError("index-map result count must match codomain rank")
        rank = len(domain.axes)
        if any(position >= rank for result in normalized for position in result.dimensions()):
            raise ValueError("index-map expression references a dimension outside its domain")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "codomain", codomain)
        object.__setattr__(self, "results", normalized)

    def mlir(self) -> str:
        results = ", ".join(result.mlir() for result in self.results)
        return (
            f"#ckl.index_map<domain = {self.domain.mlir()}, codomain = {self.codomain.mlir()}, "
            f"results = [{results}], predicate = true>"
        )


@dataclass(frozen=True)
class Distribution:
    executors: Space
    local: Space
    tile: Space
    ownership: IndexMap
    local_storage: IndexMap
    scope: str = "subgroup"
    replicated: bool = False

    def __post_init__(self) -> None:
        if self.scope not in {"subgroup", "workgroup", "cluster", "grid"}:
            raise ValueError(f"unsupported execution scope {self.scope!r}")
        if self.ownership.domain != self.executors.product(self.local):
            raise ValueError("ownership domain must be executors × local")
        if self.ownership.codomain != self.tile:
            raise ValueError("ownership codomain must be the logical tile")
        if self.local_storage.domain != self.local:
            raise ValueError("local-storage domain must be the local space")
        if len(self.local_storage.codomain.axes) != 1:
            raise ValueError("local-storage codomain must have rank one")

    def mlir(self) -> str:
        replicated = "true" if self.replicated else "false"
        return (
            f"#ckl.distribution<executors = {self.executors.mlir()}, "
            f"local = {self.local.mlir()}, tile = {self.tile.mlir()}, "
            f"ownership = {self.ownership.mlir()}, "
            f"local_storage = {self.local_storage.mlir()}, "
            f"scope = {self.scope}, replicated = {replicated}>"
        )


@dataclass(frozen=True)
class Alternative:
    name: str
    properties: Mapping[str, str | int | Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.name, "alternative name")
        for key, value in self.properties.items():
            _identifier(key, "alternative property")
            if not isinstance(value, (str, int, list, tuple)) or isinstance(value, bool):
                raise TypeError("alternative properties must be strings, integers, or string lists")
            if isinstance(value, (list, tuple)) and not all(
                isinstance(item, str) for item in value
            ):
                raise TypeError("alternative property lists must contain strings")

    def mlir(self) -> str:
        entries = [("name", self.name), *self.properties.items()]
        def format_value(value: str | int | Sequence[str]) -> str:
            if isinstance(value, str):
                return _quote(value)
            if isinstance(value, int):
                return f"{value} : i64"
            return "[" + ", ".join(_quote(item) for item in value) + "]"

        return "{" + ", ".join(f"{key} = {format_value(value)}" for key, value in entries) + "}"


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

    def load_tile(
        self,
        name: str,
        source: Value,
        offsets: Sequence[Value],
        distribution: Distribution,
    ) -> Value:
        if not isinstance(source.type, MemRefType):
            raise TypeError("load_tile source must be a memref")
        self._validate_memory_boundary(source.type, offsets, distribution)
        result = self._result(name, TileType(source.type.element_type, distribution.tile))
        offset_refs = ", ".join(offset.ref for offset in offsets)
        self._operations.append(
            f"{result.ref} = ckl.load_tile {source.ref}[{offset_refs}] "
            f"distribution = {distribution.mlir()} attributes {{}} "
            f": {source.type.mlir()} -> {result.type.mlir()}"
        )
        return result

    def store_tile(
        self,
        value: Value,
        target: Value,
        offsets: Sequence[Value],
        distribution: Distribution,
    ) -> None:
        if not isinstance(value.type, TileType):
            raise TypeError("store_tile value must be a tile")
        if not isinstance(target.type, MemRefType):
            raise TypeError("store_tile target must be a memref")
        self._validate_memory_boundary(target.type, offsets, distribution)
        if (
            value.type.element_type != target.type.element_type
            or value.type.space != distribution.tile
        ):
            raise TypeError("stored tile type must match the memref element and distribution tile")
        offset_refs = ", ".join(offset.ref for offset in offsets)
        self._operations.append(
            f"ckl.store_tile {value.ref}, {target.ref}[{offset_refs}] "
            f"distribution = {distribution.mlir()} attributes {{}} "
            f": {value.type.mlir()}, {target.type.mlir()}"
        )

    @staticmethod
    def _validate_memory_boundary(
        memref: MemRefType, offsets: Sequence[Value], distribution: Distribution
    ) -> None:
        if len(offsets) != len(memref.shape):
            raise ValueError("one tile offset is required per memref dimension")
        if any(offset.type != INDEX for offset in offsets):
            raise TypeError("tile offsets must have index type")
        if len(memref.shape) != len(distribution.tile.axes):
            raise ValueError("memref rank must match the distribution tile rank")
        for memref_extent, (_, tile_extent) in zip(memref.shape, distribution.tile.axes):
            if (
                isinstance(tile_extent, int)
                and memref_extent is not None
                and memref_extent < tile_extent
            ):
                raise ValueError("static memref extent cannot be smaller than the tile extent")

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

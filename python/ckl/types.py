"""Immutable source types and storage-free specialization descriptors."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarType:
    name: str

    @property
    def mlir(self) -> str:
        return "i1" if self.name == "bool" else self.name


f32 = ScalarType("f32")
f64 = ScalarType("f64")
i32 = ScalarType("i32")
i64 = ScalarType("i64")
index = ScalarType("index")
bool_ = ScalarType("bool")
NUMERIC_TYPES = (f32, f64, i32, i64)
SCALAR_TYPES = (*NUMERIC_TYPES, index, bool_)


@dataclass(frozen=True)
class Dimension:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("dimension names must be Python identifiers")


def dim(name: str) -> Dimension:
    return Dimension(name)


@dataclass(frozen=True)
class BufferType:
    dtype: ScalarType
    shape: tuple[int | Dimension, ...]

    def __post_init__(self) -> None:
        if self.dtype not in NUMERIC_TYPES:
            raise ValueError("buffer elements must be f32, f64, i32, or i64")
        if not isinstance(self.shape, tuple) or any(
            not isinstance(d, Dimension) and (type(d) is not int or not 0 < d < (1 << 63))
            for d in self.shape
        ):
            raise ValueError("buffer shapes must be tuples of positive extents within i64 or dimensions")


class Buffer:
    def __class_getitem__(cls, arguments: tuple) -> BufferType:
        if not isinstance(arguments, tuple) or len(arguments) != 2:
            raise ValueError("use Buffer[element_type, shape_tuple]")
        return BufferType(*arguments)


@dataclass(frozen=True)
class ConstexprType:
    dtype: ScalarType

    def __post_init__(self) -> None:
        if self.dtype not in SCALAR_TYPES:
            raise ValueError("constexpr requires a CKL scalar type")


class constexpr:
    def __class_getitem__(cls, dtype: ScalarType) -> ConstexprType:
        return ConstexprType(dtype)


@dataclass(frozen=True)
class BufferSpec:
    shape: tuple[int, ...]
    dtype: ScalarType
    layout: str = "identity"

    def __post_init__(self) -> None:
        BufferType(self.dtype, self.shape)
        if any(type(d) is not int for d in self.shape):
            raise ValueError("BufferSpec requires concrete integer extents")
        if self.layout != "identity":
            raise ValueError("only contiguous identity-layout buffers are supported")

    @property
    def mlir(self) -> str:
        return "memref<" + "".join(f"{d}x" for d in self.shape) + self.dtype.mlir + ">"

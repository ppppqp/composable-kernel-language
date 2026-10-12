"""Lower a two-kernel program without a GPU or MLIR Python bindings."""
from __future__ import annotations

import ckl

N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]


@ckl.kernel
def update(src: Vector, dst: Vector, bias: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < src.shape[0]:
        dst[i] = src[i] + bias


@ckl.program
def step(src: Vector, dst: Vector, bias: ckl.f32) -> None:
    temporary = ckl.empty_like(src)
    grid = (ckl.cdiv(src.shape[0], 256),)
    block = (256,)
    ckl.launch(update, src, temporary, bias, grid=grid, block=block)
    ckl.launch(update, temporary, dst, bias, grid=grid, block=block)


def lower():
    return step.lower(signature={
        "src": ckl.BufferSpec(shape=(257,), dtype=ckl.f32),
        "dst": ckl.BufferSpec(shape=(257,), dtype=ckl.f32),
        "bias": ckl.f32,
    })


if __name__ == "__main__":
    print(lower().mlir, end="")

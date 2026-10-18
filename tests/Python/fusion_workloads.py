"""CKL source fixtures for fusion, branching, and stencil regression tests."""
import ckl

N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]


@ckl.kernel
def affine(src: Vector, dst: Vector, scale: ckl.f32, bias: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < N:
        dst[i] = src[i] * scale + bias


@ckl.kernel
def add(a: Vector, b: Vector, dst: Vector) -> None:
    i = ckl.global_id(0)
    if i < N:
        dst[i] = a[i] + b[i]


@ckl.kernel
def fused_six(src: Vector, dst: Vector) -> None:
    i = ckl.global_id(0)
    if i < N:
        value = src[i] * 0.875 + 0.03125
        value = value * 0.875 + 0.03125
        value = value * 0.875 + 0.03125
        value = value * 0.875 + 0.03125
        value = value * 0.875 + 0.03125
        dst[i] = value * 0.875 + 0.03125


@ckl.kernel
def stencil(src: Vector, dst: Vector) -> None:
    i = ckl.global_id(0)
    if i < N:
        if i == 0:
            dst[i] = src[i]
        if i == N - 1:
            dst[i] = src[i]
        if i > 0:
            if i < N - 1:
                dst[i] = src[i - 1] * 0.25 + src[i] * 0.5 + src[i + 1] * 0.25


@ckl.program
def single_affine(src: Vector, dst: Vector) -> None:
    ckl.launch(affine, src, dst, 0.875, 0.03125, grid=(ckl.cdiv(N, 256),), block=(256,))


@ckl.program
def chain_six(src: Vector, dst: Vector) -> None:
    t0 = ckl.empty_like(src)
    ckl.launch(affine, src, t0, 0.875, 0.03125, grid=(ckl.cdiv(N, 256),), block=(256,))
    t1 = ckl.empty_like(src)
    ckl.launch(affine, t0, t1, 0.875, 0.03125, grid=(ckl.cdiv(N, 256),), block=(256,))
    t2 = ckl.empty_like(src)
    ckl.launch(affine, t1, t2, 0.875, 0.03125, grid=(ckl.cdiv(N, 256),), block=(256,))
    t3 = ckl.empty_like(src)
    ckl.launch(affine, t2, t3, 0.875, 0.03125, grid=(ckl.cdiv(N, 256),), block=(256,))
    t4 = ckl.empty_like(src)
    ckl.launch(affine, t3, t4, 0.875, 0.03125, grid=(ckl.cdiv(N, 256),), block=(256,))
    ckl.launch(affine, t4, dst, 0.875, 0.03125, grid=(ckl.cdiv(N, 256),), block=(256,))


@ckl.program
def fused_chain_six(src: Vector, dst: Vector) -> None:
    ckl.launch(fused_six, src, dst, grid=(ckl.cdiv(N, 256),), block=(256,))


@ckl.program
def fork_join(src: Vector, other: Vector, dst: Vector) -> None:
    a = ckl.empty_like(src)
    b = ckl.empty_like(src)
    ckl.launch(affine, src, a, 0.5, 0.125, grid=(ckl.cdiv(N, 256),), block=(256,))
    ckl.launch(affine, other, b, -0.25, 0.5, grid=(ckl.cdiv(N, 256),), block=(256,))
    c = ckl.empty_like(src)
    ckl.launch(add, a, b, c, grid=(ckl.cdiv(N, 256),), block=(256,))
    d = ckl.empty_like(src)
    e = ckl.empty_like(src)
    ckl.launch(affine, c, d, 0.75, -0.0625, grid=(ckl.cdiv(N, 256),), block=(256,))
    ckl.launch(affine, c, e, 1.25, 0.03125, grid=(ckl.cdiv(N, 256),), block=(256,))
    ckl.launch(add, d, e, dst, grid=(ckl.cdiv(N, 256),), block=(256,))


@ckl.program
def stencil_two(src: Vector, dst: Vector) -> None:
    temporary = ckl.empty_like(src)
    ckl.launch(stencil, src, temporary, grid=(ckl.cdiv(N, 256),), block=(256,))
    ckl.launch(stencil, temporary, dst, grid=(ckl.cdiv(N, 256),), block=(256,))


CASES = {definition.name: definition for definition in
         (single_affine, chain_six, fused_chain_six, fork_join, stencil_two)}

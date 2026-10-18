"""Dense established-state SGD, following PyTorch's four-operation update."""
import ckl

N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]


@ckl.kernel
def adjusted_gradient(p: Vector, g: Vector, adjusted: Vector, wd: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < N:
        adjusted[i] = g[i] + wd * p[i]


@ckl.kernel
def scale_velocity(v: Vector, momentum: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < N:
        v[i] = v[i] * momentum


@ckl.kernel
def accumulate_velocity(v: Vector, adjusted: Vector) -> None:
    i = ckl.global_id(0)
    if i < N:
        v[i] = v[i] + adjusted[i]


@ckl.kernel
def update_parameter(p: Vector, v: Vector, lr: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < N:
        p[i] = p[i] + v[i] * -lr


@ckl.kernel
def fused_update(p: Vector, g: Vector, v: Vector,
                 wd: ckl.f32, momentum: ckl.f32, lr: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < N:
        adjusted = g[i] + wd * p[i]
        velocity = v[i] * momentum + adjusted
        parameter = p[i] + velocity * -lr
        v[i] = velocity
        p[i] = parameter


@ckl.program
def staged(p: Vector, g: Vector, v: Vector,
           wd: ckl.f32, momentum: ckl.f32, lr: ckl.f32) -> None:
    adjusted = ckl.empty_like(g)
    ckl.launch(adjusted_gradient, p, g, adjusted, wd,
               grid=(ckl.cdiv(N, 256),), block=(256,))
    ckl.launch(scale_velocity, v, momentum, grid=(ckl.cdiv(N, 256),), block=(256,))
    ckl.launch(accumulate_velocity, v, adjusted, grid=(ckl.cdiv(N, 256),), block=(256,))
    ckl.launch(update_parameter, p, v, lr, grid=(ckl.cdiv(N, 256),), block=(256,))


@ckl.program
def manual(p: Vector, g: Vector, v: Vector,
           wd: ckl.f32, momentum: ckl.f32, lr: ckl.f32) -> None:
    ckl.launch(fused_update, p, g, v, wd, momentum, lr,
               grid=(ckl.cdiv(N, 256),), block=(256,))

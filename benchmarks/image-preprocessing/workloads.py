"""Float RGB NCHW normalization component; H and W are flattened into P."""
import ckl

B = ckl.dim("B")
P = ckl.dim("P")
Image = ckl.Buffer[ckl.f32, (B, 3, P)]
Channels = ckl.Buffer[ckl.f32, (3,)]


@ckl.kernel
def copy_image(src: Image, dst: Image) -> None:
    i = ckl.global_id(0)
    c = ckl.global_id(1)
    b = ckl.global_id(2)
    if i < P:
        dst[b, c, i] = src[b, c, i]


@ckl.kernel
def subtract_mean(dst: Image, mean: Channels) -> None:
    i = ckl.global_id(0)
    c = ckl.global_id(1)
    b = ckl.global_id(2)
    if i < P:
        dst[b, c, i] = dst[b, c, i] - mean[c]


@ckl.kernel
def divide_std(dst: Image, std: Channels) -> None:
    i = ckl.global_id(0)
    c = ckl.global_id(1)
    b = ckl.global_id(2)
    if i < P:
        dst[b, c, i] = dst[b, c, i] / std[c]


@ckl.kernel
def subtract_out(src: Image, dst: Image, mean: Channels) -> None:
    i = ckl.global_id(0)
    c = ckl.global_id(1)
    b = ckl.global_id(2)
    if i < P:
        dst[b, c, i] = src[b, c, i] - mean[c]


@ckl.kernel
def normalize(src: Image, dst: Image, mean: Channels, std: Channels) -> None:
    i = ckl.global_id(0)
    c = ckl.global_id(1)
    b = ckl.global_id(2)
    if i < P:
        dst[b, c, i] = (src[b, c, i] - mean[c]) / std[c]


@ckl.program
def staged(src: Image, dst: Image, mean: Channels, std: Channels) -> None:
    grid = (ckl.cdiv(P, 256), 3, B)
    ckl.launch(copy_image, src, dst, grid=grid, block=(256,))
    ckl.launch(subtract_mean, dst, mean, grid=grid, block=(256,))
    ckl.launch(divide_std, dst, std, grid=grid, block=(256,))


@ckl.program
def copy_elided(src: Image, dst: Image, mean: Channels, std: Channels) -> None:
    grid = (ckl.cdiv(P, 256), 3, B)
    ckl.launch(subtract_out, src, dst, mean, grid=grid, block=(256,))
    ckl.launch(divide_std, dst, std, grid=grid, block=(256,))


@ckl.program
def manual(src: Image, dst: Image, mean: Channels, std: Channels) -> None:
    ckl.launch(normalize, src, dst, mean, std,
               grid=(ckl.cdiv(P, 256), 3, B), block=(256,))

import os
from pathlib import Path

from ckl import (
    Alternative,
    CompilerOptions,
    Distribution,
    IndexMap,
    MemRefType,
    NVIDIATarget,
    Space,
    TileType,
    constant_index,
    dim,
    invoke,
    jit,
    load_tile,
    store_tile,
    task,
)


lane = Space(lane=32)

lhs_local = Space(kHalf=2, rowHalf=2, pair=2)
lhs_tile_space = Space(m=16, k=16)
lhs_distribution = Distribution(
    lane,
    lhs_local,
    lhs_tile_space,
    IndexMap(
        lane.product(lhs_local),
        lhs_tile_space,
        [dim(0) // 4 + dim(2) * 8, (dim(0) % 4) * 2 + dim(3) + dim(1) * 8],
    ),
    IndexMap(lhs_local, Space(address=8), [0 + dim(0) * 4 + dim(1) * 2 + dim(2) * 1]),
)

rhs_local = Space(kHalf=2, pair=2)
rhs_tile_space = Space(k=16, n=8)
rhs_distribution = Distribution(
    lane,
    rhs_local,
    rhs_tile_space,
    IndexMap(
        lane.product(rhs_local),
        rhs_tile_space,
        [(dim(0) % 4) * 2 + dim(2) + dim(1) * 8, dim(0) // 4],
    ),
    IndexMap(rhs_local, Space(address=4), [0 + dim(0) * 2 + dim(1) * 1]),
)

acc_local = Space(rowHalf=2, pair=2)
acc_tile_space = Space(m=16, n=8)
acc_distribution = Distribution(
    lane,
    acc_local,
    acc_tile_space,
    IndexMap(
        lane.product(acc_local),
        acc_tile_space,
        [dim(0) // 4 + dim(1) * 8, (dim(0) % 4) * 2 + dim(2)],
    ),
    IndexMap(acc_local, Space(address=4), [0 + dim(0) * 2 + dim(1) * 1]),
)

lhs_tile = TileType("f16", lhs_tile_space)
rhs_tile = TileType("f16", rhs_tile_space)
acc_tile = TileType("f32", acc_tile_space)


@task(
    alternatives=[
        Alternative(
            "mma-sync-m16n8k16-f16-f32",
            {
                "implementation_id": (
                    "nvidia.mma.sync.m16n8k16.row.col.f32.f16.f16.f32"
                ),
                "estimated_execution_cost": 1,
            },
        ),
        Alternative(
            "scalar-fallback",
            {
                "implementation_id": "example.scalar.m16n8k16.f16-f32",
                "estimated_execution_cost": 100,
            },
        ),
    ]
)
def mma(lhs: lhs_tile, rhs: rhs_tile, acc: acc_tile) -> acc_tile:
    ...


lhs_memref = MemRefType([16, 16], "f16")
rhs_memref = MemRefType([16, 8], "f16")
acc_memref = MemRefType([16, 8], "f32")


@jit(
    tasks=[mma],
    passes=(
        "--ckl-select-alternatives",
        "--ckl-nvidia-materialize-mma-sync",
        "--ckl-nvidia-lower-mma-sync",
        "--ckl-nvidia-lower-fragment-io",
    ),
    device=True,
    module_name="kernels",
    block_size=(32, 1, 1),
)
def kernel(a: lhs_memref, b: rhs_memref, c: acc_memref, d: acc_memref) -> None:
    zero = constant_index(0)
    a_tile = load_tile(a, [zero, zero], distribution=lhs_distribution)
    b_tile = load_tile(b, [zero, zero], distribution=rhs_distribution)
    c_tile = load_tile(c, [zero, zero], distribution=acc_distribution)
    result = invoke(mma, [a_tile, b_tile, c_tile])
    store_tile(result, d, [zero, zero], distribution=acc_distribution)


target = NVIDIATarget(
    chip=os.environ.get("CKL_CUDA_ARCH", "sm_120"),
    features=os.environ.get("CKL_CUDA_PTX_FEATURE", "+ptx87"),
    toolkit_root=Path(os.environ["CUDA_HOME"]),
)
compiled = kernel.compile(CompilerOptions(target=target))
print(
    "selected mma-sync-m16n8k16-f16-f32; "
    f"generated {len(compiled.gpu_objects[0].data)} CUBIN bytes"
)

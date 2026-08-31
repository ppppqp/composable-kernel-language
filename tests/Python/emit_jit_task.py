from ckl import Alternative, Space, TileType, invoke, jit, task


tile_type = TileType("f32", Space(m=4))


@task(alternatives=[Alternative("direct", {"target": "generic"})])
def copy_task(tile: tile_type) -> tile_type:
    ...


@jit(tasks=[copy_task])
def kernel(tile: tile_type) -> tile_type:
    return invoke(copy_task, [tile], alternative="direct")


first = kernel.compile()
second = kernel.compile()
assert first is second
assert first.cache_key == second.cache_key
assert first.command[0].endswith("ckl-opt")
print(first.mlir, end="")

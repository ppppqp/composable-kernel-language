from ckl import (
    Alternative,
    Distribution,
    IndexMap,
    MemRefType,
    Port,
    Space,
    TileType,
    constant_index,
    dim,
    func,
    invoke,
    jit,
    load_tile,
    store_tile,
    task,
)


executors = Space(lane=2)
local = Space(value=2)
tile_space = Space(x=4)
direct = Distribution(
    executors,
    local,
    tile_space,
    IndexMap(executors.product(local), tile_space, [dim(0) * 2 + dim(1)]),
    IndexMap(local, Space(address=2), [dim(0)]),
)
permuted = Distribution(
    executors,
    local,
    tile_space,
    IndexMap(executors.product(local), tile_space, [dim(0) * 2 + dim(1)]),
    IndexMap(local, Space(address=2), [1 + dim(0) * -1]),
)
tile_type = TileType("f32", tile_space)
memory_type = MemRefType([4], "f32")


@func
def direct_impl(tile: tile_type) -> tile_type:
    constant_index(11)
    return tile


@func
def permuted_impl(tile: tile_type) -> tile_type:
    constant_index(22)
    return tile


@task(
    alternatives=[
        Alternative(
            "direct",
            {"implementation_id": "python.direct", "estimated_execution_cost": 0},
            inputs=(Port("input", direct),),
            outputs=(Port("output", direct),),
            implementation=direct_impl,
        ),
        Alternative(
            "permuted",
            {"implementation_id": "python.permuted", "estimated_execution_cost": 0},
            inputs=(Port("input", permuted),),
            outputs=(Port("output", permuted),),
            implementation=permuted_impl,
        ),
    ]
)
def layout_choice(tile: tile_type) -> tile_type:
    ...


passes = ("--ckl-select-alternatives", "--ckl-materialize-selected-alternatives")


@jit(tasks=[layout_choice], passes=passes)
def direct_boundary(source: memory_type, target: memory_type) -> None:
    zero = constant_index(0)
    tile = load_tile(source, [zero], distribution=direct)
    result = invoke(layout_choice, [tile])
    store_tile(result, target, [zero], distribution=direct)


@jit(tasks=[layout_choice], passes=passes)
def permuted_boundary(source: memory_type, target: memory_type) -> None:
    zero = constant_index(0)
    tile = load_tile(source, [zero], distribution=permuted)
    result = invoke(layout_choice, [tile])
    store_tile(result, target, [zero], distribution=permuted)


direct_ir = direct_boundary.compile().mlir
permuted_ir = permuted_boundary.compile().mlir
assert 'alternative = "direct"' in direct_ir
assert 'implementation = "direct_impl"' in direct_ir
assert "arith.constant 11 : index" in direct_ir
assert "arith.constant 22 : index" not in direct_ir
assert 'alternative = "permuted"' in permuted_ir
assert 'implementation = "permuted_impl"' in permuted_ir
assert "arith.constant 22 : index" in permuted_ir
assert "arith.constant 11 : index" not in permuted_ir
assert "ckl.invoke" not in direct_ir + permuted_ir
assert "func.call" not in direct_ir + permuted_ir
assert "func.func private @direct_impl" not in direct_ir
assert "func.func private @permuted_impl" not in permuted_ir


@jit(tasks=[layout_choice], passes=passes, device=True, module_name="callable_kernels")
def device_boundary(source: memory_type, target: memory_type) -> None:
    zero = constant_index(0)
    tile = load_tile(source, [zero], distribution=direct)
    result = invoke(layout_choice, [tile])
    store_tile(result, target, [zero], distribution=direct)


device_ir = device_boundary.compile().mlir
assert "gpu.func @device_boundary" in device_ir
assert 'implementation = "direct_impl"' in device_ir
assert "arith.constant 11 : index" in device_ir
assert "arith.constant 22 : index" not in device_ir
assert "ckl.invoke" not in device_ir
assert "func.call" not in device_ir
assert "func.func" not in device_ir
print(direct_ir)
print(permuted_ir)
print(device_ir)

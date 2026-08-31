from ckl import Alternative, Space, TileType, func, invoke, task


tile_type = TileType("f32", Space(m=4))


@task(alternatives=[Alternative("direct", {"target": "generic"})])
def copy_task(tile: tile_type) -> tile_type:
    ...


@func(tasks=[copy_task])
def kernel(tile: tile_type) -> tile_type:
    return invoke(copy_task, [tile], alternative="direct")


module = kernel.emit()
function_op = list(module.operation.body.operations)[1]
invoke_op = list(function_op.regions[0].blocks[0].operations)[0]
assert type(list(module.operation.body.operations)[0].opview).__name__ == "TaskOp"
assert type(invoke_op.opview).__name__ == "InvokeOp"
print(module)

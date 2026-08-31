from ckl import IndexType, MemRefType, jit, load_tile, store_tile
from test_frontend import direct_distribution


source_type = MemRefType([8], "f32")
target_type = MemRefType([8], "f32")
index_type = IndexType()
distribution = direct_distribution()


@jit(device=True, module_name="copy_kernels", block_size=(4, 1, 1))
def copy(source: source_type, target: target_type, base: index_type) -> None:
    tile = load_tile(source, [base], distribution=distribution)
    store_tile(tile, target, [base], distribution=distribution)


module = copy.emit()
assert module.verify()
gpu_module = list(module.operation.body.operations)[0]
assert gpu_module.name == "gpu.module"
gpu_function = list(gpu_module.regions[0].blocks[0].operations)[0]
assert gpu_function.operation.name == "gpu.func"
assert "gpu.kernel" in gpu_function.attributes
print(module)

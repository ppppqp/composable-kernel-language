from ckl import IndexType, MemRefType, func, load_tile, store_tile
from test_frontend import direct_distribution


source_type = MemRefType([8], "f32")
target_type = MemRefType([None], "f32")
index_type = IndexType()
distribution = direct_distribution()


@func
def copy(source: source_type, target: target_type, base: index_type) -> None:
    tile = load_tile(source, [base], distribution=distribution)
    store_tile(tile, target, [base], distribution=distribution)


module = copy.emit()
assert module.verify()
print(module)

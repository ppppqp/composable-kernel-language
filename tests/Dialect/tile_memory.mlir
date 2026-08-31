#direct = #ckl.distribution<
  executors = #ckl.space<[lane = 4]>, local = #ckl.space<[value = 2]>,
  tile = #ckl.space<[x = 8]>,
  ownership = #ckl.index_map<
    domain = #ckl.space<[lane = 4, value = 2]>, codomain = #ckl.space<[x = 8]>,
    results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<
    domain = #ckl.space<[value = 2]>, codomain = #ckl.space<[value = 2]>,
    results = [dim(0)], predicate = true>,
  scope = subgroup, replicated = false>

module {
  func.func @copy(%source: memref<8xf32>, %target: memref<?xf32>, %base: index) {
    %tile = ckl.load_tile %source[%base] distribution = #direct attributes {}
        : memref<8xf32> -> !ckl.tile<f32, #ckl.space<[x = 8]>>
    ckl.store_tile %tile, %target[%base] distribution = #direct attributes {}
        : !ckl.tile<f32, #ckl.space<[x = 8]>>, memref<?xf32>
    return
  }
}

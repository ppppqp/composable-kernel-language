#direct = #ckl.distribution<
  executors = #ckl.space<[lane = 2]>, local = #ckl.space<[value = 2]>,
  tile = #ckl.space<[x = 4]>,
  ownership = #ckl.index_map<domain = #ckl.space<[lane = 2, value = 2]>,
    codomain = #ckl.space<[x = 4]>, results = [add(mul(dim(0), const(2)), dim(1))],
    predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[value = 2]>,
    codomain = #ckl.space<[address = 2]>, results = [dim(0)], predicate = true>,
  scope = subgroup, replicated = false>
#permuted = #ckl.distribution<
  executors = #ckl.space<[lane = 2]>, local = #ckl.space<[value = 2]>,
  tile = #ckl.space<[x = 4]>,
  ownership = #ckl.index_map<domain = #ckl.space<[lane = 2, value = 2]>,
    codomain = #ckl.space<[x = 4]>, results = [add(mul(dim(0), const(2)), dim(1))],
    predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[value = 2]>,
    codomain = #ckl.space<[address = 2]>,
    results = [add(const(1), mul(dim(0), const(-1)))], predicate = true>,
  scope = subgroup, replicated = false>

module {
  gpu.module @kernels {
    ckl.task @layout_choice : (!ckl.tile<f32, #ckl.space<[x = 4]>>) ->
        !ckl.tile<f32, #ckl.space<[x = 4]>> alternatives = [
      {name = "direct", implementation_id = "layout.direct", estimated_execution_cost = 0 : i64,
       inputs = [{name = "input", distribution = #direct, placement = "private"}],
       outputs = [{name = "output", distribution = #direct, placement = "private"}]},
      {name = "permuted", implementation_id = "layout.permuted", estimated_execution_cost = 0 : i64,
       inputs = [{name = "input", distribution = #permuted, placement = "private"}],
       outputs = [{name = "output", distribution = #permuted, placement = "private"}]}
    ]

    gpu.func @direct_boundary(%source: memref<4xf32>, %target: memref<4xf32>) kernel {
      %c0 = arith.constant 0 : index
      %tile = ckl.load_tile %source[%c0] distribution = #direct attributes {} :
          memref<4xf32> -> !ckl.tile<f32, #ckl.space<[x = 4]>>
      %result = ckl.invoke @layout_choice(%tile) :
          (!ckl.tile<f32, #ckl.space<[x = 4]>>) -> !ckl.tile<f32, #ckl.space<[x = 4]>>
      ckl.store_tile %result, %target[%c0] distribution = #direct attributes {} :
          !ckl.tile<f32, #ckl.space<[x = 4]>>, memref<4xf32>
      gpu.return
    }
    gpu.func @permuted_boundary(%source: memref<4xf32>, %target: memref<4xf32>) kernel {
      %c0 = arith.constant 0 : index
      %tile = ckl.load_tile %source[%c0] distribution = #permuted attributes {} :
          memref<4xf32> -> !ckl.tile<f32, #ckl.space<[x = 4]>>
      %result = ckl.invoke @layout_choice(%tile) :
          (!ckl.tile<f32, #ckl.space<[x = 4]>>) -> !ckl.tile<f32, #ckl.space<[x = 4]>>
      ckl.store_tile %result, %target[%c0] distribution = #permuted attributes {} :
          !ckl.tile<f32, #ckl.space<[x = 4]>>, memref<4xf32>
      gpu.return
    }
  }
}

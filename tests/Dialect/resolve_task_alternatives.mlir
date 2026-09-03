#direct = #ckl.distribution<
  executors = #ckl.space<[lane = 2]>, local = #ckl.space<[value = 2]>,
  tile = #ckl.space<[x = 4]>,
  ownership = #ckl.index_map<domain = #ckl.space<[lane = 2, value = 2]>,
    codomain = #ckl.space<[x = 4]>, results = [add(mul(dim(0), const(2)), dim(1))],
    predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[value = 2]>,
    codomain = #ckl.space<[address = 2]>, results = [dim(0)], predicate = true>,
  scope = subgroup, replicated = false>

module {
  func.func private @leaf_impl(%tile: !ckl.tile<f32, #ckl.space<[x = 4]>>)
      -> !ckl.tile<f32, #ckl.space<[x = 4]>> {
    %leaf_marker = arith.constant 41 : index
    return %tile : !ckl.tile<f32, #ckl.space<[x = 4]>>
  }
  ckl.task @leaf_task : (!ckl.tile<f32, #ckl.space<[x = 4]>>) ->
      !ckl.tile<f32, #ckl.space<[x = 4]>> alternatives = [
    {name = "leaf", implementation = @leaf_impl, estimated_execution_cost = 0 : i64,
     inputs = [{name = "input", distribution = #direct}],
     outputs = [{name = "output", distribution = #direct}]}
  ]

  func.func private @composite_impl(%tile: !ckl.tile<f32, #ckl.space<[x = 4]>>)
      -> !ckl.tile<f32, #ckl.space<[x = 4]>> {
    %composite_marker = arith.constant 42 : index
    %result = ckl.invoke @leaf_task(%tile) :
        (!ckl.tile<f32, #ckl.space<[x = 4]>>) -> !ckl.tile<f32, #ckl.space<[x = 4]>>
    return %result : !ckl.tile<f32, #ckl.space<[x = 4]>>
  }
  ckl.task @composite_task : (!ckl.tile<f32, #ckl.space<[x = 4]>>) ->
      !ckl.tile<f32, #ckl.space<[x = 4]>> alternatives = [
    {name = "composite", implementation = @composite_impl,
     estimated_execution_cost = 0 : i64,
     inputs = [{name = "input", distribution = #direct}],
     outputs = [{name = "output", distribution = #direct}]}
  ]

  func.func @root(%tile: !ckl.tile<f32, #ckl.space<[x = 4]>>)
      -> !ckl.tile<f32, #ckl.space<[x = 4]>> {
    %result = ckl.invoke @composite_task(%tile) :
        (!ckl.tile<f32, #ckl.space<[x = 4]>>) -> !ckl.tile<f32, #ckl.space<[x = 4]>>
    return %result : !ckl.tile<f32, #ckl.space<[x = 4]>>
  }
}

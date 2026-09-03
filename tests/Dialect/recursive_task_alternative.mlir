#direct = #ckl.distribution<
  executors = #ckl.space<[lane = 1]>, local = #ckl.space<[value = 1]>,
  tile = #ckl.space<[x = 1]>,
  ownership = #ckl.index_map<domain = #ckl.space<[lane = 1, value = 1]>,
    codomain = #ckl.space<[x = 1]>, results = [const(0)], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[value = 1]>,
    codomain = #ckl.space<[address = 1]>, results = [const(0)], predicate = true>,
  scope = subgroup, replicated = false>

module {
  func.func private @recursive_impl(%tile: !ckl.tile<f32, #ckl.space<[x = 1]>>)
      -> !ckl.tile<f32, #ckl.space<[x = 1]>> {
    %result = ckl.invoke @recursive_task(%tile) :
        (!ckl.tile<f32, #ckl.space<[x = 1]>>) -> !ckl.tile<f32, #ckl.space<[x = 1]>>
    return %result : !ckl.tile<f32, #ckl.space<[x = 1]>>
  }
  ckl.task @recursive_task : (!ckl.tile<f32, #ckl.space<[x = 1]>>) ->
      !ckl.tile<f32, #ckl.space<[x = 1]>> alternatives = [
    {name = "recursive", implementation = @recursive_impl,
     estimated_execution_cost = 0 : i64,
     inputs = [{name = "input", distribution = #direct}],
     outputs = [{name = "output", distribution = #direct}]}
  ]
  func.func @root(%tile: !ckl.tile<f32, #ckl.space<[x = 1]>>)
      -> !ckl.tile<f32, #ckl.space<[x = 1]>> {
    %result = ckl.invoke @recursive_task(%tile) :
        (!ckl.tile<f32, #ckl.space<[x = 1]>>) -> !ckl.tile<f32, #ckl.space<[x = 1]>>
    return %result : !ckl.tile<f32, #ckl.space<[x = 1]>>
  }
}

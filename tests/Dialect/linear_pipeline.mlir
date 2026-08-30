#direct = #ckl.distribution<
  executors = #ckl.space<[p = 2]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 4]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 2, y = 2]>,
    codomain = #ckl.space<[x = 4]>,
    results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>,
    codomain = #ckl.space<[y = 2]>, results = [dim(0)], predicate = true>,
  scope = workgroup, replicated = false>

#permuted = #ckl.distribution<
  executors = #ckl.space<[p = 2]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 4]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 2, y = 2]>,
    codomain = #ckl.space<[x = 4]>,
    results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>,
    codomain = #ckl.space<[r = 2]>,
    results = [add(const(1), mul(dim(0), const(-1)))], predicate = true>,
  scope = workgroup, replicated = false>

#rotated = #ckl.distribution<
  executors = #ckl.space<[p = 2]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 4]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 2, y = 2]>,
    codomain = #ckl.space<[x = 4]>,
    results = [add(mul(mod(add(dim(0), const(1)), 2), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>,
    codomain = #ckl.space<[y = 2]>, results = [dim(0)], predicate = true>,
  scope = workgroup, replicated = false>

ckl.task @source : () -> !ckl.tile<f32, #ckl.space<[x = 4]>> alternatives = [
  {name = "only", implementation_id = "source.v1",
   outputs = [{name = "out", distribution = #direct}]}
]

ckl.task @middle : (!ckl.tile<f32, #ckl.space<[x = 4]>>) ->
    !ckl.tile<f32, #ckl.space<[x = 4]>> alternatives = [
  {name = "greedy", implementation_id = "middle.greedy.v1",
   inputs = [{name = "in", distribution = #direct}],
   outputs = [{name = "out", distribution = #rotated}]},
  {name = "global", implementation_id = "middle.global.v1",
   inputs = [{name = "in", distribution = #permuted}],
   outputs = [{name = "out", distribution = #direct}]}
]

ckl.task @sink : (!ckl.tile<f32, #ckl.space<[x = 4]>>) -> () alternatives = [
  {name = "only", implementation_id = "sink.v1",
   inputs = [{name = "in", distribution = #direct}]}
]

func.func @pipeline(%tile: !ckl.tile<f32, #ckl.space<[x = 4]>>)
    -> !ckl.tile<f32, #ckl.space<[x = 4]>> {
  %first = ckl.task_compose %tile producer = @source["out"] consumer = @middle["in"]
      capabilities = [] attributes : !ckl.tile<f32, #ckl.space<[x = 4]>>
  %second = ckl.task_compose %first producer = @middle["out"] consumer = @sink["in"]
      capabilities = [] attributes : !ckl.tile<f32, #ckl.space<[x = 4]>>
  return %second : !ckl.tile<f32, #ckl.space<[x = 4]>>
}

#direct = #ckl.distribution<
  executors = #ckl.space<[p = 4]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 4, y = 2]>, codomain = #ckl.space<[x = 8]>, results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>, codomain = #ckl.space<[y = 2]>, results = [dim(0)], predicate = true>, scope = subgroup, replicated = false>
#rotated = #ckl.distribution<
  executors = #ckl.space<[p = 4]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 4, y = 2]>, codomain = #ckl.space<[x = 8]>, results = [add(mul(mod(add(dim(0), const(1)), 4), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>, codomain = #ckl.space<[y = 2]>, results = [dim(0)], predicate = true>, scope = subgroup, replicated = false>

module attributes {ckl.subgroup_size = 4 : i64} {
ckl.task @producer : (!ckl.tile<f32, #ckl.space<[x = 8]>>) -> !ckl.tile<f32, #ckl.space<[x = 8]>> alternatives = [
  {name = "direct", implementation_id = "producer.direct.v1", estimated_execution_cost = 150 : i64,
   inputs = [{name = "in", distribution = #direct}], outputs = [{name = "out", distribution = #direct}]},
  {name = "cheap", implementation_id = "producer.cheap.v1", estimated_execution_cost = 0 : i64,
   inputs = [{name = "in", distribution = #direct}], outputs = [{name = "out", distribution = #rotated}]}
]
ckl.task @left : (!ckl.tile<f32, #ckl.space<[x = 8]>>) -> !ckl.tile<f32, #ckl.space<[x = 8]>> alternatives = [
  {name = "only", implementation_id = "left.v1", inputs = [{name = "in", distribution = #direct}], outputs = [{name = "out", distribution = #direct}]}
]
ckl.task @right : (!ckl.tile<f32, #ckl.space<[x = 8]>>) -> !ckl.tile<f32, #ckl.space<[x = 8]>> alternatives = [
  {name = "only", implementation_id = "right.v1", inputs = [{name = "in", distribution = #direct}], outputs = [{name = "out", distribution = #direct}]}
]

  func.func @fanout(%tile: !ckl.tile<f32, #ckl.space<[x = 8]>>) -> (!ckl.tile<f32, #ckl.space<[x = 8]>>, !ckl.tile<f32, #ckl.space<[x = 8]>>) {
    %produced = ckl.invoke @producer(%tile) : (!ckl.tile<f32, #ckl.space<[x = 8]>>) -> !ckl.tile<f32, #ckl.space<[x = 8]>>
    %left = ckl.invoke @left(%produced) : (!ckl.tile<f32, #ckl.space<[x = 8]>>) -> !ckl.tile<f32, #ckl.space<[x = 8]>>
    %right = ckl.invoke @right(%produced) : (!ckl.tile<f32, #ckl.space<[x = 8]>>) -> !ckl.tile<f32, #ckl.space<[x = 8]>>
    return %left, %right : !ckl.tile<f32, #ckl.space<[x = 8]>>, !ckl.tile<f32, #ckl.space<[x = 8]>>
  }
}

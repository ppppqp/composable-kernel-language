#direct = #ckl.distribution<
  executors = #ckl.space<[p = 2]>, local = #ckl.space<[y = 2]>,
  tile = #ckl.space<[x = 4]>,
  ownership = #ckl.index_map<
    domain = #ckl.space<[p = 2, y = 2]>, codomain = #ckl.space<[x = 4]>,
    results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<
    domain = #ckl.space<[y = 2]>, codomain = #ckl.space<[y = 2]>,
    results = [dim(0)], predicate = true>,
  scope = workgroup, replicated = false>

#permuted = #ckl.distribution<
  executors = #ckl.space<[p = 2]>, local = #ckl.space<[y = 2]>,
  tile = #ckl.space<[x = 4]>,
  ownership = #ckl.index_map<
    domain = #ckl.space<[p = 2, y = 2]>, codomain = #ckl.space<[x = 4]>,
    results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<
    domain = #ckl.space<[y = 2]>, codomain = #ckl.space<[r = 2]>,
    results = [add(const(1), mul(dim(0), const(-1)))], predicate = true>,
  scope = workgroup, replicated = false>

ckl.task @producer : (!ckl.tile<f32, #ckl.space<[x = 4]>>) ->
    !ckl.tile<f32, #ckl.space<[x = 4]>> alternatives = [
  {name = "expensive", implementation_id = "producer.expensive.v1", origin = "user",
   registers_per_thread = 80 : i64,
   shared_memory_bytes = 8192 : i64, estimated_execution_cost = 1 : i64,
   required_capabilities = ["async-copy"],
   resources = [{name = "scratch", bytes = 8192 : i64, begin = 0 : i64, end = 2 : i64}],
   effects = [{kind = "write", resource = "scratch", stage = 0 : i64}],
   outputs = [{name = "weight", distribution = #permuted, placement = "shared"}]},
  {name = "direct", implementation_id = "producer.direct.v1", origin = "compiler",
   registers_per_thread = 16 : i64,
   estimated_execution_cost = 15 : i64,
   outputs = [{name = "weight", distribution = #direct, placement = "private"}]}
]

ckl.task @consumer : (!ckl.tile<f32, #ckl.space<[x = 4]>>) ->
    !ckl.tile<f32, #ckl.space<[x = 4]>> alternatives = [
  {name = "operand", implementation_id = "consumer.operand.v1", origin = "extension",
   registers_per_thread = 32 : i64,
   estimated_execution_cost = 20 : i64,
   required_capabilities = ["mma"],
   inputs = [{name = "rhs", distribution = #direct, placement = "private"}]}
]

func.func @select(%tile: !ckl.tile<f32, #ckl.space<[x = 4]>>)
    -> !ckl.tile<f32, #ckl.space<[x = 4]>> {
  %selected = ckl.task_compose %tile producer = @producer["weight"]
      consumer = @consumer["rhs"] capabilities = ["mma"]
      attributes {register_limit = 64 : i64, shared_memory_limit = 4096 : i64}
      : !ckl.tile<f32, #ckl.space<[x = 4]>>
  return %selected : !ckl.tile<f32, #ckl.space<[x = 4]>>
}

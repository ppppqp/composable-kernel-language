#base = #ckl.distribution<
  executors = #ckl.space<[p = 4]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 4, y = 2]>,
    codomain = #ckl.space<[x = 8]>,
    results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>,
    codomain = #ckl.space<[y = 2]>, results = [dim(0)], predicate = true>,
  scope = subgroup, replicated = false>

#reverse = #ckl.distribution<
  executors = #ckl.space<[p = 4]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 4, y = 2]>,
    codomain = #ckl.space<[x = 8]>,
    results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>,
    codomain = #ckl.space<[r = 2]>,
    results = [add(const(1), mul(dim(0), const(-1)))], predicate = true>,
  scope = subgroup, replicated = false>

#rotated = #ckl.distribution<
  executors = #ckl.space<[p = 4]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 4, y = 2]>,
    codomain = #ckl.space<[x = 8]>,
    results = [add(mul(mod(add(dim(0), const(1)), 4), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>,
    codomain = #ckl.space<[y = 2]>, results = [dim(0)], predicate = true>,
  scope = subgroup, replicated = false>

#workgroup = #ckl.distribution<
  executors = #ckl.space<[p = 4]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 4, y = 2]>,
    codomain = #ckl.space<[x = 8]>,
    results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>,
    codomain = #ckl.space<[y = 2]>, results = [dim(0)], predicate = true>,
  scope = workgroup, replicated = false>

#grid = #ckl.distribution<
  executors = #ckl.space<[p = 4]>, local = #ckl.space<[y = 2]>, tile = #ckl.space<[x = 8]>,
  ownership = #ckl.index_map<domain = #ckl.space<[p = 4, y = 2]>,
    codomain = #ckl.space<[x = 8]>,
    results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
  local_storage = #ckl.index_map<domain = #ckl.space<[y = 2]>,
    codomain = #ckl.space<[y = 2]>, results = [dim(0)], predicate = true>,
  scope = grid, replicated = false>

func.func @all_classes(%tile: !ckl.tile<f32, #ckl.space<[x = 8]>>)
    -> (!ckl.tile<f32, #ckl.space<[x = 8]>>, !ckl.tile<f32, #ckl.space<[x = 8]>>,
        !ckl.tile<f32, #ckl.space<[x = 8]>>, !ckl.tile<f32, #ckl.space<[x = 8]>>,
        !ckl.tile<f32, #ckl.space<[x = 8]>>) {
  %identity = ckl.compose %tile source = #base target = #base attributes {subgroup_size = 4 : i64} : <f32, #ckl.space<[x = 8]>>
  %local = ckl.compose %tile source = #base target = #reverse attributes {subgroup_size = 4 : i64} : <f32, #ckl.space<[x = 8]>>
  %subgroup = ckl.compose %tile source = #base target = #rotated attributes {subgroup_size = 4 : i64} : <f32, #ckl.space<[x = 8]>>
  %shared = ckl.compose %tile source = #base target = #workgroup attributes {subgroup_size = 4 : i64} : <f32, #ckl.space<[x = 8]>>
  %global = ckl.compose %tile source = #base target = #grid attributes {subgroup_size = 4 : i64} : <f32, #ckl.space<[x = 8]>>
  return %identity, %local, %subgroup, %shared, %global : !ckl.tile<f32, #ckl.space<[x = 8]>>, !ckl.tile<f32, #ckl.space<[x = 8]>>, !ckl.tile<f32, #ckl.space<[x = 8]>>, !ckl.tile<f32, #ckl.space<[x = 8]>>, !ckl.tile<f32, #ckl.space<[x = 8]>>
}

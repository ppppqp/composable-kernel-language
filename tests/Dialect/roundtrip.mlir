// A local-register permutation is enough to exercise the complete
// text -> dialect attribute -> CKLCore -> planning -> dialect operation path.
module attributes {ckl.test_map = #ckl.index_map<
  domain = #ckl.space<[x = 8]>, codomain = #ckl.space<[x = 8]>,
  results = [xor(mod(floordiv(dim(0), 2), 3), const(1))],
  predicate = and(cmp(ge, dim(0), const(0)), cmp(lt, dim(0), const(7)))>} {

func.func private @symbolic_tile(!ckl.tile<f32, #ckl.space<[m = 4, n = symbol<"N">]>>)

func.func @local_permutation(
    %arg0: !ckl.tile<f32, #ckl.space<[x = 4]>>)
    -> !ckl.tile<f32, #ckl.space<[x = 4]>> {
  %0 = ckl.compose %arg0
    source = #ckl.distribution<
      executors = #ckl.space<[p = 2]>,
      local = #ckl.space<[y = 2]>,
      tile = #ckl.space<[x = 4]>,
      ownership = #ckl.index_map<
        domain = #ckl.space<[p = 2, y = 2]>,
        codomain = #ckl.space<[x = 4]>,
        results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
      local_storage = #ckl.index_map<
        domain = #ckl.space<[y = 2]>, codomain = #ckl.space<[y = 2]>,
        results = [dim(0)], predicate = true>,
      scope = workgroup, replicated = false>
    target = #ckl.distribution<
      executors = #ckl.space<[p = 2]>,
      local = #ckl.space<[y = 2]>,
      tile = #ckl.space<[x = 4]>,
      ownership = #ckl.index_map<
        domain = #ckl.space<[p = 2, y = 2]>,
        codomain = #ckl.space<[x = 4]>,
        results = [add(mul(dim(0), const(2)), dim(1))], predicate = true>,
      local_storage = #ckl.index_map<
        domain = #ckl.space<[y = 2]>, codomain = #ckl.space<[r = 2]>,
        results = [add(const(1), mul(dim(0), const(-1)))], predicate = true>,
      scope = workgroup, replicated = false> attributes
    : <f32, #ckl.space<[x = 4]>>
  return %0 : !ckl.tile<f32, #ckl.space<[x = 4]>>
}
}

// A local-register permutation is enough to exercise the complete
// text -> dialect attribute -> CKLCore -> planning -> dialect operation path.
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
        results = ["a(m(i(0),c(2)),i(1))"], predicate = "t">,
      local_storage = #ckl.index_map<
        domain = #ckl.space<[y = 2]>, codomain = #ckl.space<[y = 2]>,
        results = ["i(0)"], predicate = "t">,
      scope = workgroup, replicated = false>
    target = #ckl.distribution<
      executors = #ckl.space<[p = 2]>,
      local = #ckl.space<[y = 2]>,
      tile = #ckl.space<[x = 4]>,
      ownership = #ckl.index_map<
        domain = #ckl.space<[p = 2, y = 2]>,
        codomain = #ckl.space<[x = 4]>,
        results = ["a(m(i(0),c(2)),i(1))"], predicate = "t">,
      local_storage = #ckl.index_map<
        domain = #ckl.space<[y = 2]>, codomain = #ckl.space<[r = 2]>,
        results = ["a(c(1),m(i(0),c(-1)))"], predicate = "t">,
      scope = workgroup, replicated = false> attributes
    : <f32, #ckl.space<[x = 4]>>
  return %0 : !ckl.tile<f32, #ckl.space<[x = 4]>>
}

// A local-register permutation is enough to exercise the complete
// text -> dialect attribute -> CKLCore -> planning -> dialect operation path.
func.func @local_permutation(
    %arg0: !ckl.tile<f32, #ckl.space<"s(x:4)">>)
    -> !ckl.tile<f32, #ckl.space<"s(x:4)">> {
  %0 = ckl.compose %arg0
    source = #ckl.distribution<"dist(1,0,6:s(p:2),6:s(y:2),6:s(x:4),47:map(s(p:2,y:2);s(x:4);[a(m(i(0),c(2)),i(1))];t),27:map(s(y:2);s(y:2);[i(0)];t))">
    target = #ckl.distribution<"dist(1,0,6:s(p:2),6:s(y:2),6:s(x:4),47:map(s(p:2,y:2);s(x:4);[a(m(i(0),c(2)),i(1))];t),44:map(s(y:2);s(r:2);[a(c(1),m(i(0),c(-1)))];t))"> attributes
    : <f32, #ckl.space<"s(x:4)">>
  return %0 : !ckl.tile<f32, #ckl.space<"s(x:4)">>
}

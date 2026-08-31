module {
  ckl.task @mma : (
      !ckl.tile<f16, #ckl.space<[m = 16, k = 16]>>,
      !ckl.tile<f16, #ckl.space<[k = 16, n = 8]>>,
      !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>)
      -> !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>> alternatives = [{
    name = "mma-sync-m16n8k16-f16-f32",
    implementation_id = "nvidia.mma.sync.m16n8k16.row.col.f32.f16.f16.f32"
  }]

  func.func @kernel(
      %lhs: !ckl.tile<f16, #ckl.space<[m = 16, k = 16]>>,
      %rhs: !ckl.tile<f16, #ckl.space<[k = 16, n = 8]>>,
      %acc: !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>)
      -> !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>> {
    %result = ckl.invoke @mma(%lhs, %rhs, %acc)
        alternative = "mma-sync-m16n8k16-f16-f32" {
          ckl.implementation_id = "nvidia.mma.sync.m16n8k16.row.col.f32.f16.f16.f32",
          ckl.graph_score = 1 : i64
        } : (!ckl.tile<f16, #ckl.space<[m = 16, k = 16]>>,
             !ckl.tile<f16, #ckl.space<[k = 16, n = 8]>>,
             !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>)
          -> !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>
    return %result : !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>
  }
}

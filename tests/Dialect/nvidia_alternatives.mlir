module attributes {
  ckl.target = "nvidia",
  ckl.architecture = "sm_80",
  ckl.available_capabilities = ["nvidia.mma.sync.m16n8k16.f16"]
} {
  ckl.task @mma : (
      !ckl.tile<f16, #ckl.space<[m = 16, k = 16]>>,
      !ckl.tile<f16, #ckl.space<[k = 16, n = 8]>>,
      !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>>)
      -> !ckl.tile<f32, #ckl.space<[m = 16, n = 8]>> alternatives = [] {
    ckl.alternative_providers = [{
      id = "nvidia.mma-sync.f16-f32",
      inputs = ["lhs", "rhs", "acc"],
      outputs = ["result"]
    }]
  }
}

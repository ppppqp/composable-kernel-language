module {
  ckl.task @choose : (!ckl.tile<f32, #ckl.space<[x = 4]>>) ->
      !ckl.tile<f32, #ckl.space<[x = 4]>> alternatives = [
    {name = "slow", implementation_id = "choose.slow", estimated_execution_cost = 20 : i64},
    {name = "fast", implementation_id = "choose.fast", estimated_execution_cost = 2 : i64}
  ]

  func.func @kernel(%arg: !ckl.tile<f32, #ckl.space<[x = 4]>>) ->
      !ckl.tile<f32, #ckl.space<[x = 4]>> {
    %result = ckl.invoke @choose(%arg) :
        (!ckl.tile<f32, #ckl.space<[x = 4]>>) -> !ckl.tile<f32, #ckl.space<[x = 4]>>
    return %result : !ckl.tile<f32, #ckl.space<[x = 4]>>
  }
}

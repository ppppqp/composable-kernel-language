ckl.task @copy : (!ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>) ->
    !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>
    alternatives = [{name = "direct", target = "generic"},
                    {name = "amd", target = "gfx"}]

func.func @bind_and_invoke(
    %tile: !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>, %m: index)
    -> !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>> {
  %bound = ckl.bind_symbols %tile [%m] symbols = ["M"]
      : !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>
        -> !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>
  %result = ckl.invoke @copy(%bound) alternative = "amd"
      : (!ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>)
        -> !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>
  return %result : !ckl.tile<f32, #ckl.space<[m = symbol<"M">]>>
}

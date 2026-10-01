module {
  func.func @source() { func.return }
  "ckl_exec.plan"() <{backend = "cuda", name = "invalid", source = @source}> ({
    "ckl_exec.heap"() <{address_space = "global", alignment = 16 : i64,
                         bytes = 64 : i64, device = 0 : i64, id = 0 : i64}> : () -> ()
    "ckl_exec.resource"() <{address_space = "global", alignment = 16 : i64,
                             bytes = 32 : i64, device = "cuda:0", heap = 0 : i64,
                             kind = "temporary", name = "overflow", offset = 48 : i64}> : () -> ()
  }) : () -> ()
}

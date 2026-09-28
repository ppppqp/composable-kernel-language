module {
  func.func @kernel(%resource: memref<16xf32>) {
    func.return
  }

  func.func @host(%resource: memref<16xf32>) {
    %done = "ckl.dispatch"(%resource) {
      block = array<i64: 0, 1, 1>,
      capabilities = [],
      device = "cuda:0",
      grid = array<i64: 1, 1, 1>,
      implementation = "kernel.v1",
      kernel = @kernel,
      operandSegmentSizes = array<i32: 1, 0>,
      shared_memory = 0 : i64
    } : (memref<16xf32>) -> !ckl.token
    func.return
  }
}

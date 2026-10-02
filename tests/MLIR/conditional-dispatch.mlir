module {
  func.func @write(%resource: memref<8xf32>) {
    %value = arith.constant 1.0 : f32
    "ckl.store"(%value, %resource) : (f32, memref<8xf32>) -> ()
    func.return
  }

  func.func @step(%condition: i1, %resource: memref<8xf32>) {
    scf.if %condition {
      %done = "ckl.dispatch"(%resource) {
        artifact = "kernels.cubin", block = array<i64: 32, 1, 1>,
        capabilities = ["cuda.direct"], device = "cuda:0",
        grid = array<i64: 1, 1, 1>, implementation = "write.v1", kernel = @write,
        operandSegmentSizes = array<i32: 1, 0>, shared_memory = 0 : i64
      } : (memref<8xf32>) -> !ckl.token
    }
    func.return
  }
}

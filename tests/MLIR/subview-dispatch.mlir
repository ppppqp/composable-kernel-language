module {
  func.func @write(%resource: memref<4xf32, strided<[1], offset: 4>>) {
    %value = arith.constant 1.0 : f32
    "ckl.store"(%value, %resource) :
      (f32, memref<4xf32, strided<[1], offset: 4>>) -> ()
    func.return
  }

  func.func @step(%resource: memref<8xf32>) {
    %view = memref.subview %resource[4] [4] [1] :
      memref<8xf32> to memref<4xf32, strided<[1], offset: 4>>
    %done = "ckl.dispatch"(%view) {
      artifact = "kernels.cubin", block = array<i64: 32, 1, 1>,
      capabilities = ["cuda.direct"], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "write.v1", kernel = @write,
      operandSegmentSizes = array<i32: 1, 0>, shared_memory = 0 : i64
    } : (memref<4xf32, strided<[1], offset: 4>>) -> !ckl.token
    func.return
  }
}

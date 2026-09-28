module {
  func.func @kernel(%src: memref<16xf32>, %dst: memref<16xf32>) {
    %value = "ckl.load"(%src) : (memref<16xf32>) -> f32
    "ckl.atomic_rmw"(%value, %dst) {ordering = "acq_rel", scope = "device"}
      : (f32, memref<16xf32>) -> f32
    "ckl.sync"(%dst) {scope = "device"} : (memref<16xf32>) -> ()
    func.return
  }

  func.func @host(%input: memref<16xf32>, %output: memref<16xf32>) {
    %done = "ckl.dispatch"(%input, %output) {
      block = array<i64: 256, 1, 1>,
      capabilities = ["sm_120"],
      device = "cuda:0",
      grid = array<i64: 1, 1, 1>,
      implementation = "kernel.ptx87.v1",
      kernel = @kernel,
      operandSegmentSizes = array<i32: 2, 0>,
      shared_memory = 0 : i64
    } : (memref<16xf32>, memref<16xf32>) -> !ckl.token
    func.return
  }
}

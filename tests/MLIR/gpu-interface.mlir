module attributes {gpu.container_module} {
  gpu.module @flydsl_kernels {
    gpu.func @write(%resource: memref<16xf32>) kernel {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1.0 : f32
      memref.store %one, %resource[%c0] : memref<16xf32>
      gpu.return
    }
  }

  func.func @read(%resource: memref<16xf32>) {
    %value = "ckl.load"(%resource) : (memref<16xf32>) -> f32
    func.return
  }

  func.func @mixed_step(%resource: memref<16xf32>) {
    %one = arith.constant 1 : index
    gpu.launch_func @flydsl_kernels::@write
      blocks in (%one, %one, %one) threads in (%one, %one, %one)
      args(%resource : memref<16xf32>)
      {ckl.abi = "cuda.direct", ckl.artifact = "flydsl.module.v1", ckl.device = "cuda:0",
       ckl.implementation = "flydsl.write.v1"}
    %done = "ckl.dispatch"(%resource) {
      block = array<i64: 1, 1, 1>, capabilities = ["cuda.direct"], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "ckl.read.v1", kernel = @read,
      operandSegmentSizes = array<i32: 1, 0>, shared_memory = 0 : i64
    } : (memref<16xf32>) -> !ckl.token
    func.return
  }
}

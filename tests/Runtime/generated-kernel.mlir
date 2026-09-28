module {
  gpu.module @kernels [#nvvm.target<chip = "sm_120", features = "+ptx87">] {
    gpu.func @increment(%buffer: memref<?xf32>, %size: index) kernel {
      %thread = gpu.thread_id x
      %block = gpu.block_id x
      %block_size = gpu.block_dim x
      %base = arith.muli %block, %block_size : index
      %index = arith.addi %base, %thread : index
      %inside = arith.cmpi ult, %index, %size : index
      scf.if %inside {
        %value = memref.load %buffer[%index] : memref<?xf32>
        %one = arith.constant 1.0 : f32
        %updated = arith.addf %value, %one : f32
        memref.store %updated, %buffer[%index] : memref<?xf32>
      }
      gpu.return
    }
  }
}

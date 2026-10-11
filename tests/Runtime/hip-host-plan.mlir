// Semantic kernel body for host-plan validation; kernels.hip supplies the test binary.
module {
  func.func @add_bias(%values: memref<1024xf32>, %count: i32, %bias: f32) {
    %value = "ckl.load"(%values) : (memref<1024xf32>) -> f32
    %updated = arith.addf %value, %bias : f32
    "ckl.store"(%updated, %values) : (f32, memref<1024xf32>) -> ()
    func.return
  }

  func.func @launch(%values: memref<1024xf32>, %count: i32) {
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    %first = "ckl.dispatch"(%values, %count, %one) {
      artifact = "kernels.hsaco", implementation = "add_bias.first",
      kernel = @add_bias, device = "rocm:0", capabilities = ["rocm.bare_ptr"],
      grid = array<i64: 4, 1, 1>, block = array<i64: 256, 1, 1>,
      shared_memory = 0 : i64, operandSegmentSizes = array<i32: 3, 0>
    } : (memref<1024xf32>, i32, f32) -> !ckl.token
    %second = "ckl.dispatch"(%values, %count, %two) {
      artifact = "kernels.hsaco", implementation = "add_bias.second",
      kernel = @add_bias, device = "rocm:0", capabilities = ["rocm.bare_ptr"],
      grid = array<i64: 4, 1, 1>, block = array<i64: 256, 1, 1>,
      shared_memory = 0 : i64, operandSegmentSizes = array<i32: 3, 0>
    } : (memref<1024xf32>, i32, f32) -> !ckl.token
    func.return
  }
}

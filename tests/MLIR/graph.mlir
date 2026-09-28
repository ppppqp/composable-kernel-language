module {
  func.func @read(%resource: memref<16xf32>) {
    %value = "ckl.load"(%resource) : (memref<16xf32>) -> f32
    func.return
  }

  func.func @write(%resource: memref<16xf32>) {
    %value = arith.constant 1.0 : f32
    "ckl.store"(%value, %resource) : (f32, memref<16xf32>) -> ()
    func.return
  }

  func.func @step() {
    %x = "ckl.alloc"() : () -> memref<16xf32>
    %y = memref.alloc() : memref<16xf32>
    %x_written = "ckl.dispatch"(%x) {
      block = array<i64: 256, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "write.v1", kernel = @write,
      operandSegmentSizes = array<i32: 1, 0>, shared_memory = 0 : i64
    } : (memref<16xf32>) -> !ckl.token
    %y_written = "ckl.dispatch"(%y) {
      block = array<i64: 256, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "write.v1", kernel = @write,
      operandSegmentSizes = array<i32: 1, 0>, shared_memory = 0 : i64
    } : (memref<16xf32>) -> !ckl.token
    %x_read = "ckl.dispatch"(%x, %y_written) {
      block = array<i64: 256, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "read.v1", kernel = @read,
      operandSegmentSizes = array<i32: 1, 1>, shared_memory = 0 : i64
    } : (memref<16xf32>, !ckl.token) -> !ckl.token
    "ckl.free"(%x) : (memref<16xf32>) -> ()
    memref.dealloc %y : memref<16xf32>
    func.return
  }
}

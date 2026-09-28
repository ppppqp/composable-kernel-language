module {
  func.func @opaque(%resource: memref<4xf32>) {
    "foreign.opaque"(%resource) : (memref<4xf32>) -> ()
    func.return
  }

  func.func @read(%resource: memref<4xf32>) {
    %value = "ckl.load"(%resource) : (memref<4xf32>) -> f32
    func.return
  }

  func.func @step() {
    %x = "ckl.alloc"() : () -> memref<4xf32>
    %y = "ckl.alloc"() : () -> memref<4xf32>
    %opaque_done = "ckl.dispatch"(%x) {
      block = array<i64: 32, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "opaque.v1", kernel = @opaque,
      operandSegmentSizes = array<i32: 1, 0>, shared_memory = 0 : i64
    } : (memref<4xf32>) -> !ckl.token
    %read_done = "ckl.dispatch"(%y) {
      block = array<i64: 32, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "read.v1", kernel = @read,
      operandSegmentSizes = array<i32: 1, 0>, shared_memory = 0 : i64
    } : (memref<4xf32>) -> !ckl.token
    func.return
  }
}

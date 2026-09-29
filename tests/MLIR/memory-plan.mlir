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

  func.func @read_dynamic(%resource: memref<?xf32>) {
    %value = "ckl.load"(%resource) : (memref<?xf32>) -> f32
    func.return
  }

  func.func @step() {
    %first = "ckl.alloc"() : () -> memref<16xf32>
    %first_written = "ckl.dispatch"(%first) {
      block = array<i64: 256, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "write.v1", kernel = @write,
      operandSegmentSizes = array<i32: 1, 0>, shared_memory = 0 : i64
    } : (memref<16xf32>) -> !ckl.token
    %first_read = "ckl.dispatch"(%first, %first_written) {
      block = array<i64: 256, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "read.v1", kernel = @read,
      operandSegmentSizes = array<i32: 1, 1>, shared_memory = 0 : i64
    } : (memref<16xf32>, !ckl.token) -> !ckl.token
    "ckl.free"(%first) : (memref<16xf32>) -> ()

    %second = "ckl.alloc"() : () -> memref<16xf32>
    %second_written = "ckl.dispatch"(%second, %first_read) {
      block = array<i64: 256, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "write.v1", kernel = @write,
      operandSegmentSizes = array<i32: 1, 1>, shared_memory = 0 : i64
    } : (memref<16xf32>, !ckl.token) -> !ckl.token
    %second_read = "ckl.dispatch"(%second, %second_written) {
      block = array<i64: 256, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "read.v1", kernel = @read,
      operandSegmentSizes = array<i32: 1, 1>, shared_memory = 0 : i64
    } : (memref<16xf32>, !ckl.token) -> !ckl.token
    "ckl.free"(%second) : (memref<16xf32>) -> ()
    func.return
  }

  func.func @dynamic(%size: index) {
    %resource = memref.alloc(%size) : memref<?xf32>
    %done = "ckl.dispatch"(%resource) {
      block = array<i64: 256, 1, 1>, capabilities = [], device = "cuda:0",
      grid = array<i64: 1, 1, 1>, implementation = "read.dynamic", kernel = @read_dynamic,
      operandSegmentSizes = array<i32: 1, 0>, shared_memory = 0 : i64
    } : (memref<?xf32>) -> !ckl.token
    memref.dealloc %resource : memref<?xf32>
    func.return
  }
}

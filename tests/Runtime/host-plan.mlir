module {
  func.func @field(%input: memref<257xf32>, %output: memref<257xf32>, %size: i32, %bias: f32) {
    %value = "ckl.load"(%input) : (memref<257xf32>) -> f32
    "ckl.store"(%value, %output) : (f32, memref<257xf32>) -> ()
    func.return
  }

  func.func @join(%first: memref<257xf32>, %second: memref<257xf32>,
                  %output: memref<257xf32>, %size: i32) {
    %value = "ckl.load"(%first) : (memref<257xf32>) -> f32
    %unused = "ckl.load"(%second) : (memref<257xf32>) -> f32
    "ckl.store"(%value, %output) : (f32, memref<257xf32>) -> ()
    func.return
  }

  func.func @host(%input_a: memref<257xf32>, %input_b: memref<257xf32>,
                  %output: memref<257xf32>) {
    %size = arith.constant 257 : i32
    %bias_a = arith.constant 0.00001 : f32
    %bias_b = arith.constant 0.00002 : f32
    %temporary_a = "ckl.alloc"() : () -> memref<257xf32>
    %output_a = "ckl.alloc"() : () -> memref<257xf32>
    %temporary_b = "ckl.alloc"() : () -> memref<257xf32>
    %output_b = "ckl.alloc"() : () -> memref<257xf32>

    %a0 = "ckl.dispatch"(%input_a, %temporary_a, %size, %bias_a) {
      block = array<i64: 256, 1, 1>, capabilities = ["cuda.direct"], device = "cuda:0",
      grid = array<i64: 2, 1, 1>, implementation = "field.direct", kernel = @field,
      operandSegmentSizes = array<i32: 4, 0>, shared_memory = 0 : i64
    } : (memref<257xf32>, memref<257xf32>, i32, f32) -> !ckl.token
    %a1 = "ckl.dispatch"(%temporary_a, %output_a, %size, %bias_a, %a0) {
      block = array<i64: 256, 1, 1>, capabilities = ["cuda.direct"], device = "cuda:0",
      grid = array<i64: 2, 1, 1>, implementation = "field.direct", kernel = @field,
      operandSegmentSizes = array<i32: 4, 1>, shared_memory = 0 : i64
    } : (memref<257xf32>, memref<257xf32>, i32, f32, !ckl.token) -> !ckl.token
    %b0 = "ckl.dispatch"(%input_b, %temporary_b, %size, %bias_b, %a1) {
      block = array<i64: 256, 1, 1>, capabilities = ["cuda.direct"], device = "cuda:0",
      grid = array<i64: 2, 1, 1>, implementation = "field.direct", kernel = @field,
      operandSegmentSizes = array<i32: 4, 1>, shared_memory = 0 : i64
    } : (memref<257xf32>, memref<257xf32>, i32, f32, !ckl.token) -> !ckl.token
    %b1 = "ckl.dispatch"(%temporary_b, %output_b, %size, %bias_b, %b0) {
      block = array<i64: 256, 1, 1>, capabilities = ["cuda.direct"], device = "cuda:0",
      grid = array<i64: 2, 1, 1>, implementation = "field.direct", kernel = @field,
      operandSegmentSizes = array<i32: 4, 1>, shared_memory = 0 : i64
    } : (memref<257xf32>, memref<257xf32>, i32, f32, !ckl.token) -> !ckl.token
    %done = "ckl.dispatch"(%output_a, %output_b, %output, %size, %b1) {
      block = array<i64: 256, 1, 1>, capabilities = ["cuda.direct"], device = "cuda:0",
      grid = array<i64: 2, 1, 1>, implementation = "join.direct", kernel = @join,
      operandSegmentSizes = array<i32: 4, 1>, shared_memory = 0 : i64
    } : (memref<257xf32>, memref<257xf32>, memref<257xf32>, i32, !ckl.token) -> !ckl.token

    "ckl.free"(%temporary_a) : (memref<257xf32>) -> ()
    "ckl.free"(%output_a) : (memref<257xf32>) -> ()
    "ckl.free"(%temporary_b) : (memref<257xf32>) -> ()
    "ckl.free"(%output_b) : (memref<257xf32>) -> ()
    func.return
  }
}

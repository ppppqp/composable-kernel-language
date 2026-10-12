module {
  // Non-multiple-of-block-size buffers exercise explicit bounds guarding.
  func.func @update(%input: memref<257xf32>, %output: memref<257xf32>)
      attributes {ckl.kernel} {
    %i = "ckl.global_id"() {axis = 0 : i32} : () -> index
    %count = arith.constant 257 : index
    %in_bounds = arith.cmpi ult, %i, %count : index
    scf.if %in_bounds {
      %value = "ckl.load_element"(%input, %i) : (memref<257xf32>, index) -> f32
      %one = arith.constant 1.0 : f32
      %updated = arith.addf %value, %one : f32
      "ckl.store_element"(%updated, %output, %i) : (f32, memref<257xf32>, index) -> ()
    }
    func.return
  }

  func.func @step(%input: memref<257xf32>, %output: memref<257xf32>) {
    %temporary = "ckl.alloc"() : () -> memref<257xf32>
    %first = "ckl.dispatch"(%input, %temporary) {
      artifact = "update.cubin", implementation = "update.cuda",
      kernel = @update, device = "cuda:0", capabilities = ["cuda.direct"],
      grid = array<i64: 2, 1, 1>, block = array<i64: 256, 1, 1>,
      shared_memory = 0 : i64, operandSegmentSizes = array<i32: 2, 0>
    } : (memref<257xf32>, memref<257xf32>) -> !ckl.token
    // No explicit token dependency: the temporary's write/read supplies the edge.
    %second = "ckl.dispatch"(%temporary, %output) {
      artifact = "update.cubin", implementation = "update.cuda",
      kernel = @update, device = "cuda:0", capabilities = ["cuda.direct"],
      grid = array<i64: 2, 1, 1>, block = array<i64: 256, 1, 1>,
      shared_memory = 0 : i64, operandSegmentSizes = array<i32: 2, 0>
    } : (memref<257xf32>, memref<257xf32>) -> !ckl.token
    "ckl.free"(%temporary) : (memref<257xf32>) -> ()
    func.return
  }

  // All device coordinates are pure; a structured loop retains indexed effects.
  func.func @coordinates(%buffer: memref<2x3xi32>) attributes {ckl.kernel} {
    %thread = "ckl.thread_id"() {axis = 0 : i32} : () -> index
    %block = "ckl.block_id"() {axis = 1 : i32} : () -> index
    %width = "ckl.block_dim"() {axis = 2 : i32} : () -> index
    %grid = "ckl.grid_dim"() {axis = 0 : i32} : () -> index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    scf.for %row = %zero to %two step %one {
      %value = "ckl.load_element"(%buffer, %row, %one)
        : (memref<2x3xi32>, index, index) -> i32
      "ckl.store_element"(%value, %buffer, %row, %one)
        : (i32, memref<2x3xi32>, index, index) -> ()
    }
    func.return
  }
}

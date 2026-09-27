module {
  func.func @leaf(%src: memref<16xf32>, %dst: memref<16xf32>) {
    %value = "ckl.load"(%src) : (memref<16xf32>) -> f32
    "ckl.store"(%value, %dst) : (f32, memref<16xf32>) -> ()
    func.return
  }

  func.func @caller(%input: memref<16xf32>, %output: memref<16xf32>) {
    func.call @leaf(%input, %output) : (memref<16xf32>, memref<16xf32>) -> ()
    func.return
  }

  func.func @through_view(%resource: memref<16xf32>) {
    %view = "ckl.view"(%resource) : (memref<16xf32>) -> memref<16xf32>
    %value = "ckl.load"(%view) : (memref<16xf32>) -> f32
    "ckl.store"(%value, %view) : (f32, memref<16xf32>) -> ()
    func.return
  }

  func.func @private_allocation() {
    %buffer = "ckl.alloc"() : () -> memref<16xf32>
    %value = "ckl.load"(%buffer) : (memref<16xf32>) -> f32
    "ckl.store"(%value, %buffer) : (f32, memref<16xf32>) -> ()
    "ckl.free"(%buffer) : (memref<16xf32>) -> ()
    func.return
  }

  func.func @upstream_interfaces(%src: memref<16xf32>, %dst: memref<16xf32>) {
    memref.copy %src, %dst : memref<16xf32> to memref<16xf32>
    func.return
  }
}

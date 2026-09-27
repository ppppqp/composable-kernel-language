module {
  func.func @unknown(%resource: memref<16xf32>) {
    "foreign.opaque"(%resource) : (memref<16xf32>) -> ()
    func.return
  }
}

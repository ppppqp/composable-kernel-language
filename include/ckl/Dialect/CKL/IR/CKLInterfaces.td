#ifndef CKL_DIALECT_CKL_IR_CKLINTERFACES_TD
#define CKL_DIALECT_CKL_IR_CKLINTERFACES_TD

include "mlir/IR/OpBase.td"

def AccessRegionOpInterface : OpInterface<"AccessRegionOpInterface"> {
  let cppNamespace = "::mlir::ckl";
  let description = [{
    Refines a memory effect to whole, unknown, or a dictionary containing static
    byte offset and length relative to the accessed operand. Indexed accesses
    with constant indices expose one element; other indexed accesses cover the
    operand. Analysis composes this region through supported contiguous subviews.
  }];
  let methods = [
    InterfaceMethod<
      "Returns whole, unknown, or a byte interval relative to the accessed operand.",
      "::mlir::Attribute", "getAccessRegion",
      (ins "::mlir::Value":$resource), [{}], [{
        return ::mlir::ckl::getOperationAccessRegion($_op, resource);
      }]>
  ];
}

def DispatchOpInterface : OpInterface<"DispatchOpInterface"> {
  let cppNamespace = "::mlir::ckl";
  let description = [{
    Identifies a schedulable CKL kernel invocation for internal analysis and
    executable lowering. Implementations expose a kernel symbol, ordered arguments, and
    explicit completion dependencies.
  }];
  let methods = [
    InterfaceMethod<"Returns the referenced kernel symbol, including any nested module path.",
      "::mlir::SymbolRefAttr", "getKernelSymbol">,
    InterfaceMethod<"Returns ordered kernel arguments.",
      "::mlir::OperandRange", "getDispatchArguments">,
    InterfaceMethod<"Returns three grid dimensions.",
      "::mlir::DenseI64ArrayAttr", "getGridDimensions">,
    InterfaceMethod<"Returns three block dimensions.",
      "::mlir::DenseI64ArrayAttr", "getBlockDimensions">,
    InterfaceMethod<"Returns dynamic shared-memory bytes.",
      "::mlir::IntegerAttr", "getDynamicSharedMemory">,
    InterfaceMethod<"Returns the target device identity.",
      "::mlir::StringAttr", "getTargetDevice">,
    InterfaceMethod<"Returns required target capabilities.",
      "::mlir::ArrayAttr", "getRequiredCapabilities">,
    InterfaceMethod<"Returns a stable implementation identity.",
      "::mlir::StringAttr", "getImplementationIdentity">,
    InterfaceMethod<"Returns the compiled artifact identity consumed by the executor.",
      "::mlir::StringAttr", "getArtifactIdentity">,
    InterfaceMethod<"Returns the CKL backend physical argument ABI, or null if unresolved.",
      "::mlir::StringAttr", "getArgumentABI">,
    InterfaceMethod<"Returns explicit SSA completion dependencies.",
      "::mlir::OperandRange", "getExplicitDependencies">
  ];
}

#endif

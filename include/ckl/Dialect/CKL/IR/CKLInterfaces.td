#ifndef CKL_DIALECT_CKL_IR_CKLINTERFACES_TD
#define CKL_DIALECT_CKL_IR_CKLINTERFACES_TD

include "mlir/IR/OpBase.td"

def AccessRegionOpInterface : OpInterface<"AccessRegionOpInterface"> {
  let cppNamespace = "::mlir::ckl";
  let description = [{
    Optionally refines a memory effect to a region of its resource. Milestone 1
    supports the conservative `whole` region; producers may override the method
    as more precise region representations are introduced.
  }];
  let methods = [
    InterfaceMethod<
      "Returns `whole` or `unknown` for an accessed resource.",
      "::mlir::StringAttr", "getAccessRegion",
      (ins "::mlir::Value":$resource), [{}], [{
        return ::mlir::StringAttr::get($_op->getContext(), "whole");
      }]>
  ];
}

def DispatchOpInterface : OpInterface<"DispatchOpInterface"> {
  let cppNamespace = "::mlir::ckl";
  let description = [{
    Identifies a schedulable kernel invocation independently of its producer
    dialect. Implementations expose a kernel symbol, ordered arguments, and
    explicit completion dependencies.
  }];
  let methods = [
    InterfaceMethod<"Returns the referenced kernel symbol.",
      "::mlir::FlatSymbolRefAttr", "getKernelSymbol">,
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
    InterfaceMethod<"Returns explicit symbolic completion dependencies.",
      "::mlir::ArrayAttr", "getExplicitDependencies">
  ];
}

#endif

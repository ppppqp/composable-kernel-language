#ifndef CKL_DIALECT_CKL_IR_CKLOPS_TD
#define CKL_DIALECT_CKL_IR_CKLOPS_TD

include "ckl/Dialect/CKL/IR/CKLDialect.td"
include "ckl/Dialect/CKL/IR/CKLInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def CKL_AllocOp : CKL_Op<"alloc"> {
  let summary = "allocates compiler-owned device memory";
  let results = (outs Res<AnyMemRef, "", [MemAlloc<DefaultResource, 0, FullEffect>]>:$result);
}

def CKL_FreeOp : CKL_Op<"free"> {
  let summary = "releases compiler-owned device memory";
  let arguments = (ins Arg<AnyMemRef, "", [MemFreeAt<0, FullEffect>]>:$resource);
}

def CKL_LoadOp : CKL_Op<"load", [AccessRegionOpInterface]> {
  let summary = "reads a whole device resource";
  let arguments = (ins Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$resource);
  let results = (outs AnyType:$value);
  let hasVerifier = 1;
}

def CKL_StoreOp : CKL_Op<"store", [AccessRegionOpInterface]> {
  let summary = "writes a whole device resource";
  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$resource);
  let hasVerifier = 1;
}

def CKL_AtomicRMWOp : CKL_Op<"atomic_rmw", [AccessRegionOpInterface]> {
  let summary = "atomically reads and writes a whole device resource";
  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>,
                                                    MemWriteAt<1, FullEffect>]>:$resource,
                       StrAttr:$ordering,
                       StrAttr:$scope);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def CKL_SyncOp : CKL_Op<"sync"> {
  let summary = "orders accesses to the supplied resources";
  let arguments = (ins Arg<Variadic<AnyMemRef>, "", [MemReadAt<0, FullEffect>,
                                                       MemWriteAt<1, FullEffect>]>:$resources,
                       StrAttr:$scope);
  let hasVerifier = 1;
}

def CKL_ViewOp : CKL_Op<"view", [Pure, ViewLikeOpInterface,
                                   SameOperandsAndResultType]> {
  let summary = "creates an alias retaining its source resource identity";
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def CKL_DispatchOp : CKL_Op<"dispatch", [
    DeclareOpInterfaceMethods<DispatchOpInterface>]> {
  let summary = "describes a schedulable GPU kernel invocation";
  let arguments = (ins Variadic<AnyType>:$arguments,
                       FlatSymbolRefAttr:$kernel,
                       DenseI64ArrayAttr:$grid,
                       DenseI64ArrayAttr:$block,
                       I64Attr:$shared_memory,
                       StrAttr:$device,
                       ArrayAttr:$capabilities,
                       StrAttr:$implementation,
                       ArrayAttr:$after);
  let hasVerifier = 1;
}

#endif

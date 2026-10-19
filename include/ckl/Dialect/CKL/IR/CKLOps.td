#ifndef CKL_DIALECT_CKL_IR_CKLOPS_TD
#define CKL_DIALECT_CKL_IR_CKLOPS_TD

include "ckl/Dialect/CKL/IR/CKLDialect.td"
include "ckl/Dialect/CKL/IR/CKLInterfaces.td"
include "ckl/Dialect/CKL/IR/CKLTypes.td"
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

def CKL_LoadElementOp : CKL_Op<"load_element", [AccessRegionOpInterface]> {
  let summary = "loads one indexed buffer element inside a CKL kernel";
  let description = [{
    Indices are zero-based and must be in bounds at execution. There is one
    index operand per ranked memref dimension. Dynamic bounds checks are the
    kernel's responsibility; effects conservatively cover the whole buffer.
  }];
  let arguments = (ins Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$resource,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$value);
  let hasVerifier = 1;
}

def CKL_StoreElementOp : CKL_Op<"store_element", [AccessRegionOpInterface]> {
  let summary = "stores one indexed buffer element inside a CKL kernel";
  let description = [{
    Uses the same indexing and bounds contract as load_element. Effects
    conservatively cover the whole buffer.
  }];
  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$resource,
                       Variadic<Index>:$indices);
  let hasVerifier = 1;
}

class CKL_DeviceIndexOp<string mnemonic> : CKL_Op<mnemonic, [Pure]> {
  let arguments = (ins I32Attr:$axis);
  let results = (outs Index:$value);
  let hasVerifier = 1;
}

def CKL_ThreadIdOp : CKL_DeviceIndexOp<"thread_id"> {
  let summary = "zero-based thread coordinate within a block on axis 0, 1, or 2";
}
def CKL_BlockIdOp : CKL_DeviceIndexOp<"block_id"> {
  let summary = "zero-based block coordinate within a grid on axis 0, 1, or 2";
}
def CKL_BlockDimOp : CKL_DeviceIndexOp<"block_dim"> {
  let summary = "number of threads per block on axis 0, 1, or 2";
}
def CKL_GridDimOp : CKL_DeviceIndexOp<"grid_dim"> {
  let summary = "number of blocks per grid on axis 0, 1, or 2";
}
def CKL_GlobalIdOp : CKL_DeviceIndexOp<"global_id"> {
  let summary = "block_id * block_dim + thread_id on axis 0, 1, or 2";
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

// TODO:
def CKL_ViewOp : CKL_Op<"view", [Pure, ViewLikeOpInterface,
                                   SameOperandsAndResultType]> {
  let summary = "creates an alias retaining its source resource identity";
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def CKL_DispatchOp : CKL_Op<"dispatch", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<DispatchOpInterface, ["getArgumentABI"]>]> {
  let summary = "describes a schedulable GPU kernel invocation";
  let arguments = (ins Variadic<AnyType>:$arguments,
                       Variadic<CKL_TokenType>:$dependencies,
                       FlatSymbolRefAttr:$kernel,
                       DenseI64ArrayAttr:$grid,
                       DenseI64ArrayAttr:$block,
                       I64Attr:$shared_memory,
                       StrAttr:$device,
                       ArrayAttr:$capabilities,
                       OptionalAttr<StrAttr>:$artifact,
                       StrAttr:$implementation);
  let results = (outs CKL_TokenType:$completion);
  let hasVerifier = 1;
}

def CKL_GraphOp : CKL_Op<"graph", [SingleBlock, NoTerminator,
                                    IsolatedFromAbove]> {
  let summary = "normalized effect-derived orchestration graph";
  let arguments = (ins FlatSymbolRefAttr:$source,
                       StrAttr:$name,
                       StrAttr:$dot);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def CKL_GraphNodeOp : CKL_Op<"graph_node", [HasParent<"GraphOp">]> {
  let summary = "resource or dispatch node in a normalized graph";
  let arguments = (ins I64Attr:$id,
                       StrAttr:$kind,
                       OptionalAttr<SymbolRefAttr>:$kernel,
                       ArrayAttr:$accesses,
                       BoolAttr:$unknown,
                       BoolAttr:$synchronizes,
                       ArrayAttr:$ordering_scopes,
                       StrAttr:$source);
}

def CKL_GraphEdgeOp : CKL_Op<"graph_edge", [HasParent<"GraphOp">]> {
  let summary = "dependency edge with one or more provenance reasons";
  let arguments = (ins I64Attr:$from,
                       I64Attr:$to,
                       ArrayAttr:$reasons);
}

#endif

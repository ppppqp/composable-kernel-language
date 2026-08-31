#ifndef CKL_DIALECT_CKLNVIDIA_IR_CKLNVIDIAOPS_TD
#define CKL_DIALECT_CKLNVIDIA_IR_CKLNVIDIAOPS_TD

include "CKLNVIDIADialect.td"
include "../../CKL/IR/CKLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CKLNVIDIA_MmaSyncOp : CKLNVIDIA_Op<"mma_sync", [Pure]> {
  let summary = "Selected warp-level NVIDIA MMA implementation";
  let description = [{
    Represents a selected m16n8k16 f16/f32 MMA realization while operands are
    still CKL logical tiles. Fragment packing and NVGPU lowering happen later.
  }];
  let arguments = (ins CKL_TileType:$lhs, CKL_TileType:$rhs, CKL_TileType:$acc);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def CKLNVIDIA_PackFragmentOp : CKLNVIDIA_Op<"pack_fragment", [Pure]> {
  let summary = "Pack a distributed CKL tile into one lane's MMA registers";
  let arguments = (ins CKL_TileType:$tile, StrAttr:$role);
  let results = (outs AnyVectorOfAnyRank:$fragment);
  let assemblyFormat = "$tile `role` `=` $role attr-dict `:` type($tile) `->` type($fragment)";
  let hasVerifier = 1;
}

def CKLNVIDIA_UnpackFragmentOp : CKLNVIDIA_Op<"unpack_fragment", [Pure]> {
  let summary = "Reconstruct a distributed CKL tile from one lane's MMA registers";
  let arguments = (ins AnyVectorOfAnyRank:$fragment, StrAttr:$role);
  let results = (outs CKL_TileType:$tile);
  let assemblyFormat = "$fragment `role` `=` $role attr-dict `:` type($fragment) `->` type($tile)";
  let hasVerifier = 1;
}

#endif

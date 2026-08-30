#ifndef CKL_DIALECT_CKL_IR_CKLOPS_TD
#define CKL_DIALECT_CKL_IR_CKLOPS_TD

include "CKLTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Represents an unresolved producer/consumer boundary
def CKL_ComposeOp : CKL_Op<"compose", [Pure, SameOperandsAndResultType]> {
  let summary = "An unresolved producer/consumer composition boundary";
  let arguments = (ins CKL_TileType:$input, CKL_DistributionAttr:$source,
                       CKL_DistributionAttr:$target,
                       DefaultValuedAttr<I64Attr, "64">:$subgroup_size);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$input `source` `=` qualified($source) `target` `=` qualified($target) `attributes` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def CKL_ConvertLayoutOp : CKL_Op<"convert_layout", [Pure, SameOperandsAndResultType]> {
  let summary = "An explicit semantic layout conversion";
  let arguments = (ins CKL_TileType:$input, CKL_DistributionAttr:$source,
                       CKL_DistributionAttr:$target, StrAttr:$kind);
  let results = (outs CKL_TileType:$result);
  let assemblyFormat = "$input `source` `=` qualified($source) `target` `=` qualified($target) `kind` `=` $kind `attributes` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

#endif

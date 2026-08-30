#ifndef CKL_DIALECT_CKL_IR_CKLTYPES_TD
#define CKL_DIALECT_CKL_IR_CKLTYPES_TD

include "CKLAttrs.td"

def CKL_TileType : CKL_Type<"Tile", "tile"> {
  let summary = "A logical tile with an element type and index space";
  let parameters = (ins "::mlir::Type":$elementType, "::mlir::ckl::SpaceAttr":$space);
  let assemblyFormat = "`<` $elementType `,` qualified($space) `>`";
  let genVerifyDecl = 1;
}

#endif

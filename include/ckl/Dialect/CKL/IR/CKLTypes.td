#ifndef CKL_DIALECT_CKL_IR_CKLTYPES_TD
#define CKL_DIALECT_CKL_IR_CKLTYPES_TD

include "ckl/Dialect/CKL/IR/CKLDialect.td"
include "mlir/IR/AttrTypeBase.td"

class CKL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<CKL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def CKL_TokenType : CKL_Type<"Token", "token"> {
  let summary = "completion token produced by a dispatch";
}

#endif

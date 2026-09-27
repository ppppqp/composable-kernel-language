#ifndef CKL_DIALECT_CKL_IR_CKLDIALECT_TD
#define CKL_DIALECT_CKL_IR_CKLDIALECT_TD

include "mlir/IR/OpBase.td"

def CKL_Dialect : Dialect {
  let name = "ckl";
  let cppNamespace = "::mlir::ckl";
  let summary = "Effect-derived GPU kernel orchestration";
}

class CKL_Op<string mnemonic, list<Trait> traits = []>
    : Op<CKL_Dialect, mnemonic, traits>;

#endif

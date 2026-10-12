#ifndef CKL_DIALECT_CKL_IR_CKLDIALECT_TD
#define CKL_DIALECT_CKL_IR_CKLDIALECT_TD

include "mlir/IR/OpBase.td"

def CKL_Dialect : Dialect {
  let name = "ckl";
  let cppNamespace = "::mlir::ckl";
  let summary = "CKL GPU kernels and effect-derived programs";
  let useDefaultTypePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class CKL_Op<string mnemonic, list<Trait> traits = []>
    : Op<CKL_Dialect, mnemonic, traits>;

#endif

#ifndef CKL_DIALECT_CKL_IR_CKLDIALECT_TD
#define CKL_DIALECT_CKL_IR_CKLDIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def CKL_Dialect : Dialect {
  let name = "ckl";
  let cppNamespace = "::mlir::ckl";
  let summary = "Layout-aware GPU tile composition";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

class CKL_Attr<string name, string attrMnemonic> : AttrDef<CKL_Dialect, name> {
  let mnemonic = attrMnemonic;
}
class CKL_Type<string name, string typeMnemonic> : TypeDef<CKL_Dialect, name> {
  let mnemonic = typeMnemonic;
}
class CKL_Op<string mnemonic, list<Trait> traits = []>
    : Op<CKL_Dialect, mnemonic, traits>;

#endif

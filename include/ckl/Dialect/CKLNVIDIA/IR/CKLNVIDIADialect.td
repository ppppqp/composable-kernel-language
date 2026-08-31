#ifndef CKL_DIALECT_CKLNVIDIA_IR_CKLNVIDIADIALECT_TD
#define CKL_DIALECT_CKLNVIDIA_IR_CKLNVIDIADIALECT_TD

include "mlir/IR/OpBase.td"

def CKLNVIDIA_Dialect : Dialect {
  let name = "ckl_nvidia";
  let cppNamespace = "::mlir::ckl::nvidia";
  let summary = "Selected NVIDIA implementations for CKL tasks";
}

class CKLNVIDIA_Op<string mnemonic, list<Trait> traits = []>
    : Op<CKLNVIDIA_Dialect, mnemonic, traits>;

#endif

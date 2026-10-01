#ifndef CKL_DIALECT_EXEC_IR_CKLEXECDIALECT_TD
#define CKL_DIALECT_EXEC_IR_CKLEXECDIALECT_TD

include "mlir/IR/OpBase.td"

def CKLExec_Dialect : Dialect {
  let name = "ckl_exec";
  let cppNamespace = "::mlir::ckl::exec";
  let summary = "Backend-ready CKL execution plans";
}

class CKLExec_Op<string mnemonic, list<Trait> traits = []>
    : Op<CKLExec_Dialect, mnemonic, traits>;

#endif

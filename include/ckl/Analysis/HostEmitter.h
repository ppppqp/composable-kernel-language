#ifndef CKL_ANALYSIS_HOSTEMITTER_H
#define CKL_ANALYSIS_HOSTEMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace mlir::ckl {

/// Emit C++ builders from verified CUDA or ROCm ckl_exec plans.
LogicalResult emitHostBuilders(ModuleOp module, llvm::raw_ostream &output);

} // namespace mlir::ckl

#endif

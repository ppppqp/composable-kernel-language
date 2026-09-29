#ifndef CKL_ANALYSIS_NVIDIAHOSTEMITTER_H
#define CKL_ANALYSIS_NVIDIAHOSTEMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace mlir::ckl {

/// Emit C++ builders for statically planned NVIDIA graphs using the direct-pointer kernel ABI.
LogicalResult emitNvidiaHostBuilders(ModuleOp module, llvm::raw_ostream &output);

} // namespace mlir::ckl

#endif

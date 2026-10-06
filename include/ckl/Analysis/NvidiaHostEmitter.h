#ifndef CKL_ANALYSIS_NVIDIAHOSTEMITTER_H
#define CKL_ANALYSIS_NVIDIAHOSTEMITTER_H

#include "ckl/Analysis/HostEmitter.h"

namespace mlir::ckl {

/// Compatibility entry point that accepts only CUDA plans.
LogicalResult emitNvidiaHostBuilders(ModuleOp module, llvm::raw_ostream &output);

} // namespace mlir::ckl

#endif

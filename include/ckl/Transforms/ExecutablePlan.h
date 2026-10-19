#ifndef CKL_TRANSFORMS_EXECUTABLEPLAN_H
#define CKL_TRANSFORMS_EXECUTABLEPLAN_H

namespace mlir::ckl {

/// Register graph-to-plan and executable-to-LLVM transformations.
void registerCKLExecutablePlanPasses();

/// Register lowering of a dedicated executable host module to the LLVM dialect.
void registerCKLExecToLLVMPass();

} // namespace mlir::ckl

#endif

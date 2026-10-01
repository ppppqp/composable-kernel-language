#ifndef CKL_TRANSFORMS_EXECUTABLEPLAN_H
#define CKL_TRANSFORMS_EXECUTABLEPLAN_H

namespace mlir::ckl {

/// Register transformations from analyzed orchestration graphs to executable plans.
void registerCKLExecutablePlanPasses();

} // namespace mlir::ckl

#endif

#ifndef CKL_ANALYSIS_EFFECTANALYSIS_H
#define CKL_ANALYSIS_EFFECTANALYSIS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace ckl {

/// Recompute and materialize effect summaries for every function in a module.
LogicalResult deriveEffectSummaries(ModuleOp module, bool strict = true);

/// Register CKL semantic analysis and verification passes.
void registerCKLPasses();

/// Register graph-construction passes. Called by registerCKLPasses.
void registerCKLGraphPasses();

} // namespace ckl
} // namespace mlir

#endif

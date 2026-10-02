#ifndef CKL_ANALYSIS_GRAPHALGORITHMS_H
#define CKL_ANALYSIS_GRAPHALGORITHMS_H

#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::ckl {

using DispatchDependencyMap = llvm::DenseMap<int64_t, llvm::SmallVector<int64_t>>;

/// Return the transitive reduction among dispatch nodes. Paths through lifetime
/// nodes are retained as dependencies between their surrounding dispatches.
DispatchDependencyMap getReducedDispatchDependencies(GraphOp graph);

/// Reject graph features that the current static executable plan cannot preserve. Analysis graphs
/// may contain these features for diagnostics, but neither executable lowering nor host emission
/// may silently change their semantics.
LogicalResult verifyStaticExecutionSubset(GraphOp graph, StringRef consumer);

} // namespace mlir::ckl

#endif

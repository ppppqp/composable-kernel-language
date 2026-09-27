#ifndef CKL_ANALYSIS_EFFECTANALYSIS_H
#define CKL_ANALYSIS_EFFECTANALYSIS_H

#include "mlir/IR/Value.h"

namespace mlir {
namespace ckl {

enum class WholeBufferAliasResult { NoAlias, MayAlias, MustAlias };

/// Strip view-like operations and return the underlying whole-buffer resource.
Value getBaseResource(Value value);

/// Conservative whole-buffer alias relation used by orchestration analysis.
WholeBufferAliasResult aliasWholeBuffers(Value lhs, Value rhs);

/// Register CKL semantic analysis and verification passes.
void registerCKLPasses();

} // namespace ckl
} // namespace mlir

#endif

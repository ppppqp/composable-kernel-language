#ifndef CKL_CORE_MEMORYEFFECTS_H
#define CKL_CORE_MEMORYEFFECTS_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace ckl {

enum AccessBits : unsigned {
  NoAccess = 0,
  Read = 1u << 0,
  Write = 1u << 1,
  Allocate = 1u << 2,
  Free = 1u << 3,
};

/// Convert individual MLIR memory effects and summary names to/from CKL bits.
unsigned classifyMemoryEffect(MemoryEffects::Effect *effect);
unsigned parseAccessEffect(StringRef effect);
StringRef stringifyAccessEffect(unsigned effect);

/// Return whether two accesses require source-order serialization.
bool hasConflictingAccesses(unsigned lhs, unsigned rhs);

/// Return true when `value` is produced by an operation that reports allocating it.
bool isAllocationValue(Value value);

/// Strip view-like operations and return the underlying whole-buffer resource.
Value getBaseResource(Value value);

enum class WholeBufferAliasResult { NoAlias, MayAlias, MustAlias };

/// Conservative whole-buffer alias relation used by orchestration analysis.
WholeBufferAliasResult aliasWholeBuffers(Value lhs, Value rhs);

} // namespace ckl
} // namespace mlir

#endif

#include "ckl/Core/MemoryEffects.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::ckl;

unsigned mlir::ckl::classifyMemoryEffect(MemoryEffects::Effect *effect) {
  if (isa<MemoryEffects::Read>(effect))
    return Read;
  if (isa<MemoryEffects::Write>(effect))
    return Write;
  if (isa<MemoryEffects::Allocate>(effect))
    return Allocate;
  if (isa<MemoryEffects::Free>(effect))
    return Free;
  return NoAccess;
}

unsigned mlir::ckl::parseAccessEffect(StringRef effect) {
  return StringSwitch<unsigned>(effect)
      .Case("read", Read)
      .Case("write", Write)
      .Case("allocate", Allocate)
      .Case("free", Free)
      .Default(NoAccess);
}

StringRef mlir::ckl::stringifyAccessEffect(unsigned effect) {
  switch (effect) {
  case Read:
    return "read";
  case Write:
    return "write";
  case Allocate:
    return "allocate";
  case Free:
    return "free";
  default:
    return {};
  }
}

bool mlir::ckl::hasConflictingAccesses(unsigned lhs, unsigned rhs) {
  constexpr unsigned nonRead = Write | Allocate | Free;
  return (lhs & nonRead) || (rhs & nonRead);
}

bool mlir::ckl::isAllocationValue(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  auto interface = dyn_cast<MemoryEffectOpInterface>(definition);
  if (!interface)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  return llvm::any_of(effects, [&](const MemoryEffects::EffectInstance &effect) {
    return isa<MemoryEffects::Allocate>(effect.getEffect()) && effect.getValue() == value;
  });
}

Value mlir::ckl::getBaseResource(Value value) {
  SmallPtrSet<Operation *, 8> visited;
  while (Operation *definition = value.getDefiningOp()) {
    if (!visited.insert(definition).second)
      break;
    auto view = dyn_cast<ViewLikeOpInterface>(definition);
    if (!view)
      break;
    value = view.getViewSource();
  }
  return value;
}

WholeBufferAliasResult mlir::ckl::aliasWholeBuffers(Value lhs, Value rhs) {
  lhs = getBaseResource(lhs);
  rhs = getBaseResource(rhs);
  if (lhs == rhs)
    return WholeBufferAliasResult::MustAlias;
  if (isAllocationValue(lhs) || isAllocationValue(rhs))
    return WholeBufferAliasResult::NoAlias;
  return WholeBufferAliasResult::MayAlias;
}

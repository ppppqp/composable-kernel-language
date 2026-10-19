#include "ckl/Core/MemoryEffects.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include <limits>

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
  /*
  Alias analysis for CKL resources. This function traverses the chain of view-like operations to
  find the base resource that a given value ultimately refers to. It uses a SmallPtrSet to track
  visited operations and avoid infinite loops in case of cycles. The function returns the base
  resource value, which is the original allocation or resource that the input value is derived from.
  */
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

namespace {
DictionaryAttr byteRange(MLIRContext *context, int64_t offset, int64_t length) {
  Builder b(context);
  return b.getDictionaryAttr({b.getNamedAttr("offset", b.getI64IntegerAttr(offset)),
                              b.getNamedAttr("length", b.getI64IntegerAttr(length))});
}
bool interval(Attribute attr, int64_t &offset, int64_t &length) {
  auto range = dyn_cast_or_null<DictionaryAttr>(attr);
  if (!range)
    return false;
  auto o = range.getAs<IntegerAttr>("offset"), l = range.getAs<IntegerAttr>("length");
  if (!o || !l || o.getInt() < 0 || l.getInt() < 0 ||
      l.getInt() > std::numeric_limits<int64_t>::max() - o.getInt())
    return false;
  offset = o.getInt();
  length = l.getInt();
  return true;
}
} // namespace
DictionaryAttr mlir::ckl::getStaticResourceRange(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
      !type.getElementType().isIntOrFloat())
    return {};
  int64_t width = type.getElementType().getIntOrFloatBitWidth();
  if (width % 8 || !width)
    return {};
  int64_t bytes = width / 8, elements = 1;
  for (int64_t size : type.getShape()) {
    if (size <= 0 || elements > std::numeric_limits<int64_t>::max() / size)
      return {};
    elements *= size;
  }
  if (elements > std::numeric_limits<int64_t>::max() / bytes)
    return {};
  int64_t offset = 0;
  if (Operation *op = value.getDefiningOp()) {
    if (op->getName().getStringRef() == "ckl.subview") {
      auto parent = getStaticResourceRange(op->getOperand(0));
      auto elementOffset = op->getAttrOfType<IntegerAttr>("offset");
      if (!parent || !elementOffset || elementOffset.getInt() < 0 ||
          elementOffset.getInt() > std::numeric_limits<int64_t>::max() / bytes)
        return {};
      int64_t local = elementOffset.getInt() * bytes;
      int64_t parentLength = parent.getAs<IntegerAttr>("length").getInt();
      if (local > parentLength || elements * bytes > parentLength - local)
        return {};
      offset = parent.getAs<IntegerAttr>("offset").getInt() + local;
    } else if (isa<ViewLikeOpInterface>(op)) {
      // Unmodeled views are not assumed to be zero-offset aliases.
      return {};
    }
  }
  return byteRange(value.getContext(), offset, elements * bytes);
}
Attribute mlir::ckl::getOperationAccessRegion(Operation *op, Value resource) {
  /*
  e.g. for ckl.load_element %r[%i, %j, %k], the access region is
  %i * stride[0] + %j * stride[1] + %k * stride[2] * sizeof(element)
  */
  auto type = dyn_cast<MemRefType>(resource.getType());
  StringRef name = op->getName().getStringRef();
  if (type && type.hasStaticShape() && type.getLayout().isIdentity() &&
      type.getElementType().isIntOrFloat() &&
      type.getElementType().getIntOrFloatBitWidth() % 8 == 0 &&
      (name == "ckl.load_element" || name == "ckl.store_element")) {
    unsigned begin = name == "ckl.load_element" ? 1 : 2;
    int64_t linear = 0;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      // Compute the linear index of the accessed element. If any of the indices are
      // not constant, or out of bounds, return "whole" to indicate that the entire
      // resource may be accessed.
      APInt index;
      if (!matchPattern(op->getOperand(begin + axis), m_ConstantInt(&index)))
        return StringAttr::get(op->getContext(), "whole");
      int64_t i = index.getSExtValue(), extent = type.getDimSize(axis);
      if (i < 0 || i >= extent || linear > (std::numeric_limits<int64_t>::max() - i) / extent)
        return StringAttr::get(op->getContext(), "whole");
      linear = linear * extent + i;
    }
    int64_t bytes = type.getElementType().getIntOrFloatBitWidth() / 8;
    if (bytes && linear <= std::numeric_limits<int64_t>::max() / bytes)
      return byteRange(op->getContext(), linear * bytes, bytes);
  }
  return StringAttr::get(op->getContext(), "whole");
}
Attribute mlir::ckl::composeAccessRegion(Value resource, Attribute relative) {
  /*
  Compose a relative access region with the static range of the resource to produce an absolute
  access region.
  */
  auto unknown = StringAttr::get(resource.getContext(), "unknown");
  if (relative == unknown)
    return unknown;
  auto range = getStaticResourceRange(resource);
  if (!range)
    return StringAttr::get(resource.getContext(), "whole");
  int64_t base = range.getAs<IntegerAttr>("offset").getInt();
  int64_t size = range.getAs<IntegerAttr>("length").getInt();
  int64_t offset, length;
  if (interval(relative, offset, length)) {
    if (offset > size || length > size - offset)
      return unknown;
    return byteRange(resource.getContext(), base + offset, length);
  }
  if (resource != getBaseResource(resource))
    return range;
  return relative;
}
Attribute mlir::ckl::mergeAccessRegions(Attribute lhs, Attribute rhs) {
  if (!lhs)
    return rhs;
  auto unknown = StringAttr::get(lhs.getContext(), "unknown");
  if (lhs == unknown || rhs == unknown)
    return unknown;
  int64_t a, n, b, m;
  if (!interval(lhs, a, n) || !interval(rhs, b, m))
    return StringAttr::get(lhs.getContext(), "whole");
  int64_t start = std::min(a, b), end = std::max(a + n, b + m);
  return byteRange(lhs.getContext(), start, end - start);
}
bool mlir::ckl::areDisjointAccessRegions(Attribute lhs, Attribute rhs) {
  int64_t a, n, b, m;
  return interval(lhs, a, n) && interval(rhs, b, m) && (a + n <= b || b + m <= a);
}

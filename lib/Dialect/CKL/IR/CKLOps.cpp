#include "ckl/Dialect/CKL/IR/CKLOps.h"

#include "ckl/Core/Layout/Distribution.h"

using namespace mlir;
using namespace mlir::ckl;

#define GET_OP_CLASSES
#include "ckl/Dialect/CKL/IR/CKLOps.cpp.inc"

namespace {
LogicalResult verifyBoundary(Operation *op, TileType type, DistributionAttr sourceAttr,
                             DistributionAttr targetAttr, std::optional<StringRef> expectedKind,
                             std::int64_t subgroupSize) {
  if (subgroupSize <= 0)
    return op->emitOpError("requires a positive subgroup size");
  try {
    auto logical = ::ckl::core::IndexSpace::deserialize(type.getSpace().getValue().str());
    auto source = ::ckl::core::deserializeDistribution(sourceAttr.getValue().str());
    auto target = ::ckl::core::deserializeDistribution(targetAttr.getValue().str());
    if (!source.tileSpace.sameShape(logical) || !target.tileSpace.sameShape(logical))
      return op->emitOpError("source and target distributions must match the logical tile space");
    auto plan = ::ckl::core::classifyConversion(source, target, subgroupSize);
    if (plan.kind == ::ckl::core::ConversionKind::Unsupported)
      return op->emitOpError("has unsupported layout boundary: ") << plan.reason;
    if (expectedKind && *expectedKind != ::ckl::core::toString(plan.kind))
      return op->emitOpError("conversion kind '") << *expectedKind
             << "' disagrees with CKLCore classification '" << ::ckl::core::toString(plan.kind)
             << "'";
  } catch (const std::exception &error) {
    return op->emitOpError("cannot verify layout boundary: ") << error.what();
  }
  return success();
}
} // namespace

LogicalResult ComposeOp::verify() {
  return verifyBoundary(*this, getInput().getType(), getSource(), getTarget(), std::nullopt,
                        getSubgroupSize());
}

LogicalResult ConvertLayoutOp::verify() {
  return verifyBoundary(*this, getInput().getType(), getSource(), getTarget(), getKind(), 64);
}

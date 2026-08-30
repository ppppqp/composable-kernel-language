#include "ckl/Dialect/CKL/IR/CKLOps.h"

#include "ckl/Core/Layout/Distribution.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

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

LogicalResult BindSymbolsOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result tile types must match");
  try {
    auto space = ::ckl::core::IndexSpace::deserialize(getInput().getType().getSpace().getValue().str());
    SmallVector<StringRef> expected;
    llvm::StringSet<> seen;
    for (const auto &axis : space.axes())
      if (axis.extentSymbol && seen.insert(*axis.extentSymbol).second)
        expected.push_back(seen.find(*axis.extentSymbol)->getKey());
    if (expected.size() != getBindings().size() || expected.size() != getSymbols().size())
      return emitOpError("requires exactly one binding for each unique symbolic extent");
    for (auto [index, name] : llvm::enumerate(getSymbols())) {
      auto string = mlir::dyn_cast<StringAttr>(name);
      if (!string || string.getValue() != expected[index])
        return emitOpError("symbol list must follow logical-space order; expected '")
               << expected[index] << "' at position " << index;
    }
  } catch (const std::exception &error) {
    return emitOpError("cannot inspect tile space: ") << error.what();
  }
  return success();
}

LogicalResult TaskOp::verify() {
  llvm::StringSet<> names;
  for (Attribute attribute : getAlternatives()) {
    auto dictionary = mlir::dyn_cast<DictionaryAttr>(attribute);
    auto name = dictionary ? dictionary.getAs<StringAttr>("name") : StringAttr{};
    if (!name)
      return emitOpError("each alternative must be a dictionary with a string 'name'");
    if (!names.insert(name.getValue()).second)
      return emitOpError("has duplicate alternative '") << name.getValue() << "'";
  }
  return success();
}

LogicalResult InvokeOp::verify() {
  auto task = SymbolTable::lookupNearestSymbolFrom<TaskOp>(*this, getCalleeAttr());
  if (!task)
    return emitOpError("references unknown task ") << getCallee();
  FunctionType type = task.getFunctionType();
  if (!llvm::equal(type.getInputs(), getInputs().getTypes()) ||
      !llvm::equal(type.getResults(), getResults().getTypes()))
    return emitOpError("operand/result types do not match task signature ") << type;
  if (auto selected = getAlternativeAttr()) {
    bool found = llvm::any_of(task.getAlternatives(), [&](Attribute attribute) {
      auto dictionary = mlir::cast<DictionaryAttr>(attribute);
      return dictionary.getAs<StringAttr>("name").getValue() == selected.getValue();
    });
    if (!found)
      return emitOpError("selects unknown alternative '") << selected.getValue() << "'";
  }
  return success();
}

LogicalResult ComposeOp::verify() {
  return verifyBoundary(*this, getInput().getType(), getSource(), getTarget(), std::nullopt,
                        getSubgroupSize());
}

LogicalResult ConvertLayoutOp::verify() {
  return verifyBoundary(*this, getInput().getType(), getSource(), getTarget(), getKind(),
                        getSubgroupSize());
}

#define CKL_VERIFY_SCHEDULED(OP, KIND)                                         \
  LogicalResult OP::verify() {                                                \
    return verifyBoundary(*this, getInput().getType(), getSource(), getTarget(), \
                          KIND, getSubgroupSize());                           \
  }
CKL_VERIFY_SCHEDULED(LocalPermuteOp, "local-permutation")
CKL_VERIFY_SCHEDULED(SubgroupExchangeOp, "subgroup-exchange")
CKL_VERIFY_SCHEDULED(SharedExchangeOp, "shared-memory-exchange")
CKL_VERIFY_SCHEDULED(GlobalExchangeOp, "global-memory-exchange")
#undef CKL_VERIFY_SCHEDULED

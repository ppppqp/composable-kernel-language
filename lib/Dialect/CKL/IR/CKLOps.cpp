#include "ckl/Dialect/CKL/IR/CKLOps.h"

#include "ckl/Core/Layout/Distribution.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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
  llvm::StringSet<> implementationIds;
  for (Attribute attribute : getAlternatives()) {
    auto dictionary = mlir::dyn_cast<DictionaryAttr>(attribute);
    auto name = dictionary ? dictionary.getAs<StringAttr>("name") : StringAttr{};
    if (!name)
      return emitOpError("each alternative must be a dictionary with a string 'name'");
    if (!names.insert(name.getValue()).second)
      return emitOpError("has duplicate alternative '") << name.getValue() << "'";
    if (Attribute value = dictionary.get("implementation_id")) {
      auto id = mlir::dyn_cast<StringAttr>(value);
      if (!id || id.getValue().empty())
        return emitOpError("alternative '") << name.getValue()
               << "' implementation_id must be a non-empty string";
      if (!implementationIds.insert(id.getValue()).second)
        return emitOpError("has duplicate implementation_id '") << id.getValue() << "'";
    }
    if (Attribute value = dictionary.get("implementation")) {
      auto reference = mlir::dyn_cast<FlatSymbolRefAttr>(value);
      if (!reference)
        return emitOpError("alternative '") << name.getValue()
               << "' implementation must be a flat symbol reference";
      Operation *implementation = SymbolTable::lookupNearestSymbolFrom(*this, reference);
      if (!implementation)
        return emitOpError("alternative '") << name.getValue()
               << "' references unknown implementation " << reference;
      auto callable = mlir::dyn_cast<FunctionOpInterface>(implementation);
      if (!callable)
        return emitOpError("alternative '") << name.getValue()
               << "' implementation is not function-like";
      if (callable.getFunctionType() != getFunctionType())
        return emitOpError("alternative '") << name.getValue()
               << "' implementation type " << callable.getFunctionType()
               << " does not match task type " << getFunctionType();
    }
    if (auto origin = dictionary.getAs<StringAttr>("origin")) {
      if (origin.getValue() != "user" && origin.getValue() != "compiler" &&
          origin.getValue() != "extension" && origin.getValue() != "library")
        return emitOpError("alternative '") << name.getValue()
               << "' has unknown origin '" << origin.getValue() << "'";
    }
    for (StringRef field : {"registers_per_thread", "shared_memory_bytes",
                            "estimated_execution_cost"})
      if (auto value = dictionary.getAs<IntegerAttr>(field); value && value.getInt() < 0)
        return emitOpError("alternative '") << name.getValue() << "' has negative " << field;
    if (auto capabilities = dictionary.getAs<ArrayAttr>("required_capabilities"))
      if (!llvm::all_of(capabilities, [](Attribute value) { return mlir::isa<StringAttr>(value); }))
        return emitOpError("alternative '") << name.getValue()
               << "' requires capabilities to be strings";
    if (auto resources = dictionary.getAs<ArrayAttr>("resources")) {
      for (Attribute attribute : resources) {
        auto resource = mlir::dyn_cast<DictionaryAttr>(attribute);
        auto resourceName = resource ? resource.getAs<StringAttr>("name") : StringAttr{};
        auto bytes = resource ? resource.getAs<IntegerAttr>("bytes") : IntegerAttr{};
        auto begin = resource ? resource.getAs<IntegerAttr>("begin") : IntegerAttr{};
        auto end = resource ? resource.getAs<IntegerAttr>("end") : IntegerAttr{};
        if (!resourceName || !bytes || !begin || !end)
          return emitOpError("alternative '") << name.getValue()
                 << "' resource requires name, bytes, begin, and end";
        if (bytes.getInt() < 0 || begin.getInt() < 0 || end.getInt() <= begin.getInt())
          return emitOpError("alternative '") << name.getValue()
                 << "' has invalid resource lifetime for '" << resourceName.getValue() << "'";
      }
    }
    if (auto effects = dictionary.getAs<ArrayAttr>("effects")) {
      for (Attribute attribute : effects) {
        auto effect = mlir::dyn_cast<DictionaryAttr>(attribute);
        auto kind = effect ? effect.getAs<StringAttr>("kind") : StringAttr{};
        auto resource = effect ? effect.getAs<StringAttr>("resource") : StringAttr{};
        auto stage = effect ? effect.getAs<IntegerAttr>("stage") : IntegerAttr{};
        if (!kind || !resource)
          return emitOpError("alternative '") << name.getValue()
                 << "' effect requires string kind and resource";
        if (kind.getValue() != "read" && kind.getValue() != "write" &&
            kind.getValue() != "consume" && kind.getValue() != "channel-put" &&
            kind.getValue() != "channel-get" && kind.getValue() != "channel-release")
          return emitOpError("alternative '") << name.getValue()
                 << "' has unknown effect kind '" << kind.getValue() << "'";
        if (stage && stage.getInt() < 0)
          return emitOpError("alternative '") << name.getValue()
                 << "' effect stage must be non-negative";
      }
    }
  }
  return success();
}

LogicalResult TaskComposeOp::verify() {
  if (getRegisterLimit() < 0 || getSharedMemoryLimit() < 0 || getSubgroupSize() <= 0)
    return emitOpError("requires non-negative resource limits and a positive subgroup size");
  if (!SymbolTable::lookupNearestSymbolFrom<TaskOp>(*this, getProducerAttr()))
    return emitOpError("references unknown producer task ") << getProducer();
  if (!SymbolTable::lookupNearestSymbolFrom<TaskOp>(*this, getConsumerAttr()))
    return emitOpError("references unknown consumer task ") << getConsumer();
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

namespace {

LogicalResult verifyTileMemoryBoundary(Operation *op, TileType tile, MemRefType memory,
                                       ValueRange offsets, DistributionAttr distribution) {
  if (memory.getElementType() != tile.getElementType())
    return op->emitOpError("memref and tile element types must match");
  if (static_cast<std::size_t>(memory.getRank()) != offsets.size())
    return op->emitOpError("requires one base offset per memref dimension");
  try {
    auto tileSpace = ::ckl::core::IndexSpace::deserialize(tile.getSpace().getValue().str());
    auto physical = ::ckl::core::deserializeDistribution(distribution.getValue().str());
    if (!tileSpace.sameShape(physical.tileSpace))
      return op->emitOpError("distribution tile space does not match the logical tile type");
    if (static_cast<std::size_t>(memory.getRank()) != tileSpace.rank())
      return op->emitOpError("memref rank must match logical tile rank");
    for (auto [dimension, axis] : llvm::enumerate(tileSpace.axes()))
      if (!memory.isDynamicDim(dimension) && axis.isStatic() &&
          memory.getDimSize(dimension) < axis.extent)
        return op->emitOpError("memref dimension is smaller than the logical tile");
  } catch (const std::exception &error) {
    return op->emitOpError("cannot inspect tile memory contract: ") << error.what();
  }
  return success();
}

} // namespace

LogicalResult LoadTileOp::verify() {
  return verifyTileMemoryBoundary(*this, getResult().getType(), getSource().getType(),
                                  getOffsets(), getDistribution());
}

LogicalResult StoreTileOp::verify() {
  return verifyTileMemoryBoundary(*this, getValue().getType(), getTarget().getType(),
                                  getOffsets(), getDistribution());
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
CKL_VERIFY_SCHEDULED(SharedStoreOp, "shared-memory-exchange")
CKL_VERIFY_SCHEDULED(WorkgroupBarrierOp, "shared-memory-exchange")
CKL_VERIFY_SCHEDULED(SharedLoadOp, "shared-memory-exchange")
CKL_VERIFY_SCHEDULED(GlobalStoreOp, "global-memory-exchange")
CKL_VERIFY_SCHEDULED(KernelBoundaryOp, "global-memory-exchange")
CKL_VERIFY_SCHEDULED(GlobalLoadOp, "global-memory-exchange")
#undef CKL_VERIFY_SCHEDULED

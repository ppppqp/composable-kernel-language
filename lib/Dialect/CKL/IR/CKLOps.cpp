#include "ckl/Dialect/CKL/IR/CKLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::ckl;

namespace {

Type getElementType(Value resource) {
  return cast<BaseMemRefType>(resource.getType()).getElementType();
}

LogicalResult verifyScope(Operation *operation, StringRef scope) {
  if (scope == "thread" || scope == "block" || scope == "device" || scope == "system")
    return success();
  return operation->emitOpError() << "has unsupported ordering scope '" << scope << "'";
}

LogicalResult verifyKernelContext(Operation *operation) {
  auto function = operation->getParentOfType<func::FuncOp>();
  if (!function || !function->hasAttrOfType<UnitAttr>("ckl.kernel"))
    return operation->emitOpError("requires an enclosing func.func marked ckl.kernel");
  return success();
}

LogicalResult verifyElementAccess(Operation *operation, Value resource, ValueRange indices,
                                 Type valueType) {
  if (failed(verifyKernelContext(operation)))
    return failure();
  auto buffer = cast<MemRefType>(resource.getType());
  if (indices.size() != static_cast<size_t>(buffer.getRank()))
    return operation->emitOpError("requires one index per buffer dimension");
  if (valueType != buffer.getElementType())
    return operation->emitOpError("value type must match the buffer element type");
  if (!isa<IntegerType, FloatType, IndexType>(valueType))
    return operation->emitOpError("requires a numeric buffer element type");
  for (auto [axis, index] : llvm::enumerate(indices)) {
    APInt constant;
    if (!matchPattern(index, m_ConstantInt(&constant)))
      continue;
    int64_t value = constant.getSExtValue();
    if (value < 0 || (!buffer.isDynamicDim(axis) && value >= buffer.getDimSize(axis)))
      return operation->emitOpError() << "constant index is out of bounds for dimension " << axis;
  }
  return success();
}

LogicalResult verifyDeviceIndex(Operation *operation, int64_t axis) {
  if (failed(verifyKernelContext(operation)))
    return failure();
  if (axis < 0 || axis > 2)
    return operation->emitOpError("requires axis 0, 1, or 2");
  return success();
}

} // namespace

LogicalResult LoadOp::verify() {
  if (getValue().getType() != getElementType(getResource()))
    return emitOpError("result type must match the resource element type");
  return success();
}

LogicalResult StoreOp::verify() {
  if (getValue().getType() != getElementType(getResource()))
    return emitOpError("value type must match the resource element type");
  return success();
}

LogicalResult LoadElementOp::verify() {
  return verifyElementAccess(*this, getResource(), getIndices(), getValue().getType());
}

LogicalResult StoreElementOp::verify() {
  return verifyElementAccess(*this, getResource(), getIndices(), getValue().getType());
}

LogicalResult ThreadIdOp::verify() { return verifyDeviceIndex(*this, getAxis()); }
LogicalResult BlockIdOp::verify() { return verifyDeviceIndex(*this, getAxis()); }
LogicalResult BlockDimOp::verify() { return verifyDeviceIndex(*this, getAxis()); }
LogicalResult GridDimOp::verify() { return verifyDeviceIndex(*this, getAxis()); }
LogicalResult GlobalIdOp::verify() { return verifyDeviceIndex(*this, getAxis()); }

LogicalResult AtomicRMWOp::verify() {
  Type elementType = getElementType(getResource());
  if (getValue().getType() != elementType || getResult().getType() != elementType)
    return emitOpError("value and result types must match the resource element type");
  if (getOrdering() != "relaxed" && getOrdering() != "acquire" && getOrdering() != "release" &&
      getOrdering() != "acq_rel" && getOrdering() != "seq_cst")
    return emitOpError() << "has unsupported atomic ordering '" << getOrdering() << "'";
  return verifyScope(getOperation(), getScope());
}

LogicalResult SyncOp::verify() { return verifyScope(getOperation(), getScope()); }

LogicalResult GraphOp::verify() {
  if (!SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getSourceAttr()))
    return emitOpError() << "references unknown source function " << getSourceAttr();
  if (getBody().empty())
    return emitOpError("requires a graph body block");

  llvm::DenseSet<uint64_t> nodeIds;
  for (Operation &operation : getBody().front()) {
    if (auto node = dyn_cast<GraphNodeOp>(operation)) {
      if (!nodeIds.insert(node.getId()).second)
        return node.emitOpError("has a duplicate node id");
      continue;
    }
    if (!isa<GraphEdgeOp>(operation))
      return operation.emitOpError("is not a graph node or edge");
  }
  for (GraphEdgeOp edge : getBody().front().getOps<GraphEdgeOp>()) {
    if (!nodeIds.contains(edge.getFrom()) || !nodeIds.contains(edge.getTo()))
      return edge.emitOpError("references an unknown graph node");
    if (edge.getFrom() >= edge.getTo())
      return edge.emitOpError("must preserve source program order");
    if (edge.getReasons().empty())
      return edge.emitOpError("requires at least one provenance reason");
  }
  return success();
}

SymbolRefAttr DispatchOp::getKernelSymbol() { return getKernelAttr(); }

OperandRange DispatchOp::getDispatchArguments() { return getArguments(); }

DenseI64ArrayAttr DispatchOp::getGridDimensions() { return getGridAttr(); }

DenseI64ArrayAttr DispatchOp::getBlockDimensions() { return getBlockAttr(); }

IntegerAttr DispatchOp::getDynamicSharedMemory() { return getSharedMemoryAttr(); }

StringAttr DispatchOp::getTargetDevice() { return getDeviceAttr(); }

ArrayAttr DispatchOp::getRequiredCapabilities() { return getCapabilities(); }

StringAttr DispatchOp::getImplementationIdentity() { return getImplementationAttr(); }

StringAttr DispatchOp::getArtifactIdentity() {
  return getArtifactAttr() ? getArtifactAttr() : getImplementationAttr();
}

StringAttr DispatchOp::getArgumentABI() {
  for (Attribute capability : getCapabilities()) {
    StringRef abi = cast<StringAttr>(capability).getValue();
    if (abi == "cuda.direct" || abi == "rocm.bare_ptr")
      return StringAttr::get(getContext(), abi);
  }
  return {};
}

OperandRange DispatchOp::getExplicitDependencies() { return getDependencies(); }

LogicalResult DispatchOp::verify() {
  auto enclosingFunction = (*this)->getParentOfType<func::FuncOp>();
  if (enclosingFunction && enclosingFunction->hasAttr("ckl.kernel"))
    return emitOpError("cannot launch a kernel from a ckl.kernel body");
  auto checkDimensions = [&](ArrayRef<int64_t> dimensions, StringRef name) -> LogicalResult {
    if (dimensions.size() != 3)
      return emitOpError() << "requires exactly three " << name << " dimensions";
    if (llvm::any_of(dimensions, [](int64_t value) { return value <= 0; }))
      return emitOpError() << "requires positive " << name << " dimensions";
    return success();
  };
  if (failed(checkDimensions(getGrid(), "grid")) ||
      failed(checkDimensions(getBlock(), "block")))
    return failure();
  if (getSharedMemoryAttr().getInt() < 0)
    return emitOpError("requires non-negative shared memory");
  if (getImplementation().empty())
    return emitOpError("requires a stable implementation identity");

  auto function =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getKernelAttr());
  if (!function)
    return emitOpError() << "references unknown kernel " << getKernelAttr();
  if (function.getNumResults() != 0)
    return emitOpError("referenced kernel must have no results");
  if (function.getNumArguments() != getArguments().size())
    return emitOpError() << "passes " << getArguments().size() << " arguments to a kernel with "
                         << function.getNumArguments() << " parameters";
  for (auto [actual, formal] : llvm::zip_equal(getArguments(), function.getArgumentTypes()))
    if (actual.getType() != formal)
      return emitOpError("argument type does not match referenced kernel");
  return success();
}

#define GET_OP_CLASSES
#include "ckl/Dialect/CKL/IR/CKLOps.cpp.inc"

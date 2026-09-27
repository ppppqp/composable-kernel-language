#include "ckl/Dialect/CKL/IR/CKLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

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

FlatSymbolRefAttr DispatchOp::getKernelSymbol() { return getKernelAttr(); }

OperandRange DispatchOp::getDispatchArguments() { return getArguments(); }

DenseI64ArrayAttr DispatchOp::getGridDimensions() { return getGridAttr(); }

DenseI64ArrayAttr DispatchOp::getBlockDimensions() { return getBlockAttr(); }

IntegerAttr DispatchOp::getDynamicSharedMemory() { return getSharedMemoryAttr(); }

StringAttr DispatchOp::getTargetDevice() { return getDeviceAttr(); }

ArrayAttr DispatchOp::getRequiredCapabilities() { return getCapabilities(); }

StringAttr DispatchOp::getImplementationIdentity() { return getImplementationAttr(); }

ArrayAttr DispatchOp::getExplicitDependencies() { return getAfter(); }

LogicalResult DispatchOp::verify() {
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

  for (Attribute dependency : getAfter())
    if (!isa<FlatSymbolRefAttr>(dependency))
      return emitOpError("expects every 'after' entry to be a flat symbol reference");

  auto function =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getKernelAttr());
  if (!function)
    return emitOpError() << "references unknown kernel " << getKernelAttr();
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

#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "ckl/Dialect/CKL/IR/CKLTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::ckl;

#include "ckl/Dialect/CKL/IR/CKLOpsDialect.cpp.inc"

void CKLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ckl/Dialect/CKL/IR/CKLOps.cpp.inc"
      >();
  registerTypes();
}

LogicalResult CKLDialect::verifyOperationAttribute(Operation *operation, NamedAttribute attribute) {
  if (attribute.getName() != "ckl.kernel")
    return success();
  auto function = dyn_cast<func::FuncOp>(operation);
  if (!function || !isa<UnitAttr>(attribute.getValue()))
    return operation->emitOpError("ckl.kernel requires a unit attribute on func.func");
  if (function.isExternal() || function.getNumResults() != 0)
    return operation->emitOpError("ckl.kernel requires a defined function with no results");
  for (Type type : function.getArgumentTypes()) {
    if (auto buffer = dyn_cast<MemRefType>(type))
      type = buffer.getElementType();
    if (!isa<IntegerType, FloatType, IndexType>(type))
      return operation->emitOpError(
          "ckl.kernel arguments must be numeric scalars or ranked numeric memrefs");
  }
  return success();
}

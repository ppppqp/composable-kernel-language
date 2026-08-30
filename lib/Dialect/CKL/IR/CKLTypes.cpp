#include "ckl/Dialect/CKL/IR/CKLTypes.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ckl;

#define GET_TYPEDEF_CLASSES
#include "ckl/Dialect/CKL/IR/CKLTypes.cpp.inc"

void CKLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ckl/Dialect/CKL/IR/CKLTypes.cpp.inc"
      >();
}

LogicalResult TileType::verify(function_ref<InFlightDiagnostic()> emitError, Type elementType,
                               SpaceAttr space) {
  if (!elementType || mlir::isa<NoneType>(elementType))
    return emitError() << "tile element type must be a concrete MLIR type";
  if (!space)
    return emitError() << "tile requires a logical index space";
  return success();
}

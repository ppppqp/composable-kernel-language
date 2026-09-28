#include "ckl/Dialect/CKL/IR/CKLTypes.h"

#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir::ckl;

#define GET_TYPEDEF_CLASSES
#include "ckl/Dialect/CKL/IR/CKLOpsTypes.cpp.inc"

void CKLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ckl/Dialect/CKL/IR/CKLOpsTypes.cpp.inc"
      >();
}

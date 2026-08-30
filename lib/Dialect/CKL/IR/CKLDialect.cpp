#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "ckl/Dialect/CKL/IR/CKLAttrs.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "ckl/Dialect/CKL/IR/CKLTypes.h"

using namespace mlir;
using namespace mlir::ckl;

#include "ckl/Dialect/CKL/IR/CKLOpsDialect.cpp.inc"

void CKLDialect::initialize() {
  registerAttributes();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "ckl/Dialect/CKL/IR/CKLOps.cpp.inc"
      >();
}

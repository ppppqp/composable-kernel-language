#include "ckl/Dialect/Exec/IR/CKLExecDialect.h"
#include "ckl/Dialect/Exec/IR/CKLExecOps.h"

using namespace mlir;
using namespace mlir::ckl::exec;

#include "ckl/Dialect/Exec/IR/CKLExecOpsDialect.cpp.inc"

void CKLExecDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ckl/Dialect/Exec/IR/CKLExecOps.cpp.inc"
      >();
}

#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIADialect.h"
#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIAOps.h"

using namespace mlir;
using namespace mlir::ckl::nvidia;

#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIAOpsDialect.cpp.inc"

void CKLNVIDIADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIAOps.cpp.inc"
      >();
}

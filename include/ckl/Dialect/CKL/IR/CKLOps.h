#ifndef CKL_DIALECT_CKL_IR_CKLOPS_H
#define CKL_DIALECT_CKL_IR_CKLOPS_H

#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "ckl/Dialect/CKL/IR/CKLInterfaces.h"
#include "ckl/Dialect/CKL/IR/CKLTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#define GET_OP_CLASSES
#include "ckl/Dialect/CKL/IR/CKLOps.h.inc"

#endif

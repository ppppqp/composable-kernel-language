#ifndef CKL_DIALECT_EXEC_IR_CKLEXECOPS_H
#define CKL_DIALECT_EXEC_IR_CKLEXECOPS_H

#include "ckl/Dialect/Exec/IR/CKLExecDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "ckl/Dialect/Exec/IR/CKLExecOps.h.inc"

#endif

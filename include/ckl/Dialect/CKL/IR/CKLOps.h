#pragma once

#include "ckl/Dialect/CKL/IR/CKLTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "ckl/Dialect/CKL/IR/CKLOps.h.inc"

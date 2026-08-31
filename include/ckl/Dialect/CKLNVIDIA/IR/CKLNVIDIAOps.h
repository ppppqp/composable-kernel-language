#pragma once

#include "ckl/Dialect/CKL/IR/CKLTypes.h"
#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIADialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIAOps.h.inc"

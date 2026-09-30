#ifndef CKL_DIALECT_CKL_IR_CKLINTERFACES_H
#define CKL_DIALECT_CKL_IR_CKLINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class DialectRegistry;
namespace ckl {
/// Attach CKL orchestration interfaces to supported upstream producer ops.
void registerProducerInterfaceExternalModels(DialectRegistry &registry);
} // namespace ckl
} // namespace mlir

#include "ckl/Dialect/CKL/IR/CKLInterfaces.h.inc"

#endif

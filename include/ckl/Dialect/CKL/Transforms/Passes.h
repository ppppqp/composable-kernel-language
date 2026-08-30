#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::ckl {

std::unique_ptr<Pass> createPlanCompositionsPass();
void registerCKLPasses();

} // namespace mlir::ckl

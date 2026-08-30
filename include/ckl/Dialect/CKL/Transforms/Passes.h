#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::ckl {

std::unique_ptr<Pass> createPlanCompositionsPass();
std::unique_ptr<Pass> createSelectAlternativesPass();
std::unique_ptr<Pass> createSelectLinearPipelinesPass();
std::unique_ptr<Pass> createScheduleConversionsPass();
void registerCKLPasses();

} // namespace mlir::ckl

#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::ckl {

std::unique_ptr<Pass> createPlanCompositionsPass();
std::unique_ptr<Pass> createSelectAlternativesPass();
std::unique_ptr<Pass> createScheduleConversionsPass();
std::unique_ptr<Pass> createEnumerateAlternativesPass();
void registerEnumerateAlternativesPass();
void registerCKLPasses();

} // namespace mlir::ckl

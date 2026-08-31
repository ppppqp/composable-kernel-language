#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::ckl::nvidia {

std::unique_ptr<Pass> createMaterializeMmaSyncPass();
std::unique_ptr<Pass> createLowerMmaSyncPass();
void registerCKLNVIDIAPasses();

} // namespace mlir::ckl::nvidia

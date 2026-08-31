#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::ckl::nvidia {

std::unique_ptr<Pass> createMaterializeMmaSyncPass();
void registerCKLNVIDIAPasses();

} // namespace mlir::ckl::nvidia

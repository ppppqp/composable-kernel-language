#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "ckl/Dialect/CKL/Transforms/Passes.h"
#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIADialect.h"
#include "ckl/Dialect/CKLNVIDIA/Transforms/Passes.h"
#include "ckl/Extensions/NVIDIA/MmaSync.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::ckl::registerCKLPasses();
  mlir::ckl::nvidia::registerCKLNVIDIAPasses();
  ckl::extensions::nvidia::registerAlternativeProviders();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::ckl::CKLDialect, mlir::ckl::nvidia::CKLNVIDIADialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::gpu::GPUDialect,
                  mlir::memref::MemRefDialect, mlir::nvgpu::NVGPUDialect,
                  mlir::vector::VectorDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CKL optimizer driver\n", registry));
}

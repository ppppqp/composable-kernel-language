#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "ckl/Dialect/CKL/Transforms/Passes.h"
#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIADialect.h"
#include "ckl/Dialect/CKLNVIDIA/Transforms/Passes.h"
#include "ckl/Extensions/NVIDIA/MmaSync.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::ckl::registerCKLPasses();
  mlir::ckl::nvidia::registerCKLNVIDIAPasses();
  ckl::extensions::nvidia::registerAlternativeProviders();

  mlir::DialectRegistry registry;
  registry.insert<mlir::ckl::CKLDialect, mlir::ckl::nvidia::CKLNVIDIADialect,
                  mlir::func::FuncDialect, mlir::nvgpu::NVGPUDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CKL optimizer driver\n", registry));
}

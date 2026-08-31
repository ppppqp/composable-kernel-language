#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "ckl/Dialect/CKL/Transforms/Passes.h"
#include "ckl/Extensions/NVIDIA/MmaSync.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::ckl::registerCKLPasses();
  ckl::extensions::nvidia::registerAlternativeProviders();

  mlir::DialectRegistry registry;
  registry.insert<mlir::ckl::CKLDialect, mlir::func::FuncDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CKL optimizer driver\n", registry));
}

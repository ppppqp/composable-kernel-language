#include "ckl/Analysis/EffectAnalysis.h"
#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "ckl/Dialect/CKL/IR/CKLInterfaces.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::ckl::registerCKLPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::ckl::registerProducerInterfaceExternalModels(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  registry.insert<mlir::ckl::CKLDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CKL orchestration optimizer bootstrap\n", registry));
}

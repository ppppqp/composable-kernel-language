#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "ckl/Analysis/EffectAnalysis.h"
#include "ckl/Dialect/CKL/IR/CKLDialect.h"
#include "ckl/Dialect/Exec/IR/CKLExecDialect.h"
#include "ckl/Transforms/CudaKernels.h"
#include "ckl/Transforms/ExecutablePlan.h"
#include "ckl/Transforms/PointwiseFusion.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/CAPI/IR.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Module.h"
#include <mutex>
#include <stdexcept>

namespace nb = nanobind;
void registerRuntimeBindings(nb::module_ &module);

NB_MODULE(_ckl_native, m) {
  m.doc() = "CKL MLIR registration and native GPU runtime bindings";
  m.def("register_context", [](nb::handle capsule) {
    static std::once_flag once;
    std::call_once(once, [] {
      mlir::ckl::registerCKLPasses();
      mlir::ckl::registerCKLExecutablePlanPasses();
      mlir::ckl::registerCKLCudaKernelPasses();
      mlir::ckl::registerCKLPointwiseFusionPass();
    });
    MlirContext raw = mlirPythonCapsuleToContext(capsule.ptr());
    if (mlirContextIsNull(raw))
      throw nb::python_error();
    auto *context = unwrap(raw);
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::registerAllGPUToLLVMIRTranslations(registry);
    registry.insert<mlir::ckl::CKLDialect, mlir::ckl::exec::CKLExecDialect>();
    context->appendDialectRegistry(registry);
    context->getOrLoadDialect<mlir::ckl::CKLDialect>();
    context->getOrLoadDialect<mlir::ckl::exec::CKLExecDialect>();
  });
  // Inspection only: compilation and ExecutionEngine ownership live in Python.
  m.def("translate_llvm", [](nb::handle capsule) {
    MlirModule raw = mlirPythonCapsuleToModule(capsule.ptr());
    if (mlirModuleIsNull(raw))
      throw nb::python_error();
    llvm::LLVMContext context;
    auto module = mlir::translateModuleToLLVMIR(unwrap(raw), context);
    if (!module)
      throw std::runtime_error("could not translate host MLIR to LLVM IR");
    std::string text;
    llvm::raw_string_ostream stream(text);
    module->print(stream, nullptr);
    return text;
  });
  registerRuntimeBindings(m);
}

#include "ckl/Dialect/CKL/Transforms/Passes.h"

#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::ckl {
namespace {

/// Return true when selection has work in executable code. Invocations inside
/// functions referenced by task alternatives are dormant templates until that
/// function is materialized at an active invocation site.
static bool hasUnselectedActiveInvocation(ModuleOp module) {
  llvm::DenseSet<Operation *> implementationTemplates;
  module.walk([&](TaskOp task) {
    for (Attribute value : task.getAlternatives()) {
      auto alternative = cast<DictionaryAttr>(value);
      auto reference = alternative.getAs<FlatSymbolRefAttr>("implementation");
      if (!reference)
        continue;
      if (Operation *implementation = SymbolTable::lookupNearestSymbolFrom(task, reference))
        implementationTemplates.insert(implementation);
    }
  });

  bool found = false;
  module.walk([&](InvokeOp invoke) {
    if (found || invoke.getAlternativeAttr())
      return;
    for (Operation *ancestor = invoke->getParentOp(); ancestor;
         ancestor = ancestor->getParentOp()) {
      if (implementationTemplates.contains(ancestor))
        return;
    }
    found = true;
  });
  return found;
}

static unsigned countSelectedCallableInvocations(ModuleOp module) {
  unsigned count = 0;
  module.walk([&](InvokeOp invoke) {
    if (invoke->hasAttr("ckl.implementation"))
      ++count;
  });
  return count;
}

class ResolveTaskAlternativesPass
    : public PassWrapper<ResolveTaskAlternativesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveTaskAlternativesPass)
  ResolveTaskAlternativesPass() = default;
  ResolveTaskAlternativesPass(const ResolveTaskAlternativesPass &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "ckl-resolve-task-alternatives"; }
  StringRef getDescription() const final {
    return "Select and materialize nested CKL task alternatives to a fixed point";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  Option<unsigned> maximumRounds{
      *this, "maximum-rounds",
      llvm::cl::desc("Maximum nested callable-alternative expansion rounds"),
      llvm::cl::init(32)};

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (maximumRounds == 0) {
      module.emitError("ckl-resolve-task-alternatives requires maximum-rounds to be positive");
      signalPassFailure();
      return;
    }

    for (unsigned round = 0; round < maximumRounds; ++round) {
      OpPassManager selectPipeline(ModuleOp::getOperationName());
      selectPipeline.addPass(createSelectAlternativesPass());
      if (failed(runPipeline(selectPipeline, module))) {
        signalPassFailure();
        return;
      }

      unsigned selectedCallables = countSelectedCallableInvocations(module);
      if (selectedCallables == 0)
        return; // Remaining selected invocations belong to target extensions.

      OpPassManager materializePipeline(ModuleOp::getOperationName());
      materializePipeline.addPass(createMaterializeSelectedAlternativesPass());
      if (failed(runPipeline(materializePipeline, module))) {
        signalPassFailure();
        return;
      }
      if (countSelectedCallableInvocations(module) != 0) {
        module.emitError("callable-alternative materialization made no progress");
        signalPassFailure();
        return;
      }
      if (!hasUnselectedActiveInvocation(module))
        return;
    }

    module.emitError("callable task expansion exceeded maximum-rounds=")
        << maximumRounds
        << "; the task graph may contain a recursive callable-alternative cycle";
    signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createResolveTaskAlternativesPass() {
  return std::make_unique<ResolveTaskAlternativesPass>();
}

void registerResolveTaskAlternativesPass() {
  PassRegistration<ResolveTaskAlternativesPass>();
}

} // namespace mlir::ckl

#include "ckl/Dialect/CKL/Transforms/Passes.h"

#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::ckl {
namespace {

// for an invoke op with a selected callable alternative, materialize the selected alternative as a
// func.call operation and remove the invoke op. If the selected alternative is a symbol reference
// to a func.func operation, the func.call operation will be created with the same symbol reference.
class MaterializeSelectedAlternativesPass
    : public PassWrapper<MaterializeSelectedAlternativesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeSelectedAlternativesPass)

  StringRef getArgument() const final { return "ckl-materialize-selected-alternatives"; }
  StringRef getDescription() const final {
    return "Materialize selected callable CKL task alternatives as func.call operations";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() final {
    SmallVector<InvokeOp> invocations;
    getOperation().walk([&](InvokeOp invoke) {
      if (invoke->hasAttr("ckl.implementation"))
        invocations.push_back(invoke);
    });

    IRRewriter rewriter(&getContext());
    for (InvokeOp invoke : invocations) {
      auto reference = invoke->getAttrOfType<FlatSymbolRefAttr>("ckl.implementation");
      if (!reference) {
        invoke.emitError("selected callable implementation must be a flat symbol reference");
        signalPassFailure();
        return;
      }
      auto implementation = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(invoke, reference);
      if (!implementation) {
        invoke.emitError("selected callable implementation does not resolve to func.func ")
            << reference;
        signalPassFailure();
        return;
      }
      auto callType =
          FunctionType::get(&getContext(), invoke.getInputs().getTypes(), invoke.getResultTypes());
      if (implementation.getFunctionType() != callType) {
        invoke.emitError("selected callable implementation type ")
            << implementation.getFunctionType() << " does not match invocation type " << callType;
        signalPassFailure();
        return;
      }

      rewriter.setInsertionPoint(invoke);
      // replace the invoke op with a func.call operation that calls the selected alternative
      auto call = func::CallOp::create(rewriter, invoke.getLoc(), reference.getValue(),
                                       invoke.getResultTypes(), invoke.getInputs());
      if (auto selected = invoke.getAlternativeAttr())
        call->setAttr("ckl.selected_alternative", selected);
      for (NamedAttribute attribute : invoke->getDiscardableAttrs())
        if (attribute.getName().getValue().starts_with("ckl.") &&
            attribute.getName().getValue() != "ckl.implementation")
          call->setAttr(attribute.getName(), attribute.getValue());
      rewriter.replaceOp(invoke, call.getResults());
    }

    SmallVector<TaskOp> deadTasks;
    // clean up any task symbols that are no longer referenced by any invoke operations
    getOperation().walk([&](TaskOp task) {
      Operation *symbolTable = task->getParentWithTrait<OpTrait::SymbolTable>();
      if (symbolTable && SymbolTable::symbolKnownUseEmpty(task, symbolTable))
        deadTasks.push_back(task);
    });
    for (TaskOp task : deadTasks)
      rewriter.eraseOp(task);
  }
};

} // namespace

std::unique_ptr<Pass> createMaterializeSelectedAlternativesPass() {
  return std::make_unique<MaterializeSelectedAlternativesPass>();
}

void registerMaterializeSelectedAlternativesPass() {
  PassRegistration<MaterializeSelectedAlternativesPass>();
}

} // namespace mlir::ckl

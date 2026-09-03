#include "ckl/Dialect/CKL/Transforms/Passes.h"

#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::ckl {
namespace {

class CKLInlinerInterface final : public InlinerInterface {
public:
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Region *destination, Region *source, bool wouldBeCloned,
                       IRMapping &mapping) const final {
    if (mlir::isa<FunctionOpInterface>(destination->getParentOp()) &&
        mlir::isa<func::FuncOp>(source->getParentOp()))
      return true;
    return InlinerInterface::isLegalToInline(destination, source, wouldBeCloned, mapping);
  }
};

class MaterializeSelectedAlternativesPass
    : public PassWrapper<MaterializeSelectedAlternativesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeSelectedAlternativesPass)

  StringRef getArgument() const final { return "ckl-materialize-selected-alternatives"; }
  StringRef getDescription() const final {
    return "Inline selected callable CKL task alternatives into their invocation sites";
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
    SmallVector<func::FuncOp> usedImplementations;
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
      if (implementation.isExternal()) {
        invoke.emitError("selected callable implementation must have a body");
        signalPassFailure();
        return;
      }

      Operation *callable = invoke->getParentOp();
      while (callable && !mlir::isa<FunctionOpInterface>(callable))
        callable = callable->getParentOp();
      if (!callable) {
        invoke.emitError("callable alternative invocation must be nested in a function-like op");
        signalPassFailure();
        return;
      }
      SmallVector<Attribute> provenance;
      if (auto existing = callable->getAttrOfType<ArrayAttr>("ckl.inlined_alternatives"))
        llvm::append_range(provenance, existing);
      NamedAttrList record;
      record.set("task", rewriter.getStringAttr(invoke.getCallee()));
      record.set("implementation", rewriter.getStringAttr(reference.getValue()));
      if (auto selected = invoke.getAlternativeAttr())
        record.set("alternative", selected);
      if (auto id = invoke->getAttrOfType<StringAttr>("ckl.implementation_id"))
        record.set("implementation_id", id);
      if (auto score = invoke->getAttrOfType<IntegerAttr>("ckl.graph_score"))
        record.set("graph_score", score);
      provenance.push_back(rewriter.getDictionaryAttr(record));
      callable->setAttr("ckl.inlined_alternatives", rewriter.getArrayAttr(provenance));

      rewriter.setInsertionPoint(invoke);
      auto call = func::CallOp::create(rewriter, invoke.getLoc(), reference.getValue(),
                                       invoke.getResultTypes(), invoke.getInputs());
      rewriter.replaceOp(invoke, call.getResults());
      CKLInlinerInterface inliner(&getContext());
      InlinerConfig config;
      if (failed(inlineCall(inliner, config.getCloneCallback(), call, implementation,
                            &implementation.getBody(), /*shouldCloneInlinedRegion=*/true))) {
        call.emitError("failed to inline selected callable implementation ") << reference;
        signalPassFailure();
        return;
      }
      call.erase();
      if (!llvm::is_contained(usedImplementations, implementation))
        usedImplementations.push_back(implementation);
    }

    SmallVector<TaskOp> deadTasks;
    getOperation().walk([&](TaskOp task) {
      Operation *symbolTable = task->getParentWithTrait<OpTrait::SymbolTable>();
      if (symbolTable && SymbolTable::symbolKnownUseEmpty(task, symbolTable))
        deadTasks.push_back(task);
    });
    for (TaskOp task : deadTasks) {
      for (Attribute value : task.getAlternatives()) {
        auto alternative = mlir::cast<DictionaryAttr>(value);
        auto reference = alternative.getAs<FlatSymbolRefAttr>("implementation");
        if (!reference)
          continue;
        if (auto implementation =
                SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(task, reference);
            implementation && !llvm::is_contained(usedImplementations, implementation))
          usedImplementations.push_back(implementation);
      }
      rewriter.eraseOp(task);
    }
    for (func::FuncOp implementation : usedImplementations) {
      Operation *symbolTable = implementation->getParentWithTrait<OpTrait::SymbolTable>();
      if (symbolTable && SymbolTable::symbolKnownUseEmpty(implementation, symbolTable))
        rewriter.eraseOp(implementation);
    }
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

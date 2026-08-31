#include "ckl/Dialect/CKLNVIDIA/Transforms/Passes.h"

#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIAOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::ckl::nvidia {
namespace {

constexpr StringLiteral mmaSyncImplementation =
    "nvidia.mma.sync.m16n8k16.row.col.f32.f16.f16.f32";

class MaterializeMmaSyncPass
    : public PassWrapper<MaterializeMmaSyncPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeMmaSyncPass)
  StringRef getArgument() const final { return "ckl-nvidia-materialize-mma-sync"; }
  StringRef getDescription() const final {
    return "Materialize selected NVIDIA MMA task invocations";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<CKLNVIDIADialect>();
  }

  void runOnOperation() final {
    SmallVector<::mlir::ckl::InvokeOp> invocations;
    getOperation().walk([&](::mlir::ckl::InvokeOp op) {
      auto implementation = op->getAttrOfType<StringAttr>("ckl.implementation_id");
      if (implementation && implementation.getValue() == mmaSyncImplementation)
        invocations.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    for (::mlir::ckl::InvokeOp invoke : invocations) {
      if (invoke.getInputs().size() != 3 || invoke.getResults().size() != 1) {
        invoke.emitError("selected m16n8k16 MMA implementation requires three inputs and one result");
        signalPassFailure();
        return;
      }
      OperationState state(invoke.getLoc(), MmaSyncOp::getOperationName());
      state.addOperands(invoke.getInputs());
      state.addTypes(invoke.getResultTypes());
      for (NamedAttribute attribute : invoke->getDiscardableAttrs())
        if (attribute.getName().getValue().starts_with("ckl."))
          state.addAttribute(attribute.getName(), attribute.getValue());
      rewriter.setInsertionPoint(invoke);
      Operation *materialized = rewriter.create(state);
      rewriter.replaceOp(invoke, materialized->getResults());
    }
  }
};

} // namespace

std::unique_ptr<Pass> createMaterializeMmaSyncPass() {
  return std::make_unique<MaterializeMmaSyncPass>();
}

void registerLowerMmaSyncPass();

void registerCKLNVIDIAPasses() {
  PassRegistration<MaterializeMmaSyncPass>();
  registerLowerMmaSyncPass();
}

} // namespace mlir::ckl::nvidia

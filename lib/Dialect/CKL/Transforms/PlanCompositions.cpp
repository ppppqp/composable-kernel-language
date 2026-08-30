#include "ckl/Dialect/CKL/Transforms/Passes.h"

#include "ckl/Core/Layout/Distribution.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::ckl {
namespace {

class PlanCompositionsPass
    : public PassWrapper<PlanCompositionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanCompositionsPass)

  StringRef getArgument() const final { return "ckl-plan-compositions"; }
  StringRef getDescription() const final {
    return "Classify CKL composition boundaries with CKLCore";
  }

  void runOnOperation() final {
    SmallVector<ComposeOp> compositions;
    getOperation().walk([&](ComposeOp op) { compositions.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (ComposeOp op : compositions) {
      try {
        auto source = ::ckl::core::deserializeDistribution(op.getSource().getValue().str());
        auto target = ::ckl::core::deserializeDistribution(op.getTarget().getValue().str());
        auto plan = ::ckl::core::classifyConversion(source, target, op.getSubgroupSize());
        if (plan.kind == ::ckl::core::ConversionKind::Unsupported) {
          op.emitError() << "cannot compose layouts: " << plan.reason;
          signalPassFailure();
          return;
        }
        OperationState state(op.getLoc(), ConvertLayoutOp::getOperationName());
        state.addOperands(op.getInput());
        state.addTypes(op.getResult().getType());
        state.addAttribute("source", op.getSource());
        state.addAttribute("target", op.getTarget());
        state.addAttribute("kind", rewriter.getStringAttr(::ckl::core::toString(plan.kind)));
        rewriter.setInsertionPoint(op);
        Operation *replacement = rewriter.create(state);
        rewriter.replaceOp(op, replacement->getResults());
      } catch (const std::exception &error) {
        op.emitError() << "failed to import CKLCore distribution: " << error.what();
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPlanCompositionsPass() {
  return std::make_unique<PlanCompositionsPass>();
}

void registerCKLPasses() {
  PassRegistration<PlanCompositionsPass>();
}

} // namespace mlir::ckl

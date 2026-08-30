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
        state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(op.getSubgroupSize()));
        state.addAttribute("reason", rewriter.getStringAttr(plan.reason));
        state.addAttribute("move_count", rewriter.getI64IntegerAttr(plan.moves.size()));
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

class ScheduleConversionsPass
    : public PassWrapper<ScheduleConversionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleConversionsPass)

  StringRef getArgument() const final { return "ckl-schedule-conversions"; }
  StringRef getDescription() const final {
    return "Materialize semantic CKL conversions as scope-specific operations";
  }

  void runOnOperation() final {
    SmallVector<ConvertLayoutOp> conversions;
    getOperation().walk([&](ConvertLayoutOp op) { conversions.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (ConvertLayoutOp op : conversions) {
      if (op.getKind() == "identity") {
        rewriter.replaceOp(op, op.getInput());
        continue;
      }
      StringRef operationName =
          op.getKind() == "local-permutation" ? LocalPermuteOp::getOperationName()
        : op.getKind() == "subgroup-exchange" ? SubgroupExchangeOp::getOperationName()
        : op.getKind() == "shared-memory-exchange" ? SharedExchangeOp::getOperationName()
        : op.getKind() == "global-memory-exchange" ? GlobalExchangeOp::getOperationName()
        : StringRef{};
      if (operationName.empty()) {
        op.emitError("has no scheduling operation for conversion kind '") << op.getKind() << "'";
        signalPassFailure();
        return;
      }
      OperationState state(op.getLoc(), operationName);
      state.addOperands(op.getInput());
      state.addTypes(op.getResult().getType());
      state.addAttribute("source", op.getSource());
      state.addAttribute("target", op.getTarget());
      state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(op.getSubgroupSize()));
      if (auto reason = op.getReasonAttr())
        state.addAttribute("reason", reason);
      if (auto count = op.getMoveCountAttr())
        state.addAttribute("move_count", count);
      rewriter.setInsertionPoint(op);
      Operation *scheduled = rewriter.create(state);
      rewriter.replaceOp(op, scheduled->getResults());
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPlanCompositionsPass() {
  return std::make_unique<PlanCompositionsPass>();
}

std::unique_ptr<Pass> createScheduleConversionsPass() {
  return std::make_unique<ScheduleConversionsPass>();
}

void registerCKLPasses() {
  PassRegistration<PlanCompositionsPass>();
  PassRegistration<ScheduleConversionsPass>();
}

} // namespace mlir::ckl

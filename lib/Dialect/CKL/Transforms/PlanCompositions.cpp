#include "ckl/Dialect/CKL/Transforms/Passes.h"

#include "ckl/Core/Layout/Distribution.h"
#include "ckl/Core/Composition/Task.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::ckl {
namespace {

FailureOr<::ckl::core::Placement> importPlacement(DictionaryAttr port, Operation *anchor) {
  auto value = port.getAs<StringAttr>("placement");
  if (!value || value.getValue() == "private") return ::ckl::core::Placement::Private;
  if (value.getValue() == "shared") return ::ckl::core::Placement::Shared;
  if (value.getValue() == "global") return ::ckl::core::Placement::Global;
  anchor->emitError("unknown task port placement '") << value.getValue() << "'";
  return failure();
}

FailureOr<::ckl::core::TaskAlternative>
importAlternative(TaskOp task, DictionaryAttr attribute, Operation *anchor) {
  auto name = attribute.getAs<StringAttr>("name");
  if (!name) return failure(); // TaskOp verifier diagnoses this first.
  ::ckl::core::TaskAlternative result;
  result.task = task.getSymName().str();
  result.name = name.getValue().str();
  if (auto value = attribute.getAs<IntegerAttr>("registers_per_thread"))
    result.registersPerThread = value.getInt();
  if (auto value = attribute.getAs<IntegerAttr>("shared_memory_bytes"))
    result.sharedMemoryBytes = value.getInt();
  if (auto values = attribute.getAs<ArrayAttr>("required_capabilities"))
    for (Attribute value : values)
      result.requiredCapabilities.push_back(mlir::cast<StringAttr>(value).getValue().str());

  auto importPorts = [&](StringRef field, std::vector<::ckl::core::PortRealization> &ports) {
    auto values = attribute.getAs<ArrayAttr>(field);
    if (!values) return success();
    for (Attribute value : values) {
      auto port = mlir::dyn_cast<DictionaryAttr>(value);
      auto portName = port ? port.getAs<StringAttr>("name") : StringAttr{};
      auto distribution = port ? port.getAs<DistributionAttr>("distribution") : DistributionAttr{};
      if (!portName || !distribution) {
        anchor->emitError("task alternative port requires string 'name' and CKL 'distribution'");
        return failure();
      }
      auto placement = importPlacement(port, anchor);
      if (failed(placement)) return failure();
      std::int64_t vectorWidth = 1;
      if (auto value = port.getAs<IntegerAttr>("vector_width")) vectorWidth = value.getInt();
      if (vectorWidth <= 0) {
        anchor->emitError("task port vector_width must be positive");
        return failure();
      }
      ports.push_back({portName.getValue().str(),
                       ::ckl::core::deserializeDistribution(distribution.getValue().str()),
                       *placement, vectorWidth});
    }
    return success();
  };
  if (failed(importPorts("inputs", result.inputs)) ||
      failed(importPorts("outputs", result.outputs)))
    return failure();
  return result;
}

class SelectAlternativesPass
    : public PassWrapper<SelectAlternativesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAlternativesPass)
  StringRef getArgument() const final { return "ckl-select-alternatives"; }
  StringRef getDescription() const final {
    return "Jointly select task alternatives using CKLCore";
  }

  void runOnOperation() final {
    SmallVector<TaskComposeOp> boundaries;
    getOperation().walk([&](TaskComposeOp op) { boundaries.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (TaskComposeOp op : boundaries) {
      auto producer = SymbolTable::lookupNearestSymbolFrom<TaskOp>(op, op.getProducerAttr());
      auto consumer = SymbolTable::lookupNearestSymbolFrom<TaskOp>(op, op.getConsumerAttr());
      std::vector<::ckl::core::TaskAlternative> producers, consumers;
      try {
        for (Attribute value : producer.getAlternatives()) {
          auto imported = importAlternative(producer, mlir::cast<DictionaryAttr>(value), op);
          if (failed(imported)) { signalPassFailure(); return; }
          producers.push_back(std::move(*imported));
        }
        for (Attribute value : consumer.getAlternatives()) {
          auto imported = importAlternative(consumer, mlir::cast<DictionaryAttr>(value), op);
          if (failed(imported)) { signalPassFailure(); return; }
          consumers.push_back(std::move(*imported));
        }
        std::vector<std::string> capabilities;
        for (Attribute value : op.getCapabilities())
          capabilities.push_back(mlir::cast<StringAttr>(value).getValue().str());
        auto decision = ::ckl::core::selectComposition(
            producers, consumers, op.getProducerPort().str(), op.getConsumerPort().str(),
            op.getSubgroupSize(), op.getRegisterLimit(), op.getSharedMemoryLimit(), capabilities);
        if (!decision.selected) {
          auto diagnostic = op.emitError("no legal task-alternative pair");
          for (const auto &candidate : decision.considered)
            diagnostic.attachNote(op.getLoc())
                << producers[candidate.producerAlternative].name << " -> "
                << consumers[candidate.consumerAlternative].name << ": "
                << candidate.explanation;
          signalPassFailure();
          return;
        }
        const auto &selected = *decision.selected;
        const auto &source = producers[selected.producerAlternative].outputs;
        const auto &target = consumers[selected.consumerAlternative].inputs;
        auto sourcePort = llvm::find_if(source, [&](const auto &p) { return p.name == op.getProducerPort(); });
        auto targetPort = llvm::find_if(target, [&](const auto &p) { return p.name == op.getConsumerPort(); });
        OperationState state(op.getLoc(), ComposeOp::getOperationName());
        state.addOperands(op.getInput());
        state.addTypes(op.getResult().getType());
        state.addAttribute("source", DistributionAttr::get(
            &getContext(), ::ckl::core::serialize(sourcePort->distribution)));
        state.addAttribute("target", DistributionAttr::get(
            &getContext(), ::ckl::core::serialize(targetPort->distribution)));
        state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(op.getSubgroupSize()));
        state.addAttribute("ckl.producer_alternative",
                           rewriter.getStringAttr(producers[selected.producerAlternative].name));
        state.addAttribute("ckl.consumer_alternative",
                           rewriter.getStringAttr(consumers[selected.consumerAlternative].name));
        SmallVector<Attribute> considered;
        for (const auto &candidate : decision.considered)
          considered.push_back(rewriter.getDictionaryAttr({
              rewriter.getNamedAttr("producer", rewriter.getStringAttr(
                  producers[candidate.producerAlternative].name)),
              rewriter.getNamedAttr("consumer", rewriter.getStringAttr(
                  consumers[candidate.consumerAlternative].name)),
              rewriter.getNamedAttr("score", rewriter.getI64IntegerAttr(candidate.score)),
              rewriter.getNamedAttr("explanation", rewriter.getStringAttr(candidate.explanation))}));
        state.addAttribute("ckl.considered_alternatives", rewriter.getArrayAttr(considered));
        rewriter.setInsertionPoint(op);
        Operation *replacement = rewriter.create(state);
        rewriter.replaceOp(op, replacement->getResults());
      } catch (const std::exception &error) {
        op.emitError("failed to import task alternatives: ") << error.what();
        signalPassFailure();
        return;
      }
    }
  }
};

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
        for (NamedAttribute attribute : op->getAttrs())
          if (attribute.getName().strref().starts_with("ckl."))
            state.addAttribute(attribute.getName(), attribute.getValue());
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

std::unique_ptr<Pass> createSelectAlternativesPass() {
  return std::make_unique<SelectAlternativesPass>();
}

std::unique_ptr<Pass> createScheduleConversionsPass() {
  return std::make_unique<ScheduleConversionsPass>();
}

void registerCKLPasses() {
  PassRegistration<SelectAlternativesPass>();
  PassRegistration<PlanCompositionsPass>();
  PassRegistration<ScheduleConversionsPass>();
}

} // namespace mlir::ckl

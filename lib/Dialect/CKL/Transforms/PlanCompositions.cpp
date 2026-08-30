#include "ckl/Dialect/CKL/Transforms/Passes.h"

#include "ckl/Core/Composition/Task.h"
#include "ckl/Core/Layout/Distribution.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::ckl {
namespace {

FailureOr<::ckl::core::Placement> importPlacement(DictionaryAttr port, Operation *anchor) {
  auto value = port.getAs<StringAttr>("placement");
  if (!value || value.getValue() == "private")
    return ::ckl::core::Placement::Private;
  if (value.getValue() == "shared")
    return ::ckl::core::Placement::Shared;
  if (value.getValue() == "global")
    return ::ckl::core::Placement::Global;
  anchor->emitError("unknown task port placement '") << value.getValue() << "'";
  return failure();
}

FailureOr<::ckl::core::TaskAlternative> importAlternative(TaskOp task, DictionaryAttr attribute,
                                                          Operation *anchor) {
  auto name = attribute.getAs<StringAttr>("name");
  if (!name)
    return failure(); // TaskOp verifier diagnoses this first.
  ::ckl::core::TaskAlternative result;
  result.task = task.getSymName().str();
  result.name = name.getValue().str();
  if (auto value = attribute.getAs<StringAttr>("implementation_id"))
    result.implementationId = value.getValue().str();
  if (auto value = attribute.getAs<IntegerAttr>("registers_per_thread"))
    result.registersPerThread = value.getInt();
  if (auto value = attribute.getAs<IntegerAttr>("shared_memory_bytes"))
    result.sharedMemoryBytes = value.getInt();
  if (auto value = attribute.getAs<IntegerAttr>("estimated_execution_cost"))
    result.estimatedExecutionCost = value.getInt();
  if (auto values = attribute.getAs<ArrayAttr>("required_capabilities"))
    for (Attribute value : values)
      result.requiredCapabilities.push_back(mlir::cast<StringAttr>(value).getValue().str());
  if (auto values = attribute.getAs<ArrayAttr>("resources")) {
    for (Attribute value : values) {
      auto resource = mlir::dyn_cast<DictionaryAttr>(value);
      auto resourceName = resource ? resource.getAs<StringAttr>("name") : StringAttr{};
      auto bytes = resource ? resource.getAs<IntegerAttr>("bytes") : IntegerAttr{};
      auto begin = resource ? resource.getAs<IntegerAttr>("begin") : IntegerAttr{};
      auto end = resource ? resource.getAs<IntegerAttr>("end") : IntegerAttr{};
      if (!resourceName || !bytes || !begin || !end) {
        anchor->emitError("task resource requires name, bytes, begin, and end");
        return failure();
      }
      result.resources.push_back(
          {resourceName.getValue().str(), bytes.getInt(), begin.getInt(), end.getInt()});
    }
  }
  if (auto values = attribute.getAs<ArrayAttr>("effects")) {
    for (Attribute value : values) {
      auto effect = mlir::dyn_cast<DictionaryAttr>(value);
      auto kind = effect ? effect.getAs<StringAttr>("kind") : StringAttr{};
      auto resource = effect ? effect.getAs<StringAttr>("resource") : StringAttr{};
      auto stage = effect ? effect.getAs<IntegerAttr>("stage") : IntegerAttr{};
      if (!kind || !resource) {
        anchor->emitError("task effect requires string kind and resource");
        return failure();
      }
      auto importedKind = kind.getValue() == "read"          ? ::ckl::core::EffectKind::Read
                          : kind.getValue() == "write"       ? ::ckl::core::EffectKind::Write
                          : kind.getValue() == "consume"     ? ::ckl::core::EffectKind::Consume
                          : kind.getValue() == "channel-put" ? ::ckl::core::EffectKind::ChannelPut
                          : kind.getValue() == "channel-get" ? ::ckl::core::EffectKind::ChannelGet
                          : kind.getValue() == "channel-release"
                              ? ::ckl::core::EffectKind::ChannelRelease
                              : ::ckl::core::EffectKind::Read;
      result.effects.push_back(
          {importedKind, resource.getValue().str(), stage ? stage.getInt() : 0});
    }
  }

  auto importPorts = [&](StringRef field, std::vector<::ckl::core::PortRealization> &ports) {
    auto values = attribute.getAs<ArrayAttr>(field);
    if (!values)
      return success();
    for (Attribute value : values) {
      auto port = mlir::dyn_cast<DictionaryAttr>(value);
      auto portName = port ? port.getAs<StringAttr>("name") : StringAttr{};
      auto distribution = port ? port.getAs<DistributionAttr>("distribution") : DistributionAttr{};
      if (!portName || !distribution) {
        anchor->emitError("task alternative port requires string 'name' and CKL 'distribution'");
        return failure();
      }
      auto placement = importPlacement(port, anchor);
      if (failed(placement))
        return failure();
      std::int64_t vectorWidth = 1;
      if (auto value = port.getAs<IntegerAttr>("vector_width"))
        vectorWidth = value.getInt();
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
    return "Globally select alternatives for linear CKL task-composition chains";
  }

  void runOnOperation() final {
    SmallVector<TaskComposeOp> boundaries;
    getOperation().walk([&](TaskComposeOp op) { boundaries.push_back(op); });
    struct Prepared {
      SmallVector<TaskComposeOp> edges;
      std::vector<std::vector<::ckl::core::TaskAlternative>> alternatives;
      ::ckl::core::PipelineSelection selection;
    };
    SmallVector<Prepared> prepared;
    struct PreparedSingle {
      TaskComposeOp edge;
      std::vector<::ckl::core::TaskAlternative> producers;
      std::vector<::ckl::core::TaskAlternative> consumers;
      ::ckl::core::CompositionDecision decision;
    };
    SmallVector<PreparedSingle> singles;

    // Solve all chains before rewriting any of them: failure leaves the IR
    // entirely unresolved instead of committing a partial global decision.
    for (TaskComposeOp root : boundaries) {
      if (mlir::isa_and_nonnull<TaskComposeOp>(root.getInput().getDefiningOp()))
        continue;
      SmallVector<TaskComposeOp> chain{root};
      TaskComposeOp current = root;
      while (current.getResult().hasOneUse()) {
        auto next = mlir::dyn_cast<TaskComposeOp>(*current.getResult().getUsers().begin());
        if (!next) break;
        chain.push_back(next);
        current = next;
      }
      if (chain.size() < 2) {
        bool hasTaskSuccessor = llvm::any_of(root.getResult().getUsers(), [](Operation *user) {
          return mlir::isa<TaskComposeOp>(user);
        });
        if (hasTaskSuccessor) {
          root.emitError("non-linear task composition requires the future DAG selector");
          signalPassFailure();
          return;
        }
        PreparedSingle single;
        single.edge = root;
        auto producer = SymbolTable::lookupNearestSymbolFrom<TaskOp>(root, root.getProducerAttr());
        auto consumer = SymbolTable::lookupNearestSymbolFrom<TaskOp>(root, root.getConsumerAttr());
        try {
          for (Attribute value : producer.getAlternatives()) {
            auto imported = importAlternative(
                producer, mlir::cast<DictionaryAttr>(value), root);
            if (failed(imported)) { signalPassFailure(); return; }
            single.producers.push_back(std::move(*imported));
          }
          for (Attribute value : consumer.getAlternatives()) {
            auto imported = importAlternative(
                consumer, mlir::cast<DictionaryAttr>(value), root);
            if (failed(imported)) { signalPassFailure(); return; }
            single.consumers.push_back(std::move(*imported));
          }
        } catch (const std::exception &error) {
          root.emitError("failed to import task alternatives: ") << error.what();
          signalPassFailure();
          return;
        }
        std::vector<std::string> capabilities;
        for (Attribute value : root.getCapabilities())
          capabilities.push_back(mlir::cast<StringAttr>(value).getValue().str());
        single.decision = ::ckl::core::selectComposition(
            single.producers, single.consumers, root.getProducerPort().str(),
            root.getConsumerPort().str(), root.getSubgroupSize(), root.getRegisterLimit(),
            root.getSharedMemoryLimit(), capabilities);
        if (!single.decision.selected) {
          auto diagnostic = root.emitError("no legal task-alternative pair");
          for (const auto &candidate : single.decision.considered)
            diagnostic.attachNote(root.getLoc())
                << single.producers[candidate.producerAlternative].name << " -> "
                << single.consumers[candidate.consumerAlternative].name << ": "
                << candidate.explanation;
          signalPassFailure();
          return;
        }
        singles.push_back(std::move(single));
        continue;
      }

      auto subgroupSize = chain.front().getSubgroupSize();
      auto registerLimit = chain.front().getRegisterLimit();
      auto sharedMemoryLimit = chain.front().getSharedMemoryLimit();
      auto capabilitiesAttr = chain.front().getCapabilitiesAttr();
      for (TaskComposeOp edge : chain) {
        if (edge.getSubgroupSize() != subgroupSize ||
            edge.getRegisterLimit() != registerLimit ||
            edge.getSharedMemoryLimit() != sharedMemoryLimit ||
            edge.getCapabilitiesAttr() != capabilitiesAttr) {
          edge.emitError("linear pipeline requires consistent planning limits and capabilities");
          signalPassFailure();
          return;
        }
      }

      SmallVector<TaskOp> tasks;
      tasks.push_back(SymbolTable::lookupNearestSymbolFrom<TaskOp>(
          chain.front(), chain.front().getProducerAttr()));
      for (TaskComposeOp edge : chain)
        tasks.push_back(SymbolTable::lookupNearestSymbolFrom<TaskOp>(edge, edge.getConsumerAttr()));
      for (std::size_t index = 1; index < chain.size(); ++index) {
        auto producer = SymbolTable::lookupNearestSymbolFrom<TaskOp>(
            chain[index], chain[index].getProducerAttr());
        if (tasks[index] != producer) {
          chain[index].emitError("linear chain has inconsistent intermediate task symbol");
          signalPassFailure();
          return;
        }
      }

      Prepared item;
      item.edges = chain;
      item.alternatives.resize(tasks.size());
      try {
        for (std::size_t stage = 0; stage < tasks.size(); ++stage)
          for (Attribute value : tasks[stage].getAlternatives()) {
            auto imported = importAlternative(
                tasks[stage], mlir::cast<DictionaryAttr>(value), chain.front());
            if (failed(imported)) { signalPassFailure(); return; }
            item.alternatives[stage].push_back(std::move(*imported));
          }
      } catch (const std::exception &error) {
        chain.front().emitError("failed to import pipeline alternatives: ") << error.what();
        signalPassFailure();
        return;
      }

      std::vector<::ckl::core::PipelineStage> stages;
      for (std::size_t stage = 0; stage < tasks.size(); ++stage) {
        std::string input = stage == 0 ? "" : chain[stage - 1].getConsumerPort().str();
        std::string output = stage + 1 == tasks.size()
                                 ? "" : chain[stage].getProducerPort().str();
        stages.push_back({tasks[stage].getSymName().str() + "#" + std::to_string(stage),
                          item.alternatives[stage], std::move(input), std::move(output)});
      }
      std::vector<std::string> capabilities;
      for (Attribute value : chain.front().getCapabilities())
        capabilities.push_back(mlir::cast<StringAttr>(value).getValue().str());
      auto decision = ::ckl::core::selectLinearPipeline(
          stages, subgroupSize, registerLimit, sharedMemoryLimit, capabilities);
      if (!decision.selected) {
        auto diagnostic = chain.front().emitError("no legal linear task pipeline");
        for (const std::string &message : decision.diagnostics)
          diagnostic.attachNote(chain.front().getLoc()) << message;
        signalPassFailure();
        return;
      }
      item.selection = std::move(*decision.selected);
      prepared.push_back(std::move(item));
    }

    IRRewriter rewriter(&getContext());
    for (Prepared &pipeline : prepared) {
      SmallVector<Attribute> provenance;
      for (const std::string &entry : pipeline.selection.provenance)
        provenance.push_back(rewriter.getStringAttr(entry));
      for (std::size_t index = 0; index < pipeline.edges.size(); ++index) {
        TaskComposeOp edge = pipeline.edges[index];
        const auto &producer = pipeline.alternatives[index]
            [pipeline.selection.alternatives[index]];
        const auto &consumer = pipeline.alternatives[index + 1]
            [pipeline.selection.alternatives[index + 1]];
        auto source = llvm::find_if(producer.outputs, [&](const auto &port) {
          return port.name == edge.getProducerPort();
        });
        auto target = llvm::find_if(consumer.inputs, [&](const auto &port) {
          return port.name == edge.getConsumerPort();
        });
        OperationState state(edge.getLoc(), ComposeOp::getOperationName());
        state.addOperands(edge.getInput());
        state.addTypes(edge.getResult().getType());
        state.addAttribute("source", DistributionAttr::get(
            &getContext(), ::ckl::core::serialize(source->distribution)));
        state.addAttribute("target", DistributionAttr::get(
            &getContext(), ::ckl::core::serialize(target->distribution)));
        state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(edge.getSubgroupSize()));
        state.addAttribute("ckl.producer_alternative", rewriter.getStringAttr(producer.name));
        state.addAttribute("ckl.consumer_alternative", rewriter.getStringAttr(consumer.name));
        state.addAttribute("ckl.pipeline_score",
                           rewriter.getI64IntegerAttr(pipeline.selection.score));
        state.addAttribute("ckl.pipeline_stage", rewriter.getI64IntegerAttr(index));
        state.addAttribute("ckl.pipeline_provenance", rewriter.getArrayAttr(provenance));
        rewriter.setInsertionPoint(edge);
        Operation *replacement = rewriter.create(state);
        rewriter.replaceOp(edge, replacement->getResults());
      }
    }
    for (PreparedSingle &single : singles) {
      TaskComposeOp edge = single.edge;
      const auto &selected = *single.decision.selected;
      const auto &producer = single.producers[selected.producerAlternative];
      const auto &consumer = single.consumers[selected.consumerAlternative];
      auto source = llvm::find_if(producer.outputs, [&](const auto &port) {
        return port.name == edge.getProducerPort();
      });
      auto target = llvm::find_if(consumer.inputs, [&](const auto &port) {
        return port.name == edge.getConsumerPort();
      });
      OperationState state(edge.getLoc(), ComposeOp::getOperationName());
      state.addOperands(edge.getInput());
      state.addTypes(edge.getResult().getType());
      state.addAttribute("source", DistributionAttr::get(
          &getContext(), ::ckl::core::serialize(source->distribution)));
      state.addAttribute("target", DistributionAttr::get(
          &getContext(), ::ckl::core::serialize(target->distribution)));
      state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(edge.getSubgroupSize()));
      state.addAttribute("ckl.producer_alternative", rewriter.getStringAttr(producer.name));
      state.addAttribute("ckl.consumer_alternative", rewriter.getStringAttr(consumer.name));
      state.addAttribute("ckl.producer_implementation_id", rewriter.getStringAttr(
          producer.implementationId.empty() ? producer.task + ":" + producer.name
                                            : producer.implementationId));
      state.addAttribute("ckl.consumer_implementation_id", rewriter.getStringAttr(
          consumer.implementationId.empty() ? consumer.task + ":" + consumer.name
                                            : consumer.implementationId));
      SmallVector<Attribute> considered;
      for (const auto &candidate : single.decision.considered)
        considered.push_back(rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("producer", rewriter.getStringAttr(
                single.producers[candidate.producerAlternative].name)),
            rewriter.getNamedAttr("consumer", rewriter.getStringAttr(
                single.consumers[candidate.consumerAlternative].name)),
            rewriter.getNamedAttr("score", rewriter.getI64IntegerAttr(candidate.score)),
            rewriter.getNamedAttr("explanation",
                                  rewriter.getStringAttr(candidate.explanation))}));
      state.addAttribute("ckl.considered_alternatives", rewriter.getArrayAttr(considered));
      rewriter.setInsertionPoint(edge);
      Operation *replacement = rewriter.create(state);
      rewriter.replaceOp(edge, replacement->getResults());
    }
  }
};

class PlanCompositionsPass : public PassWrapper<PlanCompositionsPass, OperationPass<ModuleOp>> {
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
          op.getKind() == "local-permutation"        ? LocalPermuteOp::getOperationName()
          : op.getKind() == "subgroup-exchange"      ? SubgroupExchangeOp::getOperationName()
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

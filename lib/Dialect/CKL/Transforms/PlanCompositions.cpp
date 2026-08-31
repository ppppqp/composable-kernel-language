#include "ckl/Dialect/CKL/Transforms/Passes.h"

#include "ckl/Core/Composition/Task.h"
#include "ckl/Core/Layout/Distribution.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

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
  if (auto value = attribute.getAs<FlatSymbolRefAttr>("implementation"))
    result.implementationSymbol = value.getValue().str();
  if (auto value = attribute.getAs<StringAttr>("origin"))
    result.origin = value.getValue() == "user"        ? ::ckl::core::AlternativeOrigin::User
                    : value.getValue() == "compiler"  ? ::ckl::core::AlternativeOrigin::Compiler
                    : value.getValue() == "extension" ? ::ckl::core::AlternativeOrigin::Extension
                    : value.getValue() == "library"   ? ::ckl::core::AlternativeOrigin::Library
                                                      : ::ckl::core::AlternativeOrigin::Unspecified;
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

FailureOr<std::string> getLogicalPortName(TaskOp task, StringRef field, std::size_t index,
                                          Operation *anchor) {
  std::optional<std::string> expected;
  for (Attribute value : task.getAlternatives()) {
    auto alternative = mlir::cast<DictionaryAttr>(value);
    auto ports = alternative.getAs<ArrayAttr>(field);
    if (!ports || index >= ports.size()) {
      anchor->emitError("task alternative '")
          << alternative.getAs<StringAttr>("name").getValue() << "' does not describe positional "
          << field << " port " << index;
      return failure();
    }
    auto port = mlir::dyn_cast<DictionaryAttr>(ports[index]);
    auto name = port ? port.getAs<StringAttr>("name") : StringAttr{};
    if (!name) {
      anchor->emitError("task alternative port requires a string name");
      return failure();
    }
    if (!expected)
      expected = name.getValue().str();
    else if (*expected != name.getValue()) {
      anchor->emitError("task alternatives disagree on positional ") << field << " port " << index;
      return failure();
    }
  }
  return expected ? *expected : std::string{};
}

// Selects explicit ckl.task_compose components and graphs inferred from ckl.invoke SSA dataflow.
class SelectAlternativesPass : public PassWrapper<SelectAlternativesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAlternativesPass)
  SelectAlternativesPass() = default;
  SelectAlternativesPass(const SelectAlternativesPass &pass) : PassWrapper(pass) {}
  StringRef getArgument() const final { return "ckl-select-alternatives"; }
  StringRef getDescription() const final {
    return "Globally select alternatives for CKL task-composition components";
  }

  Option<unsigned> maximumCombinations{
      *this, "maximum-combinations",
      llvm::cl::desc("Maximum assignments explored for a non-linear task graph"),
      llvm::cl::init(4096)};

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
    struct PreparedGraph {
      SmallVector<TaskComposeOp> edges;
      std::vector<std::vector<::ckl::core::TaskAlternative>> alternatives;
      std::vector<::ckl::core::TaskGraphEdge> coreEdges;
      ::ckl::core::TaskGraphSelection selection;
    };

    /*
    Discovers connected invoke components from producer-result → consumer-operand edges and treats
    each invocation as a distinct graph node.
    */
    SmallVector<PreparedGraph> graphs;
    struct InvokeEdge {
      std::size_t producer;
      std::size_t consumer;
      unsigned resultIndex;
      unsigned operandIndex;
      std::string producerPort;
      std::string consumerPort;
    };
    struct PreparedInvokeGraph {
      SmallVector<InvokeOp> invokes;
      std::vector<std::vector<::ckl::core::TaskAlternative>> alternatives;
      std::vector<InvokeEdge> edges;
      std::vector<::ckl::core::TaskGraphEdge> coreEdges;
      ::ckl::core::TaskGraphSelection selection;
      std::int64_t subgroupSize;
    };
    SmallVector<PreparedInvokeGraph> invokeGraphs;

    // Infer graph topology from direct SSA flow between task invocations.  Port
    // names remain semantic rather than positional in CKLCore; position is used
    // here only to map an invoke signature to the consistently named ports in
    // every alternative of its task declaration.
    SmallVector<InvokeOp> invokes;
    getOperation().walk([&](InvokeOp op) { invokes.push_back(op); });
    llvm::DenseMap<Operation *, std::size_t> invokeIndex;
    for (auto [index, invoke] : llvm::enumerate(invokes))
      invokeIndex[invoke] = index;
    struct GlobalInvokeEdge {
      std::size_t producer;
      std::size_t consumer;
      unsigned resultIndex;
      unsigned operandIndex;
    };
    SmallVector<GlobalInvokeEdge> invokeEdges;
    std::vector<SmallVector<std::size_t>> adjacency(invokes.size());
    for (auto [producerIndex, producer] : llvm::enumerate(invokes)) {
      for (OpResult result : producer->getResults()) {
        for (OpOperand &use : result.getUses()) {
          auto found = invokeIndex.find(use.getOwner());
          if (found == invokeIndex.end())
            continue;
          std::size_t consumerIndex = found->second;
          invokeEdges.push_back(
              {producerIndex, consumerIndex, result.getResultNumber(), use.getOperandNumber()});
          adjacency[producerIndex].push_back(consumerIndex);
          adjacency[consumerIndex].push_back(producerIndex);
        }
      }
    }

    auto module = getOperation();
    auto subgroupSizeAttr = module->getAttrOfType<IntegerAttr>("ckl.subgroup_size");
    auto registerLimitAttr = module->getAttrOfType<IntegerAttr>("ckl.register_limit");
    auto sharedMemoryLimitAttr = module->getAttrOfType<IntegerAttr>("ckl.shared_memory_limit");
    auto capabilitiesAttr = module->getAttrOfType<ArrayAttr>("ckl.available_capabilities");
    const std::int64_t subgroupSize = subgroupSizeAttr ? subgroupSizeAttr.getInt() : 64;
    const std::int64_t registerLimit = registerLimitAttr ? registerLimitAttr.getInt() : 256;
    const std::int64_t sharedMemoryLimit =
        sharedMemoryLimitAttr ? sharedMemoryLimitAttr.getInt() : 65536;
    if (subgroupSize <= 0 || registerLimit < 0 || sharedMemoryLimit < 0) {
      module.emitError("invoke graph requires a positive subgroup size and non-negative limits");
      signalPassFailure();
      return;
    }
    std::vector<std::string> availableCapabilities;
    if (capabilitiesAttr) {
      for (Attribute value : capabilitiesAttr) {
        auto capability = mlir::dyn_cast<StringAttr>(value);
        if (!capability) {
          module.emitError("ckl.available_capabilities must contain only strings");
          signalPassFailure();
          return;
        }
        availableCapabilities.push_back(capability.getValue().str());
      }
    }

    llvm::DenseSet<std::size_t> visited;
    for (std::size_t root = 0; root < invokes.size(); ++root) {
      if (adjacency[root].empty() || !visited.insert(root).second)
        continue;
      SmallVector<std::size_t> component;
      SmallVector<std::size_t> worklist{root};
      while (!worklist.empty()) {
        std::size_t node = worklist.pop_back_val();
        component.push_back(node);
        for (std::size_t neighbor : adjacency[node])
          if (visited.insert(neighbor).second)
            worklist.push_back(neighbor);
      }
      llvm::DenseMap<std::size_t, std::size_t> localIndex;
      PreparedInvokeGraph preparedGraph;
      preparedGraph.subgroupSize = subgroupSize;
      for (auto [local, global] : llvm::enumerate(component)) {
        localIndex[global] = local;
        preparedGraph.invokes.push_back(invokes[global]);
      }
      preparedGraph.alternatives.resize(component.size());
      std::vector<::ckl::core::TaskGraphNode> nodes;
      for (auto [local, invoke] : llvm::enumerate(preparedGraph.invokes)) {
        auto task = SymbolTable::lookupNearestSymbolFrom<TaskOp>(invoke, invoke.getCalleeAttr());
        try {
          for (Attribute value : task.getAlternatives()) {
            auto dictionary = mlir::cast<DictionaryAttr>(value);
            // honors pinned alternative if present, otherwise considers all alternatives
            if (auto pinned = invoke.getAlternativeAttr();
                pinned && dictionary.getAs<StringAttr>("name").getValue() != pinned.getValue())
              continue;
            auto imported = importAlternative(task, dictionary, invoke);
            if (failed(imported)) {
              signalPassFailure();
              return;
            }
            preparedGraph.alternatives[local].push_back(std::move(*imported));
          }
        } catch (const std::exception &error) {
          invoke.emitError("failed to import invoke alternatives: ") << error.what();
          signalPassFailure();
          return;
        }
        nodes.push_back({task.getSymName().str() + "#" + std::to_string(local),
                         preparedGraph.alternatives[local]});
      }
      for (const GlobalInvokeEdge &edge : invokeEdges) {
        auto producer = localIndex.find(edge.producer);
        auto consumer = localIndex.find(edge.consumer);
        if (producer == localIndex.end() || consumer == localIndex.end())
          continue;
        InvokeOp producerOp = invokes[edge.producer];
        InvokeOp consumerOp = invokes[edge.consumer];
        if (!mlir::isa<TileType>(producerOp->getResult(edge.resultIndex).getType())) {
          producerOp.emitError("direct invoke composition currently requires CKL tile values");
          signalPassFailure();
          return;
        }
        auto producerTask =
            SymbolTable::lookupNearestSymbolFrom<TaskOp>(producerOp, producerOp.getCalleeAttr());
        auto consumerTask =
            SymbolTable::lookupNearestSymbolFrom<TaskOp>(consumerOp, consumerOp.getCalleeAttr());
        auto producerPort =
            getLogicalPortName(producerTask, "outputs", edge.resultIndex, producerOp);
        auto consumerPort =
            getLogicalPortName(consumerTask, "inputs", edge.operandIndex, consumerOp);
        if (failed(producerPort) || failed(consumerPort)) {
          signalPassFailure();
          return;
        }
        preparedGraph.edges.push_back({producer->second, consumer->second, edge.resultIndex,
                                       edge.operandIndex, *producerPort, *consumerPort});
        preparedGraph.coreEdges.push_back(
            {producer->second, consumer->second, *producerPort, *consumerPort});
      }
      auto decision = ::ckl::core::selectTaskGraph(nodes, preparedGraph.coreEdges, subgroupSize,
                                                   registerLimit, sharedMemoryLimit,
                                                   maximumCombinations, availableCapabilities);
      if (!decision.selected) {
        auto diagnostic = preparedGraph.invokes.front().emitError(
            "no proven optimal selection for direct invoke graph");
        for (const std::string &message : decision.diagnostics)
          diagnostic.attachNote(preparedGraph.invokes.front().getLoc()) << message;
        signalPassFailure();
        return;
      }
      preparedGraph.selection = std::move(*decision.selected);
      invokeGraphs.push_back(std::move(preparedGraph));
    }

    // Solve all chains before rewriting any of them: failure leaves the IR
    // entirely unresolved instead of committing a partial global decision.
    for (TaskComposeOp root : boundaries) {
      if (mlir::isa_and_nonnull<TaskComposeOp>(root.getInput().getDefiningOp()))
        continue;
      SmallVector<TaskComposeOp> chain{root};
      TaskComposeOp current = root;
      while (current.getResult().hasOneUse()) {
        auto next = mlir::dyn_cast<TaskComposeOp>(*current.getResult().getUsers().begin());
        if (!next)
          break;
        chain.push_back(next);
        current = next;
      }
      SmallVector<TaskComposeOp> successors;
      for (Operation *user : current.getResult().getUsers())
        if (auto next = mlir::dyn_cast<TaskComposeOp>(user))
          successors.push_back(next);
      if (successors.size() > 1) {
        SmallVector<TaskComposeOp> graphEdges;
        SmallVector<TaskComposeOp> worklist{root};
        llvm::DenseMap<Operation *, std::size_t> consumerNode;
        SmallVector<TaskOp> tasks;
        tasks.push_back(SymbolTable::lookupNearestSymbolFrom<TaskOp>(root, root.getProducerAttr()));
        std::vector<::ckl::core::TaskGraphEdge> coreEdges;
        while (!worklist.empty()) {
          TaskComposeOp edge = worklist.pop_back_val();
          std::size_t producerNode = 0;
          if (auto parent = mlir::dyn_cast_or_null<TaskComposeOp>(edge.getInput().getDefiningOp()))
            producerNode = consumerNode.lookup(parent);
          auto producerTask =
              SymbolTable::lookupNearestSymbolFrom<TaskOp>(edge, edge.getProducerAttr());
          if (producerTask != tasks[producerNode]) {
            edge.emitError("task graph edge has inconsistent producer invocation");
            signalPassFailure();
            return;
          }
          const std::size_t consumer = tasks.size();
          tasks.push_back(
              SymbolTable::lookupNearestSymbolFrom<TaskOp>(edge, edge.getConsumerAttr()));
          consumerNode[edge] = consumer;
          graphEdges.push_back(edge);
          coreEdges.push_back(
              {producerNode, consumer, edge.getProducerPort().str(), edge.getConsumerPort().str()});
          for (Operation *user : edge.getResult().getUsers())
            if (auto child = mlir::dyn_cast<TaskComposeOp>(user))
              worklist.push_back(child);
        }

        auto subgroupSize = root.getSubgroupSize();
        auto registerLimit = root.getRegisterLimit();
        auto sharedMemoryLimit = root.getSharedMemoryLimit();
        auto capabilitiesAttr = root.getCapabilitiesAttr();
        for (TaskComposeOp edge : graphEdges) {
          if (edge.getSubgroupSize() != subgroupSize || edge.getRegisterLimit() != registerLimit ||
              edge.getSharedMemoryLimit() != sharedMemoryLimit ||
              edge.getCapabilitiesAttr() != capabilitiesAttr) {
            edge.emitError("task graph requires consistent planning limits and capabilities");
            signalPassFailure();
            return;
          }
        }

        PreparedGraph item;
        item.edges = graphEdges;
        item.coreEdges = coreEdges;
        item.alternatives.resize(tasks.size());
        try {
          for (std::size_t node = 0; node < tasks.size(); ++node)
            for (Attribute value : tasks[node].getAlternatives()) {
              auto imported =
                  importAlternative(tasks[node], mlir::cast<DictionaryAttr>(value), root);
              if (failed(imported)) {
                signalPassFailure();
                return;
              }
              item.alternatives[node].push_back(std::move(*imported));
            }
        } catch (const std::exception &error) {
          root.emitError("failed to import task graph alternatives: ") << error.what();
          signalPassFailure();
          return;
        }
        std::vector<::ckl::core::TaskGraphNode> nodes;
        for (std::size_t node = 0; node < tasks.size(); ++node)
          nodes.push_back({tasks[node].getSymName().str() + "#" + std::to_string(node),
                           item.alternatives[node]});
        std::vector<std::string> capabilities;
        for (Attribute value : root.getCapabilities())
          capabilities.push_back(mlir::cast<StringAttr>(value).getValue().str());
        auto decision =
            ::ckl::core::selectTaskGraph(nodes, coreEdges, subgroupSize, registerLimit,
                                         sharedMemoryLimit, maximumCombinations, capabilities);
        if (!decision.selected) {
          auto diagnostic = root.emitError("no proven optimal task-graph selection");
          for (const std::string &message : decision.diagnostics)
            diagnostic.attachNote(root.getLoc()) << message;
          signalPassFailure();
          return;
        }
        item.selection = std::move(*decision.selected);
        graphs.push_back(std::move(item));
        continue;
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
            auto imported = importAlternative(producer, mlir::cast<DictionaryAttr>(value), root);
            if (failed(imported)) {
              signalPassFailure();
              return;
            }
            single.producers.push_back(std::move(*imported));
          }
          for (Attribute value : consumer.getAlternatives()) {
            auto imported = importAlternative(consumer, mlir::cast<DictionaryAttr>(value), root);
            if (failed(imported)) {
              signalPassFailure();
              return;
            }
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
        if (edge.getSubgroupSize() != subgroupSize || edge.getRegisterLimit() != registerLimit ||
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
            auto imported =
                importAlternative(tasks[stage], mlir::cast<DictionaryAttr>(value), chain.front());
            if (failed(imported)) {
              signalPassFailure();
              return;
            }
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
        std::string output = stage + 1 == tasks.size() ? "" : chain[stage].getProducerPort().str();
        stages.push_back({tasks[stage].getSymName().str() + "#" + std::to_string(stage),
                          item.alternatives[stage], std::move(input), std::move(output)});
      }
      std::vector<std::string> capabilities;
      for (Attribute value : chain.front().getCapabilities())
        capabilities.push_back(mlir::cast<StringAttr>(value).getValue().str());
      auto decision = ::ckl::core::selectLinearPipeline(stages, subgroupSize, registerLimit,
                                                        sharedMemoryLimit, capabilities);
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
    for (PreparedInvokeGraph &graph : invokeGraphs) {
      SmallVector<Attribute> provenance;
      for (const std::string &entry : graph.selection.provenance)
        provenance.push_back(rewriter.getStringAttr(entry));
      for (auto [node, invoke] : llvm::enumerate(graph.invokes)) {
        const auto &alternative = graph.alternatives[node][graph.selection.alternatives[node]];
        invoke->setAttr("alternative", rewriter.getStringAttr(alternative.name));
        invoke->setAttr("ckl.implementation_id",
                        rewriter.getStringAttr(alternative.implementationId.empty()
                                                   ? alternative.task + ":" + alternative.name
                                                   : alternative.implementationId));
        if (!alternative.implementationSymbol.empty())
          invoke->setAttr("ckl.implementation",
                          FlatSymbolRefAttr::get(&getContext(), alternative.implementationSymbol));
        invoke->setAttr("ckl.graph_score", rewriter.getI64IntegerAttr(graph.selection.score));
        invoke->setAttr("ckl.graph_combinations_explored",
                        rewriter.getI64IntegerAttr(graph.selection.combinationsExplored));
        invoke->setAttr("ckl.graph_provenance", rewriter.getArrayAttr(provenance));
      }
      for (auto [edgeIndex, edge] : llvm::enumerate(graph.edges)) {
        InvokeOp producer = graph.invokes[edge.producer];
        InvokeOp consumer = graph.invokes[edge.consumer];
        const auto &producerAlternative =
            graph.alternatives[edge.producer][graph.selection.alternatives[edge.producer]];
        const auto &consumerAlternative =
            graph.alternatives[edge.consumer][graph.selection.alternatives[edge.consumer]];
        auto source = llvm::find_if(producerAlternative.outputs, [&](const auto &port) {
          return port.name == edge.producerPort;
        });
        auto target = llvm::find_if(consumerAlternative.inputs, [&](const auto &port) {
          return port.name == edge.consumerPort;
        });
        Value input = consumer->getOperand(edge.operandIndex);
        OperationState state(consumer.getLoc(), ComposeOp::getOperationName());
        state.addOperands(input);
        state.addTypes(input.getType());
        state.addAttribute(
            "source",
            DistributionAttr::get(&getContext(), ::ckl::core::serialize(source->distribution)));
        state.addAttribute(
            "target",
            DistributionAttr::get(&getContext(), ::ckl::core::serialize(target->distribution)));
        state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(graph.subgroupSize));
        state.addAttribute("ckl.producer_alternative",
                           rewriter.getStringAttr(producerAlternative.name));
        state.addAttribute("ckl.consumer_alternative",
                           rewriter.getStringAttr(consumerAlternative.name));
        state.addAttribute("ckl.graph_score", rewriter.getI64IntegerAttr(graph.selection.score));
        state.addAttribute("ckl.graph_edge", rewriter.getI64IntegerAttr(edgeIndex));
        state.addAttribute("ckl.graph_combinations_explored",
                           rewriter.getI64IntegerAttr(graph.selection.combinationsExplored));
        state.addAttribute("ckl.graph_provenance", rewriter.getArrayAttr(provenance));
        rewriter.setInsertionPoint(consumer);
        Operation *composition = rewriter.create(state);
        consumer->setOperand(edge.operandIndex, composition->getResult(0));
      }
    }
    for (Prepared &pipeline : prepared) {
      SmallVector<Attribute> provenance;
      for (const std::string &entry : pipeline.selection.provenance)
        provenance.push_back(rewriter.getStringAttr(entry));
      for (std::size_t index = 0; index < pipeline.edges.size(); ++index) {
        TaskComposeOp edge = pipeline.edges[index];
        const auto &producer = pipeline.alternatives[index][pipeline.selection.alternatives[index]];
        const auto &consumer =
            pipeline.alternatives[index + 1][pipeline.selection.alternatives[index + 1]];
        auto source = llvm::find_if(producer.outputs, [&](const auto &port) {
          return port.name == edge.getProducerPort();
        });
        auto target = llvm::find_if(
            consumer.inputs, [&](const auto &port) { return port.name == edge.getConsumerPort(); });
        OperationState state(edge.getLoc(), ComposeOp::getOperationName());
        state.addOperands(edge.getInput());
        state.addTypes(edge.getResult().getType());
        state.addAttribute(
            "source",
            DistributionAttr::get(&getContext(), ::ckl::core::serialize(source->distribution)));
        state.addAttribute(
            "target",
            DistributionAttr::get(&getContext(), ::ckl::core::serialize(target->distribution)));
        state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(edge.getSubgroupSize()));
        state.addAttribute("ckl.producer_alternative", rewriter.getStringAttr(producer.name));
        state.addAttribute("ckl.consumer_alternative", rewriter.getStringAttr(consumer.name));
        if (!producer.implementationSymbol.empty())
          state.addAttribute("ckl.producer_implementation",
                             FlatSymbolRefAttr::get(&getContext(), producer.implementationSymbol));
        if (!consumer.implementationSymbol.empty())
          state.addAttribute("ckl.consumer_implementation",
                             FlatSymbolRefAttr::get(&getContext(), consumer.implementationSymbol));
        state.addAttribute("ckl.pipeline_score",
                           rewriter.getI64IntegerAttr(pipeline.selection.score));
        state.addAttribute("ckl.pipeline_stage", rewriter.getI64IntegerAttr(index));
        state.addAttribute("ckl.pipeline_provenance", rewriter.getArrayAttr(provenance));
        rewriter.setInsertionPoint(edge);
        Operation *replacement = rewriter.create(state);
        rewriter.replaceOp(edge, replacement->getResults());
      }
    }
    for (PreparedGraph &graph : graphs) {
      SmallVector<Attribute> provenance;
      for (const std::string &entry : graph.selection.provenance)
        provenance.push_back(rewriter.getStringAttr(entry));
      for (std::size_t index = 0; index < graph.edges.size(); ++index) {
        TaskComposeOp edge = graph.edges[index];
        const auto &coreEdge = graph.coreEdges[index];
        const auto &producer =
            graph.alternatives[coreEdge.producer][graph.selection.alternatives[coreEdge.producer]];
        const auto &consumer =
            graph.alternatives[coreEdge.consumer][graph.selection.alternatives[coreEdge.consumer]];
        auto source = llvm::find_if(producer.outputs, [&](const auto &port) {
          return port.name == edge.getProducerPort();
        });
        auto target = llvm::find_if(
            consumer.inputs, [&](const auto &port) { return port.name == edge.getConsumerPort(); });
        OperationState state(edge.getLoc(), ComposeOp::getOperationName());
        state.addOperands(edge.getInput());
        state.addTypes(edge.getResult().getType());
        state.addAttribute(
            "source",
            DistributionAttr::get(&getContext(), ::ckl::core::serialize(source->distribution)));
        state.addAttribute(
            "target",
            DistributionAttr::get(&getContext(), ::ckl::core::serialize(target->distribution)));
        state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(edge.getSubgroupSize()));
        state.addAttribute("ckl.producer_alternative", rewriter.getStringAttr(producer.name));
        state.addAttribute("ckl.consumer_alternative", rewriter.getStringAttr(consumer.name));
        if (!producer.implementationSymbol.empty())
          state.addAttribute("ckl.producer_implementation",
                             FlatSymbolRefAttr::get(&getContext(), producer.implementationSymbol));
        if (!consumer.implementationSymbol.empty())
          state.addAttribute("ckl.consumer_implementation",
                             FlatSymbolRefAttr::get(&getContext(), consumer.implementationSymbol));
        state.addAttribute("ckl.graph_score", rewriter.getI64IntegerAttr(graph.selection.score));
        state.addAttribute("ckl.graph_edge", rewriter.getI64IntegerAttr(index));
        state.addAttribute("ckl.graph_combinations_explored",
                           rewriter.getI64IntegerAttr(graph.selection.combinationsExplored));
        state.addAttribute("ckl.graph_provenance", rewriter.getArrayAttr(provenance));
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
      auto source = llvm::find_if(
          producer.outputs, [&](const auto &port) { return port.name == edge.getProducerPort(); });
      auto target = llvm::find_if(
          consumer.inputs, [&](const auto &port) { return port.name == edge.getConsumerPort(); });
      OperationState state(edge.getLoc(), ComposeOp::getOperationName());
      state.addOperands(edge.getInput());
      state.addTypes(edge.getResult().getType());
      state.addAttribute("source", DistributionAttr::get(&getContext(), ::ckl::core::serialize(
                                                                            source->distribution)));
      state.addAttribute("target", DistributionAttr::get(&getContext(), ::ckl::core::serialize(
                                                                            target->distribution)));
      state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(edge.getSubgroupSize()));
      state.addAttribute("ckl.producer_alternative", rewriter.getStringAttr(producer.name));
      state.addAttribute("ckl.consumer_alternative", rewriter.getStringAttr(consumer.name));
      if (!producer.implementationSymbol.empty())
        state.addAttribute("ckl.producer_implementation",
                           FlatSymbolRefAttr::get(&getContext(), producer.implementationSymbol));
      if (!consumer.implementationSymbol.empty())
        state.addAttribute("ckl.consumer_implementation",
                           FlatSymbolRefAttr::get(&getContext(), consumer.implementationSymbol));
      state.addAttribute("ckl.producer_implementation_id",
                         rewriter.getStringAttr(producer.implementationId.empty()
                                                    ? producer.task + ":" + producer.name
                                                    : producer.implementationId));
      state.addAttribute("ckl.consumer_implementation_id",
                         rewriter.getStringAttr(consumer.implementationId.empty()
                                                    ? consumer.task + ":" + consumer.name
                                                    : consumer.implementationId));
      SmallVector<Attribute> considered;
      for (const auto &candidate : single.decision.considered)
        considered.push_back(rewriter.getDictionaryAttr(
            {rewriter.getNamedAttr(
                 "producer",
                 rewriter.getStringAttr(single.producers[candidate.producerAlternative].name)),
             rewriter.getNamedAttr(
                 "consumer",
                 rewriter.getStringAttr(single.consumers[candidate.consumerAlternative].name)),
             rewriter.getNamedAttr("score", rewriter.getI64IntegerAttr(candidate.score)),
             rewriter.getNamedAttr("explanation", rewriter.getStringAttr(candidate.explanation))}));
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

// removes ckl.compose and ckl.convert_layout
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
      auto createPhase = [&](StringRef name, Value input) {
        OperationState state(op.getLoc(), name);
        state.addOperands(input);
        state.addTypes(op.getResult().getType());
        state.addAttribute("source", op.getSource());
        state.addAttribute("target", op.getTarget());
        state.addAttribute("subgroup_size", rewriter.getI64IntegerAttr(op.getSubgroupSize()));
        if (auto reason = op.getReasonAttr())
          state.addAttribute("reason", reason);
        if (auto count = op.getMoveCountAttr())
          state.addAttribute("move_count", count);
        return rewriter.create(state);
      };
      if (op.getKind() == "shared-memory-exchange" || op.getKind() == "global-memory-exchange") {
        rewriter.setInsertionPoint(op);
        const bool shared = op.getKind() == "shared-memory-exchange";
        Operation *store = createPhase(shared ? SharedStoreOp::getOperationName()
                                              : GlobalStoreOp::getOperationName(),
                                       op.getInput());
        Operation *boundary = createPhase(shared ? WorkgroupBarrierOp::getOperationName()
                                                 : KernelBoundaryOp::getOperationName(),
                                          store->getResult(0));
        Operation *load = createPhase(shared ? SharedLoadOp::getOperationName()
                                             : GlobalLoadOp::getOperationName(),
                                      boundary->getResult(0));
        rewriter.replaceOp(op, load->getResults());
        continue;
      }
      StringRef operationName =
          op.getKind() == "local-permutation"   ? LocalPermuteOp::getOperationName()
          : op.getKind() == "subgroup-exchange" ? SubgroupExchangeOp::getOperationName()
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
  registerEnumerateAlternativesPass();
  PassRegistration<SelectAlternativesPass>();
  PassRegistration<PlanCompositionsPass>();
  PassRegistration<ScheduleConversionsPass>();
}

} // namespace mlir::ckl

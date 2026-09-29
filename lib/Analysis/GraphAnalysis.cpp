#include "ckl/Analysis/EffectAnalysis.h"

#include "ckl/Core/MemoryEffects.h"
#include "ckl/Core/MemoryPlanner.h"
#include "ckl/Dialect/CKL/IR/CKLInterfaces.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <map>
#include <string>

using namespace mlir;
using namespace mlir::ckl;

namespace {

// operational form of one access after a materialized kernel (after dispatch)
struct BoundAccess {
  Value resource;
  unsigned bits = 0;
  StringRef region;
};

// represents a graph node before it is materialized as ckl.graph_node
struct NodeInfo {
  Operation *source = nullptr;
  StringRef kind;
  FlatSymbolRefAttr kernel;
  SmallVector<BoundAccess> accesses;
  ArrayAttr materializedAccesses;
  ArrayAttr orderingScopes;
  bool unknown = false;
  bool synchronizes = false;
  bool nestedControl = false;
};

struct PlannedResourceInfo {
  std::string name;
  std::string device;
  std::string addressSpace;
  std::size_t bytes = 0;
  SmallVector<unsigned> users;
  bool temporary = false;
};

unsigned parseEffects(ArrayAttr effects) {
  unsigned bits = 0;
  for (StringAttr effect : effects.getAsRange<StringAttr>()) {
    bits |= parseAccessEffect(effect.getValue());
  }
  return bits;
}

std::string printLocation(Location location) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  location.print(stream);
  return storage;
}

std::string resourceName(Value value, func::FuncOp function,
                         DenseMap<Operation *, unsigned> &allocationIds) {
  value = getBaseResource(value);
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() == &function.getBody().front())
      return ("arg" + Twine(argument.getArgNumber())).str();
  }
  if (Operation *definition = value.getDefiningOp()) {
    if (isAllocationValue(value)) {
      auto [iterator, inserted] = allocationIds.try_emplace(definition, allocationIds.size());
      (void)inserted;
      return ("alloc" + Twine(iterator->second)).str();
    }
  }
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  value.printAsOperand(stream, OpPrintingFlags());
  return storage;
}

FailureOr<std::size_t> getStaticResourceBytes(Value resource) {
  auto type = dyn_cast<MemRefType>(resource.getType());
  if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity())
    return failure();
  unsigned elementBits = 0;
  if (auto integer = dyn_cast<IntegerType>(type.getElementType()))
    elementBits = integer.getWidth();
  else if (auto floating = dyn_cast<FloatType>(type.getElementType()))
    elementBits = floating.getWidth();
  else
    return failure();
  std::size_t elementBytes = (elementBits + 7) / 8;
  std::size_t elements = static_cast<std::size_t>(type.getNumElements());
  if (!elementBytes || !elements ||
      elements > std::numeric_limits<std::size_t>::max() / elementBytes ||
      elements * elementBytes > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
    return failure();
  return elements * elementBytes;
}

std::string getAddressSpace(Value resource) {
  auto type = dyn_cast<MemRefType>(resource.getType());
  if (!type || !type.getMemorySpace())
    return "global";
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  type.getMemorySpace().print(stream);
  return storage;
}

FailureOr<int> getDeviceOrdinal(StringRef device) {
  if (!device.consume_front("cuda:"))
    return failure();
  int ordinal = 0;
  if (device.getAsInteger(10, ordinal) || ordinal < 0)
    return failure();
  return ordinal;
}

DictionaryAttr makeUnavailablePlan(Builder &builder, StringRef reason) {
  // create a dictionary attribute to indicate that the memory plan is unavailable
  return builder.getDictionaryAttr({
      builder.getNamedAttr("reason", builder.getStringAttr(reason)),
      builder.getNamedAttr("status", builder.getStringAttr("unavailable")),
  });
}

void attachMemoryPlan(GraphOp graph, ArrayRef<NodeInfo> nodes,
                      const std::map<std::pair<unsigned, unsigned>, SmallVector<Attribute>> &edges,
                      func::FuncOp host, DenseMap<Operation *, unsigned> &allocationIds,
                      Builder &builder) {
  constexpr std::size_t allocationAlignment = 256;

  // value -> index in resources
  DenseMap<Value, unsigned> indices;
  SmallVector<PlannedResourceInfo> resources;
  std::string unavailableReason;

  for (auto [nodeId, node] : llvm::enumerate(nodes)) {
    // walk only dispatch nodes
    if (node.kind != "dispatch")
      continue;
    auto deviceAttr = node.source->getAttrOfType<StringAttr>("device");
    StringRef device = deviceAttr ? deviceAttr.getValue() : StringRef();
    for (const BoundAccess &access : node.accesses) {
      Value value = getBaseResource(access.resource);
      auto [iterator, inserted] = indices.try_emplace(value, resources.size());
      if (inserted) {
        // if this is the first time we see this resource, create a new PlannedResourceInfo
        PlannedResourceInfo info;
        info.name = resourceName(value, host, allocationIds);
        info.device = device.str();
        info.addressSpace = getAddressSpace(value);
        info.temporary = isAllocationValue(value);
        if (FailureOr<std::size_t> bytes = getStaticResourceBytes(value); succeeded(bytes))
          info.bytes = *bytes;
        else if (info.temporary && unavailableReason.empty())
          unavailableReason = "temporary resource " + info.name +
                              " requires a static identity-layout integer or float memref";
        resources.push_back(std::move(info));
      }
      PlannedResourceInfo &info = resources[iterator->second];
      if (info.device != device && unavailableReason.empty())
        unavailableReason = "resource " + info.name + " is used on multiple devices";
      if (info.users.empty() || info.users.back() != nodeId)
        // if one resource reads and writes the same resource, the node ID is recorded only once
        info.users.push_back(nodeId);
    }
  }

  SmallVector<Attribute> materializedResources;
  // finalize the resource information and attach it to the graph
  for (const PlannedResourceInfo &resource : resources) {
    SmallVector<NamedAttribute> attributes = {
        builder.getNamedAttr("address_space", builder.getStringAttr(resource.addressSpace)),
        builder.getNamedAttr("alignment", builder.getI64IntegerAttr(allocationAlignment)),
        builder.getNamedAttr("device", builder.getStringAttr(resource.device)),
        builder.getNamedAttr("kind",
                             builder.getStringAttr(resource.temporary ? "temporary" : "external")),
        builder.getNamedAttr("name", builder.getStringAttr(resource.name)),
    };
    if (resource.bytes)
      attributes.push_back(
          builder.getNamedAttr("bytes", builder.getI64IntegerAttr(resource.bytes)));
    SmallVector<int64_t> users(resource.users.begin(), resource.users.end());
    attributes.push_back(builder.getNamedAttr("users", builder.getDenseI64ArrayAttr(users)));
    materializedResources.push_back(builder.getDictionaryAttr(attributes));
  }
  graph->setAttr("ckl.resources", builder.getArrayAttr(materializedResources));

  if (!unavailableReason.empty()) {
    graph->setAttr("ckl.memory_plan", makeUnavailablePlan(builder, unavailableReason));
    return;
  }

  std::vector<planning::Resource> plannerResources;
  SmallVector<unsigned> plannerToGraphResource;
  for (auto [resourceId, resource] : llvm::enumerate(resources)) {
    if (!resource.temporary)
      continue;
    FailureOr<int> device = getDeviceOrdinal(resource.device);
    if (failed(device)) {
      graph->setAttr("ckl.memory_plan",
                     makeUnavailablePlan(builder, "only cuda:<ordinal> devices are supported"));
      return;
    }
    std::vector<planning::NodeId> users(resource.users.begin(), resource.users.end());
    plannerResources.push_back({resource.name, resource.bytes, allocationAlignment,
                                resource.addressSpace, *device, planning::ResourceKind::Temporary,
                                std::move(users)});
    plannerToGraphResource.push_back(resourceId);
  }

  std::vector<planning::GraphEdge> plannerEdges;
  for (const auto &[edge, _] : edges) {
    plannerEdges.push_back({edge.first, edge.second});
  }

  planning::MemoryPlanResult planned =
      planning::MemoryPlanner().tryPlan(nodes.size(), plannerEdges, plannerResources);
  if (!planned) {
    graph->setAttr("ckl.memory_plan", makeUnavailablePlan(builder, planned.error));
    return;
  }
  const planning::MemoryPlan &plan = *planned.value;

  SmallVector<Attribute> heaps;
  for (const planning::Heap &heap : plan.heaps)
    heaps.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("address_space", builder.getStringAttr(heap.addressSpace)),
        builder.getNamedAttr("alignment", builder.getI64IntegerAttr(heap.alignment)),
        builder.getNamedAttr("bytes", builder.getI64IntegerAttr(heap.bytes)),
        builder.getNamedAttr("device", builder.getI64IntegerAttr(heap.device)),
    }));

  SmallVector<Attribute> assignments;
  for (const planning::Assignment &assignment : plan.assignments) {
    const PlannedResourceInfo &resource = resources[plannerToGraphResource[assignment.resource]];
    assignments.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("bytes", builder.getI64IntegerAttr(assignment.bytes)),
        builder.getNamedAttr("heap", builder.getI64IntegerAttr(assignment.heap)),
        builder.getNamedAttr("offset", builder.getI64IntegerAttr(assignment.offset)),
        builder.getNamedAttr("resource", builder.getStringAttr(resource.name)),
        builder.getNamedAttr("slot", builder.getI64IntegerAttr(assignment.slot)),
    }));
  }

  SmallVector<Attribute> decisions;
  for (const planning::ReuseDecision &decision : plan.decisions) {
    SmallVector<Attribute> sharesWith;
    for (planning::ResourceId previous : decision.sharesWith)
      sharesWith.push_back(builder.getStringAttr(resources[plannerToGraphResource[previous]].name));
    decisions.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("reason", builder.getStringAttr(decision.reason)),
        builder.getNamedAttr(
            "resource",
            builder.getStringAttr(resources[plannerToGraphResource[decision.resource]].name)),
        builder.getNamedAttr("shares_with", builder.getArrayAttr(sharesWith)),
        builder.getNamedAttr("slot", builder.getI64IntegerAttr(decision.slot)),
    }));
  }

  graph->setAttr(
      "ckl.memory_plan",
      builder.getDictionaryAttr({
          builder.getNamedAttr("assignments", builder.getArrayAttr(assignments)),
          builder.getNamedAttr("baseline_bytes", builder.getI64IntegerAttr(plan.baselineBytes)),
          builder.getNamedAttr("decisions", builder.getArrayAttr(decisions)),
          builder.getNamedAttr("heaps", builder.getArrayAttr(heaps)),
          builder.getNamedAttr("planned_bytes", builder.getI64IntegerAttr(plan.plannedBytes)),
          builder.getNamedAttr("status", builder.getStringAttr("planned")),
      }));
}

DictionaryAttr makeReason(Builder &builder, StringRef kind, StringRef detail, StringRef resource,
                          StringRef alias, Operation *from, Operation *to) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr("alias", builder.getStringAttr(alias)),
      builder.getNamedAttr("detail", builder.getStringAttr(detail)),
      builder.getNamedAttr("from_location", builder.getStringAttr(printLocation(from->getLoc()))),
      builder.getNamedAttr("kind", builder.getStringAttr(kind)),
      builder.getNamedAttr("resource", builder.getStringAttr(resource)),
      builder.getNamedAttr("to_location", builder.getStringAttr(printLocation(to->getLoc()))),
  });
}

FailureOr<NodeInfo> bindDispatch(DispatchOpInterface dispatch, func::FuncOp host,
                                 DenseMap<Operation *, unsigned> &allocationIds, Builder &builder) {

  // for the dispatch, resolve the referenced kernel and its effect summary
  auto kernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(dispatch.getOperation(),
                                                                   dispatch.getKernelSymbol());
  if (!kernel)
    return failure();
  auto summary = kernel->getAttrOfType<DictionaryAttr>("ckl.effect_summary");
  if (!summary)
    return failure();

  NodeInfo node;
  node.source = dispatch.getOperation();
  node.kind = "dispatch";
  node.kernel = dispatch.getKernelSymbol();
  node.orderingScopes = summary.getAs<ArrayAttr>("ordering_scopes");
  node.unknown = summary.getAs<BoolAttr>("unknown").getValue();
  node.synchronizes = summary.getAs<BoolAttr>("synchronizes").getValue();
  node.nestedControl = dispatch->getParentOp() != host;

  SmallVector<Attribute> accesses;
  auto summaryAccesses = summary.getAs<ArrayAttr>("accesses");
  OperandRange actuals = dispatch.getDispatchArguments();
  for (DictionaryAttr access : summaryAccesses.getAsRange<DictionaryAttr>()) {
    // map every summarized access to its actual dispatch operand
    unsigned argument = access.getAs<IntegerAttr>("arg").getInt();
    if (argument >= actuals.size())
      return failure();
    Value resource = getBaseResource(actuals[argument]);
    ArrayAttr effects = access.getAs<ArrayAttr>("effects");
    StringAttr region = access.getAs<StringAttr>("region");
    // record the access in the node and materialize it for the graph node
    node.accesses.push_back({resource, parseEffects(effects), region.getValue()});
    accesses.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("arg", builder.getI64IntegerAttr(argument)),
        builder.getNamedAttr("effects", effects),
        builder.getNamedAttr("region", region),
        builder.getNamedAttr("resource",
                             builder.getStringAttr(resourceName(resource, host, allocationIds))),
    }));
  }
  node.materializedAccesses = builder.getArrayAttr(accesses);
  return node;
}

LogicalResult appendLifetimeNode(Operation *operation, func::FuncOp host,
                                 DenseMap<Operation *, unsigned> &allocationIds, Builder &builder,
                                 SmallVectorImpl<NodeInfo> &nodes) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
  if (!interface)
    return success();
  // examines every non-dispatch operation through MemoryEffectOpInterface to determine if it has
  // lifetime effects
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);

  NodeInfo node;
  node.source = operation;
  node.orderingScopes = builder.getArrayAttr({});
  node.nestedControl = operation->getParentOp() != host;
  bool allocates = false;
  bool frees = false;
  SmallVector<Attribute> accesses;
  for (const MemoryEffects::EffectInstance &effect : effects) {
    unsigned bits = 0;
    if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
      bits = Allocate;
      allocates = true;
    } else if (isa<MemoryEffects::Free>(effect.getEffect())) {
      bits = Free;
      frees = true;
    } else {
      continue;
    }
    Value resource = effect.getValue();
    if (!resource)
      return operation->emitError("lifetime effect has no addressable resource");
    resource = getBaseResource(resource);
    // record the access in the node
    node.accesses.push_back({resource, bits, "whole"});
    accesses.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr(
            "effects", builder.getArrayAttr({builder.getStringAttr(stringifyAccessEffect(bits))})),
        builder.getNamedAttr("region", builder.getStringAttr("whole")),
        builder.getNamedAttr("resource",
                             builder.getStringAttr(resourceName(resource, host, allocationIds))),
    }));
  }
  if (accesses.empty())
    return success();
  node.kind = allocates && frees ? "lifetime" : allocates ? "alloc" : "free";
  node.materializedAccesses = builder.getArrayAttr(accesses);
  nodes.push_back(std::move(node));
  return success();
}

LogicalResult buildGraph(ModuleOp module, func::FuncOp function) {
  Builder builder(module.getContext());
  DenseMap<Operation *, unsigned> allocationIds;
  SmallVector<NodeInfo> nodes;

  WalkResult result = function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (auto dispatch = dyn_cast<DispatchOpInterface>(operation)) {
      // if see a dispatch, bind it to its kernel and effect summary and record it as a graph node
      FailureOr<NodeInfo> node = bindDispatch(dispatch, function, allocationIds, builder);
      if (failed(node))
        return WalkResult::interrupt();
      nodes.push_back(std::move(*node));
    } else if (failed(appendLifetimeNode(operation, function, allocationIds, builder, nodes)))
      // if see a non-dispatch operation with lifetime effects, record it as a graph node
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return function.emitError("failed to bind a dispatch effect summary");
  if (llvm::none_of(nodes, [](const NodeInfo &node) { return node.kind == "dispatch"; }))
    // if there are no dispatches, there is no graph to build
    return success();

  DenseMap<Operation *, unsigned> nodeIds;
  for (auto [id, node] : llvm::enumerate(nodes))
    nodeIds[node.source] = id;

  // edges are stored in a map from (from, to) node ids to a list of reasons for the edge
  using Edge = std::pair<unsigned, unsigned>;
  std::map<Edge, SmallVector<Attribute>>
      edges; // an edge can have multiple reasons, e.g. multiple conflicting accesses
  auto addReason = [&](unsigned from, unsigned to, DictionaryAttr reason) {
    if (from != to)
      edges[{from, to}].push_back(reason);
  };

  for (unsigned to = 0; to < nodes.size(); ++to) {
    if (auto dispatch = dyn_cast<DispatchOpInterface>(nodes[to].source)) {
      for (Value dependency : dispatch.getExplicitDependencies()) {
        Operation *producer = dependency.getDefiningOp();
        auto found = nodeIds.find(producer);
        if (found == nodeIds.end())
          // if the producer is not a graph node, it is an error
          return nodes[to].source->emitError("completion dependency is not a graph node");
        // a dispatch has an explicit dependency on a producer, so add an edge
        addReason(found->second, to,
                  makeReason(builder, "explicit", "completion token", "", "", producer,
                             nodes[to].source));
      }
      for (Value argument : dispatch.getDispatchArguments()) {
        auto found = nodeIds.find(argument.getDefiningOp());
        if (found != nodeIds.end())
          // a dispatch argument has a dependency on its defining operation, so add an edge
          addReason(found->second, to,
                    makeReason(builder, "ssa", "dispatch argument", "", "", found->first,
                               nodes[to].source));
      }
    }
  }

  for (unsigned from = 0; from < nodes.size(); ++from) {
    for (unsigned to = from + 1; to < nodes.size(); ++to) {
      // if either node has an unknown effect, add a barrier reason
      if (nodes[from].unknown || nodes[to].unknown)
        addReason(from, to,
                  makeReason(builder, "barrier", "unknown effect", "", "", nodes[from].source,
                             nodes[to].source));
      for (const BoundAccess &lhs : nodes[from].accesses) {
        for (const BoundAccess &rhs : nodes[to].accesses) {
          WholeBufferAliasResult alias = aliasWholeBuffers(lhs.resource, rhs.resource);
          if (alias == WholeBufferAliasResult::NoAlias ||
              !hasConflictingAccesses(lhs.bits, rhs.bits))
            // if the two accesses do not alias or do not conflict, no edge is needed
            continue;
          std::string lhsName = resourceName(lhs.resource, function, allocationIds);
          std::string rhsName = resourceName(rhs.resource, function, allocationIds);
          std::string resource = lhsName == rhsName ? lhsName : lhsName + "~" + rhsName;
          StringRef aliasName = alias == WholeBufferAliasResult::MustAlias ? "must" : "may";
          bool lifetime = ((lhs.bits | rhs.bits) & (Allocate | Free)) != 0;
          // if the two accesses alias and conflict, add an edge with a reason describing the
          // conflict
          addReason(from, to,
                    makeReason(builder, lifetime ? "lifetime" : "memory",
                               lifetime ? "resource lifetime" : "whole-buffer conflict", resource,
                               aliasName, nodes[from].source, nodes[to].source));
        }
      }
    }
  }

  if (llvm::any_of(nodes, [](const NodeInfo &node) { return node.nestedControl; })) {
    // if any node is nested in control flow, add a reason for every pair of nodes that are
    // nested in control flow to ensure that the graph is sound
    for (unsigned to = 1; to < nodes.size(); ++to)
      addReason(to - 1, to,
                makeReason(builder, "control", "nested control flow", "", "", nodes[to - 1].source,
                           nodes[to].source));
  }

  std::string dotStorage;
  llvm::raw_string_ostream dot(dotStorage);
  dot << "digraph \"" << function.getSymName() << "\" {\n";
  for (auto [id, node] : llvm::enumerate(nodes)) {
    dot << "  n" << id << " [label=\"" << id << ": " << node.kind;
    if (node.kernel)
      dot << " " << node.kernel.getValue();
    dot << "\"];\n";
  }
  for (const auto &[edge, reasons] : edges)
    dot << "  n" << edge.first << " -> n" << edge.second << " [label=\""
        << cast<DictionaryAttr>(reasons.front()).getAs<StringAttr>("kind").getValue() << "\"];\n";
  dot << "}\n";

  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  auto graph = GraphOp::create(moduleBuilder, function.getLoc(),
                               FlatSymbolRefAttr::get(module.getContext(), function.getSymName()),
                               builder.getStringAttr((function.getSymName() + ".graph").str()),
                               builder.getStringAttr(dotStorage));
  graph.getBody().push_back(new Block());
  attachMemoryPlan(graph, nodes, edges, function, allocationIds, builder);
  OpBuilder graphBuilder = OpBuilder::atBlockBegin(&graph.getBody().front());
  for (auto [id, node] : llvm::enumerate(nodes)) {
    GraphNodeOp graphNode = GraphNodeOp::create(
        graphBuilder, node.source->getLoc(), builder.getI64IntegerAttr(id),
        builder.getStringAttr(node.kind), node.kernel, node.materializedAccesses,
        builder.getBoolAttr(node.unknown), builder.getBoolAttr(node.synchronizes),
        node.orderingScopes, builder.getStringAttr(printLocation(node.source->getLoc())));
    if (node.kind == "dispatch") {
      for (StringRef attribute : {"grid", "block", "shared_memory", "device", "implementation",
                                  "capabilities"})
        if (Attribute value = node.source->getAttr(attribute))
          graphNode->setAttr(("ckl.launch_" + attribute).str(), value);
      SmallVector<Attribute> arguments;
      auto dispatch = cast<DispatchOpInterface>(node.source);
      for (auto [index, argument] : llvm::enumerate(dispatch.getDispatchArguments())) {
        SmallVector<NamedAttribute> attributes = {
            builder.getNamedAttr("index", builder.getI64IntegerAttr(index)),
            builder.getNamedAttr("type", TypeAttr::get(argument.getType())),
        };
        if (isa<BaseMemRefType>(argument.getType())) {
          attributes.push_back(builder.getNamedAttr("kind", builder.getStringAttr("resource")));
          attributes.push_back(builder.getNamedAttr(
              "resource", builder.getStringAttr(resourceName(argument, function, allocationIds))));
        } else if (Operation *definition = argument.getDefiningOp()) {
          if (Attribute value = definition->getAttr("value")) {
            attributes.push_back(builder.getNamedAttr("kind", builder.getStringAttr("constant")));
            attributes.push_back(builder.getNamedAttr("value", value));
          } else {
            attributes.push_back(builder.getNamedAttr("kind", builder.getStringAttr("unsupported")));
          }
        } else if (auto blockArgument = dyn_cast<BlockArgument>(argument)) {
          attributes.push_back(builder.getNamedAttr("kind", builder.getStringAttr("scalar")));
          attributes.push_back(builder.getNamedAttr(
              "name", builder.getStringAttr(("arg" + Twine(blockArgument.getArgNumber())).str())));
        }
        arguments.push_back(builder.getDictionaryAttr(attributes));
      }
      graphNode->setAttr("ckl.arguments", builder.getArrayAttr(arguments));
    }
  }
  for (const auto &[edge, reasons] : edges)
    GraphEdgeOp::create(graphBuilder, function.getLoc(), builder.getI64IntegerAttr(edge.first),
                        builder.getI64IntegerAttr(edge.second), builder.getArrayAttr(reasons));
  return success();
}

struct BuildGraphPass : public PassWrapper<BuildGraphPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuildGraphPass)

  BuildGraphPass() = default;
  BuildGraphPass(const BuildGraphPass &pass) : PassWrapper(pass) {}

  Option<bool> strict{*this, "strict", llvm::cl::desc("Reject missing effect semantics"),
                      llvm::cl::init(true)};

  StringRef getArgument() const final { return "ckl-build-graph"; }
  StringRef getDescription() const final {
    return "Build a sound whole-buffer orchestration graph";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<GraphOp> oldGraphs(module.getOps<GraphOp>());
    for (GraphOp graph : oldGraphs)
      graph.erase();
    if (failed(deriveEffectSummaries(module, strict))) {
      signalPassFailure();
      return;
    }
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp function : functions) {
      if (failed(buildGraph(module, function))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

void mlir::ckl::registerCKLGraphPasses() { PassRegistration<BuildGraphPass>(); }

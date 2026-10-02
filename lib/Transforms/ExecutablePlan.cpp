#include "ckl/Transforms/ExecutablePlan.h"

#include "ckl/Analysis/GraphAlgorithms.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "ckl/Dialect/Exec/IR/CKLExecOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::ckl;

namespace {

FailureOr<StringRef> getBackend(ArrayRef<GraphNodeOp> dispatches) {
  StringRef backend;
  for (GraphNodeOp node : dispatches) {
    auto device = node->getAttrOfType<StringAttr>("ckl.launch_device");
    if (!device)
      return failure();
    StringRef current = device.getValue().split(':').first;
    if (current.empty() || (!backend.empty() && current != backend))
      return failure();
    backend = current;
  }
  return backend.empty() ? FailureOr<StringRef>(failure()) : backend;
}

ArrayAttr materializePhysicalArguments(GraphNodeOp node, Builder &builder) {
  auto arguments = node->getAttrOfType<ArrayAttr>("ckl.arguments");
  if (!arguments)
    return {};
  SmallVector<Attribute> physical;
  for (auto [slot, attribute] : llvm::enumerate(arguments)) {
    auto argument = dyn_cast<DictionaryAttr>(attribute);
    if (!argument)
      return {};
    SmallVector<NamedAttribute> fields(argument.begin(), argument.end());
    auto logicalIndex = argument.getAs<IntegerAttr>("index");
    if (!logicalIndex || !argument.getAs<StringAttr>("kind"))
      return {};
    fields.push_back(builder.getNamedAttr("logical_index", logicalIndex));
    fields.push_back(builder.getNamedAttr("slot", builder.getI64IntegerAttr(slot)));
    physical.push_back(builder.getDictionaryAttr(fields));
  }
  return builder.getArrayAttr(physical);
}

LogicalResult lowerGraph(GraphOp graph) {
  /*
  This function lowers a CKL graph to an executable plan. The graph must have been analyzed and have
  a successful static memory plan. The function performs the following steps:
  1. It checks for the presence of a memory plan and resources in the graph's attributes. If they
  are missing or the status is not "planned", it emits an error.
  2. It collects all dispatch nodes from the graph and determines the backend (e.g., "cuda") used by
  the dispatches. If the backend cannot be determined, it emits an error.
  3. It creates a new exec::PlanOp in the module, which will hold the executable plan. It also
  creates exec::HeapOp and exec::ResourceOp for each heap and resource defined in the memory plan.
  4. For each dispatch node, it materializes the physical arguments and creates an exec::KernelOp
  with the appropriate launch parameters and dependencies.
  5. Finally, it erases the original graph operation from the module.
  */
  auto memoryPlan = graph->getAttrOfType<DictionaryAttr>("ckl.memory_plan");
  auto resources = graph->getAttrOfType<ArrayAttr>("ckl.resources");
  auto status = memoryPlan ? memoryPlan.getAs<StringAttr>("status") : StringAttr();
  if (!memoryPlan || !resources || !status || status.getValue() != "planned")
    return graph.emitError("executable lowering requires a successful static memory plan");
  if (failed(verifyStaticExecutionSubset(graph, "executable lowering")))
    return failure();

  SmallVector<GraphNodeOp> dispatches;
  for (GraphNodeOp node : graph.getBody().front().getOps<GraphNodeOp>())
    if (node.getKind() == "dispatch")
      dispatches.push_back(node);
  FailureOr<StringRef> backend = getBackend(dispatches);
  if (failed(backend))
    return graph.emitError("executable lowering requires one known backend");
  DispatchDependencyMap dependencies = getReducedDispatchDependencies(graph);

  OpBuilder builder(graph);
  auto plan = exec::PlanOp::create(builder, graph.getLoc(), graph.getSourceAttr(),
                                   builder.getStringAttr((graph.getName() + ".exec").str()),
                                   builder.getStringAttr(*backend));
  plan.getBody().push_back(new Block());
  builder.setInsertionPointToEnd(&plan.getBody().front());

  auto heaps = memoryPlan.getAs<ArrayAttr>("heaps");
  for (auto [id, attribute] : llvm::enumerate(heaps)) {
    auto heap = cast<DictionaryAttr>(attribute);
    exec::HeapOp::create(builder, graph.getLoc(), builder.getI64IntegerAttr(id),
                         heap.getAs<IntegerAttr>("bytes"), heap.getAs<IntegerAttr>("alignment"),
                         heap.getAs<IntegerAttr>("device"),
                         heap.getAs<StringAttr>("address_space"));
  }

  llvm::StringMap<DictionaryAttr> assignments;
  for (DictionaryAttr assignment :
       memoryPlan.getAs<ArrayAttr>("assignments").getAsRange<DictionaryAttr>())
    assignments[assignment.getAs<StringAttr>("resource").getValue()] = assignment;

  for (DictionaryAttr resource : resources.getAsRange<DictionaryAttr>()) {
    StringAttr name = resource.getAs<StringAttr>("name");
    StringAttr kind = resource.getAs<StringAttr>("kind");
    IntegerAttr bytes = resource.getAs<IntegerAttr>("bytes");
    IntegerAttr heap;
    IntegerAttr offset;
    if (kind.getValue() == "temporary") {
      auto found = assignments.find(name.getValue());
      if (found == assignments.end())
        return graph.emitError("temporary resource is missing a memory-plan assignment");
      heap = found->second.getAs<IntegerAttr>("heap");
      offset = found->second.getAs<IntegerAttr>("offset");
    }
    exec::ResourceOp::create(builder, graph.getLoc(), name, kind,
                             resource.getAs<StringAttr>("device"),
                             resource.getAs<StringAttr>("address_space"),
                             resource.getAs<IntegerAttr>("alignment"), bytes, heap, offset);
  }

  for (GraphNodeOp node : dispatches) {
    auto kernel = node->getAttrOfType<SymbolRefAttr>("kernel");
    auto implementation = node->getAttrOfType<StringAttr>("ckl.launch_implementation");
    auto artifact = node->getAttrOfType<StringAttr>("ckl.launch_artifact");
    auto abi = node->getAttrOfType<StringAttr>("ckl.launch_abi");
    auto device = node->getAttrOfType<StringAttr>("ckl.launch_device");
    auto grid = node->getAttrOfType<DenseI64ArrayAttr>("ckl.launch_grid");
    auto block = node->getAttrOfType<DenseI64ArrayAttr>("ckl.launch_block");
    auto sharedMemory = node->getAttrOfType<IntegerAttr>("ckl.launch_shared_memory");
    ArrayAttr arguments = materializePhysicalArguments(node, builder);
    if (!kernel || !implementation || !artifact || !abi || !device || !grid || !block ||
        !sharedMemory || !arguments)
      return node.emitError("executable lowering requires complete launch and ABI metadata");
    exec::KernelOp::create(builder, node.getLoc(), builder.getI64IntegerAttr(node.getId()), kernel,
                           implementation, artifact, abi, device, grid, block, sharedMemory,
                           arguments,
                           builder.getDenseI64ArrayAttr(dependencies.lookup(node.getId())));
  }

  graph.erase();
  return success();
}

struct LowerGraphToExecPass : public PassWrapper<LowerGraphToExecPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphToExecPass)

  StringRef getArgument() const final { return "ckl-lower-graph-to-exec"; }
  StringRef getDescription() const final {
    return "Lower analyzed CKL graphs to verified executable plans";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<exec::CKLExecDialect>();
  }

  void runOnOperation() override {
    SmallVector<GraphOp> graphs(getOperation().getOps<GraphOp>());
    for (GraphOp graph : graphs)
      if (failed(lowerGraph(graph))) {
        signalPassFailure();
        return;
      }
  }
};

} // namespace

void mlir::ckl::registerCKLExecutablePlanPasses() { PassRegistration<LowerGraphToExecPass>(); }

#include "ckl/Analysis/NvidiaHostEmitter.h"

#include "ckl/Analysis/GraphAlgorithms.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <map>
#include <string>

using namespace mlir;
using namespace mlir::ckl;

namespace {

std::string identifier(StringRef value) {
  std::string result;
  result.reserve(value.size() + 1);
  for (char character : value)
    result.push_back(std::isalnum(static_cast<unsigned char>(character)) ? character : '_');
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

FailureOr<std::string> cppScalarType(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 32)
      return std::string("std::int32_t");
    if (integer.getWidth() == 64)
      return std::string("std::int64_t");
  }
  if (auto floating = dyn_cast<FloatType>(type)) {
    if (floating.getWidth() == 32)
      return std::string("float");
    if (floating.getWidth() == 64)
      return std::string("double");
  }
  return failure();
}

LogicalResult emitConstant(DictionaryAttr argument, llvm::raw_ostream &output) {
  Attribute value = argument.get("value");
  Type type = argument.getAs<TypeAttr>("type").getValue();
  FailureOr<std::string> cppType = cppScalarType(type);
  if (failed(cppType))
    return failure();
  if (auto integer = dyn_cast<IntegerAttr>(value)) {
    output << "static_cast<" << *cppType << ">(" << integer.getInt() << ")";
    return success();
  }
  if (auto floating = dyn_cast<FloatAttr>(value)) {
    output << "static_cast<" << *cppType << ">(";
    floating.getValue().print(output);
    output << ")";
    return success();
  }
  return failure();
}

struct ScalarBinding {
  std::string name;
  Type type;
};

LogicalResult emitGraph(GraphOp graph, llvm::raw_ostream &output) {
  auto memoryPlan = graph->getAttrOfType<DictionaryAttr>("ckl.memory_plan");
  if (!memoryPlan || memoryPlan.getAs<StringAttr>("status").getValue() != "planned")
    return graph.emitError("NVIDIA host emission requires a successful static memory plan");
  auto resources = graph->getAttrOfType<ArrayAttr>("ckl.resources");
  if (!resources)
    return graph.emitError("NVIDIA host emission requires materialized resources");
  if (failed(verifyStaticExecutionSubset(graph, "NVIDIA host emission")))
    return failure();

  SmallVector<GraphNodeOp> dispatches;
  for (GraphNodeOp node : graph.getBody().front().getOps<GraphNodeOp>()) {
    if (node.getKind() == "dispatch")
      dispatches.push_back(node);
  }
  DispatchDependencyMap dependencies = getReducedDispatchDependencies(graph);

  std::map<std::string, Type> scalarBindings;
  for (GraphNodeOp node : dispatches) {
    auto capabilities = node->getAttrOfType<ArrayAttr>("ckl.launch_capabilities");
    bool directPointerABI = capabilities && llvm::any_of(
                                                capabilities.getAsRange<StringAttr>(),
                                                [](StringAttr capability) {
                                                  return capability.getValue() == "cuda.direct";
                                                });
    if (!directPointerABI)
      return node.emitError(
          "NVIDIA host emission requires the cuda.direct dispatch capability");
    auto arguments = node->getAttrOfType<ArrayAttr>("ckl.arguments");
    if (!arguments)
      return node.emitError("NVIDIA host emission requires dispatch argument descriptors");
    for (DictionaryAttr argument : arguments.getAsRange<DictionaryAttr>()) {
      StringRef kind = argument.getAs<StringAttr>("kind").getValue();
      if (kind == "unsupported")
        return node.emitError("NVIDIA host emission does not support a computed scalar argument");
      if (kind == "scalar") {
        std::string name = argument.getAs<StringAttr>("name").getValue().str();
        Type type = argument.getAs<TypeAttr>("type").getValue();
        if (failed(cppScalarType(type)))
          return node.emitError("NVIDIA host emission supports only i32, i64, f32, and f64 scalars");
        scalarBindings.emplace(std::move(name), type);
      } else if (kind == "constant" &&
                 failed(cppScalarType(argument.getAs<TypeAttr>("type").getValue()))) {
        return node.emitError("NVIDIA host emission encountered an unsupported constant type");
      }
    }
  }

  std::string graphName = identifier(graph.getName());
  output << "struct " << graphName << "Plan {\n"
         << "  std::vector<mlir::ckl::runtime::NvidiaBuffer> heaps;\n"
         << "  std::vector<mlir::ckl::runtime::NvidiaBuffer> retained;\n"
         << "  mlir::ckl::runtime::ExecutionPlan plan;\n"
         << "};\n\n";
  output << graphName << "Plan build_" << graphName
         << "(mlir::ckl::runtime::NvidiaRuntime &runtime";

  SmallVector<DictionaryAttr> externalResources;
  for (DictionaryAttr resource : resources.getAsRange<DictionaryAttr>())
    if (resource.getAs<StringAttr>("kind").getValue() == "external") {
      externalResources.push_back(resource);
      output << ",\n    const mlir::ckl::runtime::NvidiaBuffer &"
             << identifier(resource.getAs<StringAttr>("name").getValue());
    }
  for (const auto &[name, type] : scalarBindings)
    output << ",\n    " << *cppScalarType(type) << ' ' << identifier(name);
  output << ") {\n  " << graphName << "Plan result;\n";

  auto heaps = memoryPlan.getAs<ArrayAttr>("heaps");
  for (DictionaryAttr heap : heaps.getAsRange<DictionaryAttr>())
    output << "  result.heaps.push_back(runtime.allocate("
           << heap.getAs<IntegerAttr>("bytes").getInt() << "));\n";

  for (auto [externalIndex, resource] : llvm::enumerate(externalResources)) {
    StringRef name = resource.getAs<StringAttr>("name").getValue();
    output << "  result.retained.push_back(" << identifier(name) << ");\n"
           << "  std::uint64_t resource_" << identifier(name) << " = result.retained["
           << externalIndex << "].address();\n";
  }
  auto assignments = memoryPlan.getAs<ArrayAttr>("assignments");
  for (DictionaryAttr assignment : assignments.getAsRange<DictionaryAttr>()) {
    StringRef name = assignment.getAs<StringAttr>("resource").getValue();
    output << "  std::uint64_t resource_" << identifier(name) << " = result.heaps["
           << assignment.getAs<IntegerAttr>("heap").getInt() << "].address() + "
           << assignment.getAs<IntegerAttr>("offset").getInt() << ";\n";
  }

  for (GraphNodeOp node : dispatches) {
    std::size_t id = node.getId();
    auto kernel = node->getAttrOfType<SymbolRefAttr>("kernel");
    auto implementation = node->getAttrOfType<StringAttr>("ckl.launch_implementation");
    auto artifact = node->getAttrOfType<StringAttr>("ckl.launch_artifact");
    auto abi = node->getAttrOfType<StringAttr>("ckl.launch_abi");
    auto device = node->getAttrOfType<StringAttr>("ckl.launch_device");
    auto grid = node->getAttrOfType<DenseI64ArrayAttr>("ckl.launch_grid");
    auto block = node->getAttrOfType<DenseI64ArrayAttr>("ckl.launch_block");
    auto shared = node->getAttrOfType<IntegerAttr>("ckl.launch_shared_memory");
    if (!kernel || !implementation || !artifact || !abi || !device || !grid || !block ||
        !shared)
      return node.emitError("NVIDIA host emission requires complete launch metadata");
    auto [backend, ordinal] = device.getValue().split(':');
    int64_t deviceOrdinal = 0;
    if (backend != "cuda" || ordinal.getAsInteger(10, deviceOrdinal) || deviceOrdinal < 0)
      return node.emitError("NVIDIA host emission requires a cuda:<ordinal> device");
    output << "  mlir::ckl::runtime::KernelArguments arguments_" << id << ";\n";
    for (DictionaryAttr argument :
         node->getAttrOfType<ArrayAttr>("ckl.arguments").getAsRange<DictionaryAttr>()) {
      StringRef kind = argument.getAs<StringAttr>("kind").getValue();
      output << "  arguments_" << id << ".add(";
      if (kind == "resource")
        output << "resource_"
               << identifier(argument.getAs<StringAttr>("resource").getValue());
      else if (kind == "scalar")
        output << identifier(argument.getAs<StringAttr>("name").getValue());
      else if (failed(emitConstant(argument, output)))
        return node.emitError("failed to emit a constant kernel argument");
      output << ");\n";
    }

    auto gridValues = grid.asArrayRef();
    auto blockValues = block.asArrayRef();
    output << "  [[maybe_unused]] auto node_" << id
           << " = result.plan.addKernel({\"" << artifact.getValue() << "\", \""
           << kernel.getLeafReference().getValue() << "\", \"" << abi.getValue() << "\", "
           << deviceOrdinal << ", {" << gridValues[0] << ", " << gridValues[1] << ", "
           << gridValues[2] << "}, {" << blockValues[0] << ", " << blockValues[1] << ", "
           << blockValues[2] << "}, " << shared.getInt() << ", std::move(arguments_" << id
           << ")}, {";
    llvm::interleaveComma(dependencies.lookup(id), output,
                          [&](std::size_t dependency) { output << "node_" << dependency; });
    output << "});\n";
  }
  output << "  return result;\n}\n\n";
  return success();
}

} // namespace

LogicalResult mlir::ckl::emitNvidiaHostBuilders(ModuleOp module, llvm::raw_ostream &output) {
  output << "// Generated by ckl-hostgen. Unresolved direct-pointer execution plan.\n"
         << "#include \"ckl/Runtime/NvidiaRuntime.h\"\n"
         << "#include <cstdint>\n#include <utility>\n#include <vector>\n\n"
         << "namespace ckl_generated {\n\n";
  bool emitted = false;
  for (GraphOp graph : module.getOps<GraphOp>()) {
    if (failed(emitGraph(graph, output)))
      return failure();
    emitted = true;
  }
  if (!emitted)
    return module.emitError("NVIDIA host emission requires at least one ckl.graph");
  output << "} // namespace ckl_generated\n";
  return success();
}

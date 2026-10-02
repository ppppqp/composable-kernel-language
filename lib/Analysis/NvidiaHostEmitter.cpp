#include "ckl/Analysis/NvidiaHostEmitter.h"

#include "ckl/Dialect/Exec/IR/CKLExecOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
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

void emitCString(StringRef value, llvm::raw_ostream &output) {
  output << '"';
  output.write_escaped(value);
  output << '"';
}

LogicalResult emitPlan(exec::PlanOp plan, llvm::raw_ostream &output) {
  if (plan.getBackend() != "cuda")
    return plan.emitError("NVIDIA host emission requires a cuda executable plan");

  SmallVector<exec::HeapOp> heaps(plan.getBody().front().getOps<exec::HeapOp>());
  llvm::sort(heaps, [](exec::HeapOp lhs, exec::HeapOp rhs) { return lhs.getId() < rhs.getId(); });
  DenseMap<int64_t, std::size_t> heapIndices;
  for (auto [index, heap] : llvm::enumerate(heaps))
    heapIndices[heap.getId()] = index;

  SmallVector<exec::ResourceOp> resources(plan.getBody().front().getOps<exec::ResourceOp>());
  SmallVector<exec::KernelOp> kernels(plan.getBody().front().getOps<exec::KernelOp>());
  llvm::sort(kernels,
             [](exec::KernelOp lhs, exec::KernelOp rhs) { return lhs.getId() < rhs.getId(); });
  std::map<std::string, Type> scalarBindings;
  for (exec::KernelOp kernel : kernels) {
    if (kernel.getAbi() != "cuda.direct")
      return kernel.emitError("NVIDIA host emission requires the cuda.direct kernel ABI");
    for (DictionaryAttr argument : kernel.getArguments().getAsRange<DictionaryAttr>()) {
      StringRef kind = argument.getAs<StringAttr>("kind").getValue();
      if (kind == "scalar") {
        std::string name = argument.getAs<StringAttr>("name").getValue().str();
        Type type = argument.getAs<TypeAttr>("type").getValue();
        if (failed(cppScalarType(type)))
          return kernel.emitError(
              "NVIDIA host emission supports only i32, i64, f32, and f64 scalars");
        auto [found, inserted] = scalarBindings.emplace(name, type);
        if (!inserted && found->second != type)
          return kernel.emitError("scalar binding is used with inconsistent physical types");
      } else if (kind == "constant" &&
                 failed(cppScalarType(argument.getAs<TypeAttr>("type").getValue()))) {
        return kernel.emitError("NVIDIA host emission encountered an unsupported constant type");
      }
    }
  }

  std::string planName = identifier(plan.getName());
  output << "struct " << planName << "Plan {\n"
         << "  std::vector<mlir::ckl::runtime::NvidiaBuffer> heaps;\n"
         << "  std::vector<mlir::ckl::runtime::NvidiaBuffer> retained;\n"
         << "  mlir::ckl::runtime::ExecutionPlan plan;\n"
         << "};\n\n";
  output << planName << "Plan build_" << planName
         << "(mlir::ckl::runtime::NvidiaRuntime &runtime";

  SmallVector<exec::ResourceOp> externalResources;
  for (exec::ResourceOp resource : resources)
    if (resource.getKind() == "external") {
      externalResources.push_back(resource);
      output << ",\n    const mlir::ckl::runtime::NvidiaBuffer &"
             << identifier(resource.getName());
    }
  for (const auto &[name, type] : scalarBindings)
    output << ",\n    " << *cppScalarType(type) << ' ' << identifier(name);
  output << ") {\n  " << planName << "Plan result;\n";

  for (exec::HeapOp heap : heaps)
    output << "  result.heaps.push_back(runtime.allocate(" << heap.getBytes() << "));\n";

  for (auto [externalIndex, resource] : llvm::enumerate(externalResources)) {
    StringRef name = resource.getName();
    output << "  result.retained.push_back(" << identifier(name) << ");\n"
           << "  std::uint64_t resource_" << identifier(name) << " = result.retained["
           << externalIndex << "].address();\n";
  }
  for (exec::ResourceOp resource : resources) {
    if (resource.getKind() != "temporary")
      continue;
    auto found = heapIndices.find(resource.getHeapAttr().getInt());
    if (found == heapIndices.end())
      return resource.emitError("references an unknown emitted heap");
    output << "  std::uint64_t resource_" << identifier(resource.getName())
           << " = result.heaps[" << found->second << "].address() + "
           << resource.getOffsetAttr().getInt() << ";\n";
  }

  for (exec::KernelOp kernel : kernels) {
    std::size_t id = kernel.getId();
    auto [backend, ordinal] = kernel.getDevice().split(':');
    int64_t deviceOrdinal = 0;
    if (backend != "cuda" || ordinal.getAsInteger(10, deviceOrdinal) || deviceOrdinal < 0)
      return kernel.emitError("NVIDIA host emission requires a cuda:<ordinal> device");
    output << "  mlir::ckl::runtime::KernelArguments arguments_" << id << ";\n";
    SmallVector<DictionaryAttr> arguments(kernel.getArguments().getAsRange<DictionaryAttr>());
    llvm::sort(arguments, [](DictionaryAttr lhs, DictionaryAttr rhs) {
      return lhs.getAs<IntegerAttr>("slot").getInt() < rhs.getAs<IntegerAttr>("slot").getInt();
    });
    for (DictionaryAttr argument : arguments) {
      StringRef kind = argument.getAs<StringAttr>("kind").getValue();
      output << "  arguments_" << id << ".add(";
      if (kind == "resource")
        output << "resource_"
               << identifier(argument.getAs<StringAttr>("resource").getValue());
      else if (kind == "scalar")
        output << identifier(argument.getAs<StringAttr>("name").getValue());
      else if (failed(emitConstant(argument, output)))
        return kernel.emitError("failed to emit a constant kernel argument");
      output << ");\n";
    }

    auto gridValues = kernel.getGrid();
    auto blockValues = kernel.getBlock();
    output << "  [[maybe_unused]] auto node_" << id << " = result.plan.addKernel({";
    emitCString(kernel.getArtifact(), output);
    output << ", ";
    emitCString(kernel.getKernel().getLeafReference().getValue(), output);
    output << ", ";
    emitCString(kernel.getAbi(), output);
    output << ", "
           << deviceOrdinal << ", {" << gridValues[0] << ", " << gridValues[1] << ", "
           << gridValues[2] << "}, {" << blockValues[0] << ", " << blockValues[1] << ", "
           << blockValues[2] << "}, " << kernel.getSharedMemory() << ", std::move(arguments_" << id
           << ")}, {";
    llvm::interleaveComma(kernel.getDependencies(), output,
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
  for (exec::PlanOp plan : module.getOps<exec::PlanOp>()) {
    if (failed(emitPlan(plan, output)))
      return failure();
    emitted = true;
  }
  if (!emitted)
    return module.emitError("NVIDIA host emission requires at least one ckl_exec.plan");
  output << "} // namespace ckl_generated\n";
  return success();
}

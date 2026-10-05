#include "ckl/Dialect/Exec/IR/CKLExecDialect.h"
#include "ckl/Dialect/Exec/IR/CKLExecOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::ckl;

namespace {
llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                        llvm::cl::desc("<input JSON manifest>"),
                                        llvm::cl::init("-"));
llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output MLIR file"),
                                         llvm::cl::value_desc("filename"), llvm::cl::init("-"));

class ManifestImporter {
public:
  explicit ManifestImporter(MLIRContext &context)
      : context(context), builder(&context), location(builder.getUnknownLoc()) {}

  FailureOr<OwningOpRef<ModuleOp>> import(const llvm::json::Value &document) {
    const auto *root = document.getAsObject();
    if (!root)
      return failOr<OwningOpRef<ModuleOp>>("manifest", "must be a JSON object");
    auto version = root->getInteger("schema_version");
    if (!version || *version != 1)
      return failOr<OwningOpRef<ModuleOp>>("schema_version", "must be the integer 1");
    const auto *planObject = root->getObject("plan");
    if (!planObject)
      return failOr<OwningOpRef<ModuleOp>>("plan", "must be an object");

    auto sourceText = string(*planObject, "source", "plan.source");
    auto name = string(*planObject, "name", "plan.name");
    auto backend = string(*planObject, "backend", "plan.backend");
    if (!sourceText || !name || !backend)
      return failure();
    auto source = dyn_cast_or_null<SymbolRefAttr>(parseAttribute(*sourceText, &context));
    if (!source)
      return failOr<OwningOpRef<ModuleOp>>("plan.source",
                                           "must be an MLIR symbol reference such as @host");

    OwningOpRef<ModuleOp> module = ModuleOp::create(location);
    builder.setInsertionPointToEnd(module->getBody());
    auto plan = exec::PlanOp::create(builder, location, source, *name, *backend);
    Block &body = plan.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&body);

    if (failed(importHeaps(*planObject)) || failed(importResources(*planObject)) ||
        failed(importKernels(*planObject)))
      return failure();
    if (failed(module->verify()))
      return failOr<OwningOpRef<ModuleOp>>("plan",
                                           "does not satisfy ckl_exec verification");
    return module;
  }

private:
  LogicalResult fail(llvm::StringRef path, llvm::StringRef message) {
    llvm::errs() << "ckl-import-manifest: " << path << ' ' << message << '\n';
    return failure();
  }

  template <typename T> FailureOr<T> failOr(llvm::StringRef path, llvm::StringRef message) {
    llvm::errs() << "ckl-import-manifest: " << path << ' ' << message << '\n';
    return failure();
  }

  std::optional<llvm::StringRef> string(const llvm::json::Object &object, llvm::StringRef key,
                                        llvm::StringRef path) {
    auto value = object.getString(key);
    if (!value || value->empty()) {
      llvm::errs() << "ckl-import-manifest: " << path << " must be a non-empty string\n";
      return std::nullopt;
    }
    return *value;
  }

  std::optional<int64_t> integer(const llvm::json::Object &object, llvm::StringRef key,
                                 llvm::StringRef path) {
    auto value = object.getInteger(key);
    if (!value) {
      llvm::errs() << "ckl-import-manifest: " << path << " must be an integer\n";
      return std::nullopt;
    }
    return *value;
  }

  const llvm::json::Array *array(const llvm::json::Object &object, llvm::StringRef key,
                                 llvm::StringRef path) {
    const auto *value = object.getArray(key);
    if (!value)
      llvm::errs() << "ckl-import-manifest: " << path << " must be an array\n";
    return value;
  }

  FailureOr<SmallVector<int64_t>> integerArray(const llvm::json::Object &object,
                                                llvm::StringRef key, llvm::StringRef path,
                                                std::optional<size_t> size = std::nullopt) {
    const auto *values = array(object, key, path);
    if (!values)
      return failure();
    if (size && values->size() != *size)
      return failOr<SmallVector<int64_t>>(path, "has the wrong number of elements");
    SmallVector<int64_t> result;
    for (const auto &[index, value] : llvm::enumerate(*values)) {
      auto number = value.getAsInteger();
      if (!number)
        return failOr<SmallVector<int64_t>>(
            (path + "[" + llvm::Twine(index) + "]").str(), "must be an integer");
      result.push_back(*number);
    }
    return result;
  }

  LogicalResult importHeaps(const llvm::json::Object &plan) {
    const auto *heaps = array(plan, "heaps", "plan.heaps");
    if (!heaps)
      return failure();
    for (const auto &[index, value] : llvm::enumerate(*heaps)) {
      std::string path = (llvm::Twine("plan.heaps[") + llvm::Twine(index) + "]").str();
      const auto *heap = value.getAsObject();
      if (!heap)
        return fail(path, "must be an object");
      auto id = integer(*heap, "id", path + ".id");
      auto bytes = integer(*heap, "bytes", path + ".bytes");
      auto alignment = integer(*heap, "alignment", path + ".alignment");
      auto device = integer(*heap, "device", path + ".device");
      auto addressSpace = string(*heap, "address_space", path + ".address_space");
      if (!id || !bytes || !alignment || !device || !addressSpace)
        return failure();
      exec::HeapOp::create(builder, location, *id, *bytes, *alignment, *device, *addressSpace);
    }
    return success();
  }

  LogicalResult importResources(const llvm::json::Object &plan) {
    const auto *resources = array(plan, "resources", "plan.resources");
    if (!resources)
      return failure();
    for (const auto &[index, value] : llvm::enumerate(*resources)) {
      std::string path = (llvm::Twine("plan.resources[") + llvm::Twine(index) + "]").str();
      const auto *resource = value.getAsObject();
      if (!resource)
        return fail(path, "must be an object");
      auto name = string(*resource, "name", path + ".name");
      auto kind = string(*resource, "kind", path + ".kind");
      auto device = string(*resource, "device", path + ".device");
      auto addressSpace = string(*resource, "address_space", path + ".address_space");
      auto alignment = integer(*resource, "alignment", path + ".alignment");
      if (!name || !kind || !device || !addressSpace || !alignment)
        return failure();
      IntegerAttr bytes;
      IntegerAttr heap;
      IntegerAttr offset;
      if (auto value = resource->getInteger("bytes"))
        bytes = builder.getI64IntegerAttr(*value);
      if (auto value = resource->getInteger("heap"))
        heap = builder.getI64IntegerAttr(*value);
      if (auto value = resource->getInteger("offset"))
        offset = builder.getI64IntegerAttr(*value);
      exec::ResourceOp::create(builder, location, *name, *kind, *device, *addressSpace,
                               *alignment, bytes, heap, offset);
    }
    return success();
  }

  FailureOr<DictionaryAttr> importArgument(const llvm::json::Object &argument,
                                           llvm::StringRef path) {
    auto slot = integer(argument, "slot", (path + ".slot").str());
    auto logicalIndex = integer(argument, "logical_index", (path + ".logical_index").str());
    auto kind = string(argument, "kind", (path + ".kind").str());
    auto typeText = string(argument, "type", (path + ".type").str());
    if (!slot || !logicalIndex || !kind || !typeText)
      return failure();
    Type type = parseType(*typeText, &context);
    if (!type)
      return failOr<DictionaryAttr>((path + ".type").str(), "is not a valid MLIR type");

    NamedAttrList attributes;
    attributes.append("slot", builder.getI64IntegerAttr(*slot));
    attributes.append("logical_index", builder.getI64IntegerAttr(*logicalIndex));
    attributes.append("kind", builder.getStringAttr(*kind));
    attributes.append("type", TypeAttr::get(type));
    if (*kind == "resource") {
      auto resource = string(argument, "resource", (path + ".resource").str());
      auto packing = string(argument, "packing", (path + ".packing").str());
      if (!resource || !packing)
        return failure();
      attributes.append("resource", builder.getStringAttr(*resource));
      attributes.append("packing", builder.getStringAttr(*packing));
    } else if (*kind == "scalar") {
      auto name = string(argument, "name", (path + ".name").str());
      if (!name)
        return failure();
      attributes.append("name", builder.getStringAttr(*name));
    } else if (*kind == "constant") {
      const llvm::json::Value *value = argument.get("value");
      if (!value)
        return failOr<DictionaryAttr>((path + ".value").str(), "is required");
      Attribute constant;
      if (auto integerType = dyn_cast<IntegerType>(type)) {
        auto number = value->getAsInteger();
        if (!number)
          return failOr<DictionaryAttr>((path + ".value").str(), "must be an integer");
        constant = builder.getIntegerAttr(integerType, *number);
      } else if (auto floatType = dyn_cast<FloatType>(type)) {
        auto number = value->getAsNumber();
        if (!number)
          return failOr<DictionaryAttr>((path + ".value").str(), "must be a number");
        constant = builder.getFloatAttr(floatType, *number);
      } else {
        return failOr<DictionaryAttr>((path + ".type").str(),
                                      "constant type must be integer or floating point");
      }
      attributes.append("value", constant);
    }
    return builder.getDictionaryAttr(attributes);
  }

  LogicalResult importKernels(const llvm::json::Object &plan) {
    const auto *kernels = array(plan, "kernels", "plan.kernels");
    if (!kernels)
      return failure();
    for (const auto &[index, value] : llvm::enumerate(*kernels)) {
      std::string path = (llvm::Twine("plan.kernels[") + llvm::Twine(index) + "]").str();
      const auto *kernel = value.getAsObject();
      if (!kernel)
        return fail(path, "must be an object");
      auto id = integer(*kernel, "id", path + ".id");
      auto symbolText = string(*kernel, "kernel", path + ".kernel");
      auto implementation = string(*kernel, "implementation", path + ".implementation");
      auto artifact = string(*kernel, "artifact", path + ".artifact");
      auto abi = string(*kernel, "abi", path + ".abi");
      auto device = string(*kernel, "device", path + ".device");
      auto sharedMemory = integer(*kernel, "shared_memory", path + ".shared_memory");
      auto grid = integerArray(*kernel, "grid", path + ".grid", 3);
      auto block = integerArray(*kernel, "block", path + ".block", 3);
      auto dependencies = integerArray(*kernel, "dependencies", path + ".dependencies");
      const auto *arguments = array(*kernel, "arguments", path + ".arguments");
      if (!id || !symbolText || !implementation || !artifact || !abi || !device ||
          !sharedMemory || failed(grid) || failed(block) || failed(dependencies) || !arguments)
        return failure();
      auto symbol = dyn_cast_or_null<SymbolRefAttr>(parseAttribute(*symbolText, &context));
      if (!symbol)
        return fail(path + ".kernel", "must be an MLIR symbol reference");
      SmallVector<Attribute> descriptors;
      for (const auto &[argumentIndex, argumentValue] : llvm::enumerate(*arguments)) {
        const auto *argument = argumentValue.getAsObject();
        std::string argumentPath =
            (llvm::Twine(path) + ".arguments[" + llvm::Twine(argumentIndex) + "]").str();
        if (!argument)
          return fail(argumentPath, "must be an object");
        FailureOr<DictionaryAttr> descriptor = importArgument(*argument, argumentPath);
        if (failed(descriptor))
          return failure();
        descriptors.push_back(*descriptor);
      }
      exec::KernelOp::create(builder, location, *id, symbol, *implementation, *artifact, *abi,
                             *device, *grid, *block, *sharedMemory,
                             builder.getArrayAttr(descriptors), *dependencies);
    }
    return success();
  }

  MLIRContext &context;
  OpBuilder builder;
  Location location;
};
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "CKL executable-manifest importer\n");

  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << '\n';
    return 1;
  }
  llvm::Expected<llvm::json::Value> document = llvm::json::parse(input->getBuffer());
  if (!document) {
    llvm::errs() << "ckl-import-manifest: " << llvm::toString(document.takeError()) << '\n';
    return 1;
  }

  DialectRegistry registry;
  registry.insert<exec::CKLExecDialect>();
  MLIRContext context(registry);
  context.getOrLoadDialect<exec::CKLExecDialect>();
  FailureOr<OwningOpRef<ModuleOp>> module = ManifestImporter(context).import(*document);
  if (failed(module))
    return 1;

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << '\n';
    return 1;
  }
  (*module)->print(output->os());
  output->os() << '\n';
  output->keep();
  return 0;
}

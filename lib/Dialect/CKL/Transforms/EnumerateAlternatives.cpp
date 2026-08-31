#include "ckl/Dialect/CKL/Transforms/Passes.h"

#include "ckl/Core/Composition/AlternativeProvider.h"
#include "ckl/Core/Layout/Distribution.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::ckl {
namespace {

StringRef stringifyOrigin(::ckl::core::AlternativeOrigin origin) {
  switch (origin) {
  case ::ckl::core::AlternativeOrigin::User:
    return "user";
  case ::ckl::core::AlternativeOrigin::Compiler:
    return "compiler";
  case ::ckl::core::AlternativeOrigin::Extension:
    return "extension";
  case ::ckl::core::AlternativeOrigin::Library:
    return "library";
  case ::ckl::core::AlternativeOrigin::Unspecified:
    return "unspecified";
  }
  llvm_unreachable("unknown alternative origin");
}

StringRef stringifyPlacement(::ckl::core::Placement placement) {
  switch (placement) {
  case ::ckl::core::Placement::Private:
    return "private";
  case ::ckl::core::Placement::Shared:
    return "shared";
  case ::ckl::core::Placement::Global:
    return "global";
  }
  llvm_unreachable("unknown placement");
}

DictionaryAttr exportAlternative(MLIRContext *context,
                                 const ::ckl::core::TaskAlternative &alternative) {
  Builder builder(context);
  auto exportPorts = [&](const std::vector<::ckl::core::PortRealization> &ports) {
    SmallVector<Attribute> result;
    for (const auto &port : ports)
      result.push_back(builder.getDictionaryAttr(
          {builder.getNamedAttr("name", builder.getStringAttr(port.name)),
           builder.getNamedAttr(
               "distribution",
               DistributionAttr::get(context,
                                     ::ckl::core::serialize(port.distribution))),
           builder.getNamedAttr("placement",
                                builder.getStringAttr(stringifyPlacement(port.placement))),
           builder.getNamedAttr("vector_width", builder.getI64IntegerAttr(port.vectorWidth))}));
    return builder.getArrayAttr(result);
  };
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("name", builder.getStringAttr(alternative.name)),
      builder.getNamedAttr("implementation_id",
                           builder.getStringAttr(alternative.implementationId)),
      builder.getNamedAttr("origin", builder.getStringAttr(stringifyOrigin(alternative.origin))),
      builder.getNamedAttr("inputs", exportPorts(alternative.inputs)),
      builder.getNamedAttr("outputs", exportPorts(alternative.outputs)),
      builder.getNamedAttr("registers_per_thread",
                           builder.getI64IntegerAttr(alternative.registersPerThread)),
      builder.getNamedAttr("shared_memory_bytes",
                           builder.getI64IntegerAttr(alternative.sharedMemoryBytes)),
      builder.getNamedAttr("estimated_execution_cost",
                           builder.getI64IntegerAttr(alternative.estimatedExecutionCost))};
  SmallVector<Attribute> capabilities;
  for (const std::string &capability : alternative.requiredCapabilities)
    capabilities.push_back(builder.getStringAttr(capability));
  fields.push_back(builder.getNamedAttr("required_capabilities",
                                        builder.getArrayAttr(capabilities)));
  if (!alternative.implementationSymbol.empty())
    fields.push_back(builder.getNamedAttr(
        "implementation", FlatSymbolRefAttr::get(context, alternative.implementationSymbol)));
  return builder.getDictionaryAttr(fields);
}

class EnumerateAlternativesPass
    : public PassWrapper<EnumerateAlternativesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnumerateAlternativesPass)
  StringRef getArgument() const final { return "ckl-enumerate-alternatives"; }
  StringRef getDescription() const final {
    return "Materialize alternatives contributed by registered target providers";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    std::string target;
    std::string architecture;
    if (auto value = module->getAttrOfType<StringAttr>("ckl.target"))
      target = value.getValue().str();
    if (auto value = module->getAttrOfType<StringAttr>("ckl.architecture"))
      architecture = value.getValue().str();
    std::vector<std::string> capabilities;
    if (auto values = module->getAttrOfType<ArrayAttr>("ckl.available_capabilities"))
      for (Attribute value : values) {
        auto capability = dyn_cast<StringAttr>(value);
        if (!capability) {
          module.emitError("ckl.available_capabilities must contain only strings");
          signalPassFailure();
          return;
        }
        capabilities.push_back(capability.getValue().str());
      }

    WalkResult result = module.walk([&](TaskOp task) -> WalkResult {
      auto requests = task->getAttrOfType<ArrayAttr>("ckl.alternative_providers");
      if (!requests)
        return WalkResult::advance();
      SmallVector<Attribute> alternatives(task.getAlternatives().begin(),
                                          task.getAlternatives().end());
      llvm::StringSet<> names;
      llvm::StringSet<> implementationIds;
      for (Attribute value : alternatives) {
        auto dictionary = cast<DictionaryAttr>(value);
        names.insert(dictionary.getAs<StringAttr>("name").getValue());
        if (auto id = dictionary.getAs<StringAttr>("implementation_id"))
          implementationIds.insert(id.getValue());
      }
      for (Attribute value : requests) {
        auto request = dyn_cast<DictionaryAttr>(value);
        auto providerId = request ? request.getAs<StringAttr>("id") : StringAttr{};
        auto inputs = request ? request.getAs<ArrayAttr>("inputs") : ArrayAttr{};
        auto outputs = request ? request.getAs<ArrayAttr>("outputs") : ArrayAttr{};
        if (!providerId || !inputs || !outputs) {
          task.emitError("each ckl.alternative_providers entry requires id, inputs, and outputs");
          return WalkResult::interrupt();
        }
        const auto *provider =
            ::ckl::core::findTaskAlternativeProvider(providerId.getValue().str());
        if (!provider) {
          task.emitError("unknown task-alternative provider '") << providerId.getValue() << "'";
          return WalkResult::interrupt();
        }
        ::ckl::core::TaskAlternativeRequest coreRequest;
        coreRequest.task = task.getSymName().str();
        coreRequest.target = target;
        coreRequest.architecture = architecture;
        coreRequest.availableCapabilities = capabilities;
        for (auto [field, destination] :
             {std::pair<ArrayAttr, std::vector<std::string> &>(inputs, coreRequest.inputPorts),
              std::pair<ArrayAttr, std::vector<std::string> &>(outputs,
                                                               coreRequest.outputPorts)}) {
          for (Attribute port : field) {
            auto name = dyn_cast<StringAttr>(port);
            if (!name) {
              task.emitError("provider input/output port names must be strings");
              return WalkResult::interrupt();
            }
            destination.push_back(name.getValue().str());
          }
        }
        auto collection = ::ckl::core::collectTaskAlternatives(coreRequest, {provider});
        if (collection.alternatives.empty()) {
          auto diagnostic = task.emitError("provider '")
                            << providerId.getValue() << "' produced no valid alternatives";
          for (const std::string &message : collection.diagnostics)
            diagnostic.attachNote(task.getLoc()) << message;
          return WalkResult::interrupt();
        }
        for (const auto &alternative : collection.alternatives) {
          if (implementationIds.contains(alternative.implementationId))
            continue; // Makes provider enumeration idempotent.
          if (!names.insert(alternative.name).second) {
            task.emitError("provider alternative duplicates name '") << alternative.name << "'";
            return WalkResult::interrupt();
          }
          implementationIds.insert(alternative.implementationId);
          alternatives.push_back(exportAlternative(&getContext(), alternative));
        }
      }
      task->setAttr("alternatives", ArrayAttr::get(&getContext(), alternatives));
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createEnumerateAlternativesPass() {
  return std::make_unique<EnumerateAlternativesPass>();
}

void registerEnumerateAlternativesPass() {
  PassRegistration<EnumerateAlternativesPass>();
}

} // namespace mlir::ckl

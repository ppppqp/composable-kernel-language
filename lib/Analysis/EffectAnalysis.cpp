#include "ckl/Analysis/EffectAnalysis.h"

#include "ckl/Core/MemoryEffects.h"
#include "ckl/Dialect/CKL/IR/CKLInterfaces.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::ckl;

namespace {

struct KernelSummary {
  // argumentEffects[i] is a bitmask of AccessBits for the i-th argument of the kernel.
  // For example, if argumentEffects[0] == Read | Write, then the kernel reads and writes to the
  // first argument.
  SmallVector<unsigned> argumentEffects;

  // unknownRegions[i] is true if the kernel may read or write to an unknown region
  SmallVector<bool> unknownRegions;

  // orderingScopes is a list of scopes that the kernel synchronizes on.
  SmallVector<std::string> orderingScopes;

  // synchronizes is true if the kernel synchronizes on any ordering scope.
  bool synchronizes = false;
  bool unknown = false;
};

bool isResourceType(Type type) { return isa<MemRefType, UnrankedMemRefType>(type); }

class SummaryAnalysis {
public:
  explicit SummaryAnalysis(bool strict) : strict(strict) {}

  FailureOr<KernelSummary> summarize(func::FuncOp function) {
    if (auto found = summaries.find(function); found != summaries.end())
      // already analyzed
      return found->second;

    KernelSummary summary;
    summary.argumentEffects.resize(function.getNumArguments());
    summary.unknownRegions.resize(function.getNumArguments());
    if (function.isDeclaration()) {
      markUnknown(summary, function);
      if (strict) {
        function.emitError("cannot summarize an external function in strict mode");
        return failure();
      }
      summaries.try_emplace(function, summary);
      return summary;
    }

    if (!active.insert(function).second) {
      // analyzing an already active function (likely recursive function)
      markUnknown(summary, function);
      if (strict) {
        function.emitError(
            "recursive effect summaries are conservative and unavailable in strict mode");
        return failure();
      }
      return summary;
    }

    bool failedAnalysis = false;
    for (Block &block : function.getBody())
      for (Operation &operation : block)
        if (failed(analyzeOperation(&operation, function, summary)))
          failedAnalysis = true;
    active.erase(function);
    if (failedAnalysis)
      return failure();
    summaries.try_emplace(function, summary);
    return summary;
  }

private:
  void markUnknown(KernelSummary &summary, func::FuncOp function) const {
    summary.unknown = true;
    summary.synchronizes = true;
    if (!llvm::is_contained(summary.orderingScopes, "unknown"))
      summary.orderingScopes.emplace_back("unknown");
    for (unsigned index = 0; index < summary.argumentEffects.size(); ++index) {
      if (!isResourceType(function.getArgument(index).getType()))
        continue;
      summary.argumentEffects[index] |= Read | Write;
      summary.unknownRegions[index] = true;
    }
  }

  LogicalResult addEffect(Value value, unsigned bits, func::FuncOp function, KernelSummary &summary,
                          Operation *source, bool unknownRegion = false) {
    Value base = getBaseResource(value);
    if (auto argument = dyn_cast<BlockArgument>(base)) {
      if (argument.getOwner() == &function.getBody().front()) {
        summary.argumentEffects[argument.getArgNumber()] |= bits;
        summary.unknownRegions[argument.getArgNumber()] =
            summary.unknownRegions[argument.getArgNumber()] || unknownRegion;
        return success();
      }
    }

    if (isAllocationValue(base))
      return success();

    return handleUnknown(source, function, summary,
                         "cannot resolve an effect to a function resource");
  }

  LogicalResult applyCalleeSummary(Operation *operation, ValueRange actuals, func::FuncOp callee,
                                   func::FuncOp function, KernelSummary &summary) {
    // run callee summary
    FailureOr<KernelSummary> calleeSummary = summarize(callee);
    if (failed(calleeSummary))
      return failure();
    if (actuals.size() != calleeSummary->argumentEffects.size())
      return handleUnknown(operation, function, summary,
                           "callee argument count does not match its effect summary");

    // propagate synchronization info to caller
    summary.synchronizes |= calleeSummary->synchronizes;
    summary.unknown |= calleeSummary->unknown;
    for (const std::string &orderingScope : calleeSummary->orderingScopes)
      if (!llvm::is_contained(summary.orderingScopes, orderingScope))
        summary.orderingScopes.push_back(orderingScope);
    for (auto [index, actual] : llvm::enumerate(actuals)) {
      unsigned effects = calleeSummary->argumentEffects[index];
      if (effects && failed(addEffect(actual, effects, function, summary, operation,
                                      calleeSummary->unknownRegions[index])))
        return failure();
    }
    return success();
  }

  LogicalResult handleUnknown(Operation *operation, func::FuncOp function, KernelSummary &summary,
                              StringRef reason) {
    if (strict)
      return operation->emitError() << reason << " in strict effect mode";
    markUnknown(summary, function);
    return success();
  }

  LogicalResult analyzeOperation(Operation *operation, func::FuncOp function,
                                 KernelSummary &summary) {
    if (isa<func::ReturnOp>(operation))
      return success();

    if (auto call = dyn_cast<func::CallOp>(operation)) {
      // calling another function
      // recursively apply the callee's summary
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
      if (!callee)
        return handleUnknown(operation, function, summary, "cannot resolve callee");
      return applyCalleeSummary(operation, call.getOperands(), callee, function, summary);
    }

    if (auto dispatch = dyn_cast<DispatchOpInterface>(operation)) {
      // dispatching another kernel
      auto callee =
          SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(operation, dispatch.getKernelSymbol());
      if (!callee)
        return handleUnknown(operation, function, summary, "cannot resolve dispatched kernel");
      summary.synchronizes |= !dispatch.getExplicitDependencies().empty();
      return applyCalleeSummary(operation, dispatch.getDispatchArguments(), callee, function,
                                summary);
    }

    if (auto atomic = dyn_cast<AtomicRMWOp>(operation)) {
      // atomic read/write
      std::string orderingScope = (atomic.getOrdering() + "@" + atomic.getScope()).str();
      if (!llvm::is_contained(summary.orderingScopes, orderingScope))
        summary.orderingScopes.push_back(std::move(orderingScope));
    }
    if (auto sync = dyn_cast<SyncOp>(operation)) {
      // synchronization
      summary.synchronizes = true;
      std::string orderingScope = ("sync@" + sync.getScope()).str();
      if (!llvm::is_contained(summary.orderingScopes, orderingScope))
        summary.orderingScopes.push_back(std::move(orderingScope));
    }

    bool modeled = false;
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(operation)) {
      // memory effects
      modeled = true;
      SmallVector<MemoryEffects::EffectInstance> effects;
      interface.getEffects(effects);
      for (const MemoryEffects::EffectInstance &effect : effects) {
        unsigned bits = classifyMemoryEffect(effect.getEffect());
        if (!bits)
          continue;
        Value value = effect.getValue();
        if (!value) {
          if (isa<SyncOp>(operation))
            continue;
          if (failed(handleUnknown(operation, function, summary,
                                   "memory effect has no addressable resource")))
            return failure();
          continue;
        }
        bool unknownRegion = false;
        if (auto regionInterface = dyn_cast<AccessRegionOpInterface>(operation)) {
          StringAttr region = regionInterface.getAccessRegion(value);
          if (!region || (region.getValue() != "whole" && region.getValue() != "unknown")) {
            if (failed(handleUnknown(operation, function, summary,
                                     "access-region interface returned an unsupported region")))
              return failure();
            continue;
          }
          unknownRegion = region.getValue() == "unknown";
        }
        // add the effect to current kernel
        if (failed(addEffect(value, bits, function, summary, operation, unknownRegion)))
          return failure();
      }
    }

    if (!modeled && !isMemoryEffectFree(operation) &&
        !operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      if (failed(handleUnknown(operation, function, summary,
                               "operation has no complete memory-effect model")))
        return failure();

    for (Region &region : operation->getRegions())
      // recursively analyze it's region
      for (Block &block : region)
        for (Operation &nested : block)
          if (failed(analyzeOperation(&nested, function, summary)))
            return failure();
    return success();
  }

  bool strict;
  DenseMap<Operation *, KernelSummary> summaries;
  SmallPtrSet<Operation *, 8> active;
};

DictionaryAttr materializeSummary(MLIRContext *context, func::FuncOp function,
                                  const KernelSummary &summary) {
  Builder builder(context);
  SmallVector<Attribute> arguments;
  for (auto [index, bits] : llvm::enumerate(summary.argumentEffects)) {
    if (!bits)
      continue;
    SmallVector<Attribute> effects;
    auto add = [&](unsigned mask) {
      if (bits & mask)
        effects.push_back(builder.getStringAttr(stringifyAccessEffect(mask)));
    };
    add(Read);
    add(Write);
    add(Allocate);
    add(Free);
    arguments.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("arg", builder.getI64IntegerAttr(index)),
        builder.getNamedAttr("effects", builder.getArrayAttr(effects)),
        builder.getNamedAttr(
            "region", builder.getStringAttr(summary.unknownRegions[index] ? "unknown" : "whole")),
    }));
  }

  SmallVector<Attribute> mayAlias;
  for (unsigned lhs = 0; lhs < function.getNumArguments(); ++lhs) {
    if (!isResourceType(function.getArgument(lhs).getType()))
      continue;
    for (unsigned rhs = lhs + 1; rhs < function.getNumArguments(); ++rhs) {
      if (!isResourceType(function.getArgument(rhs).getType()))
        continue;
      if (aliasWholeBuffers(function.getArgument(lhs), function.getArgument(rhs)) ==
          WholeBufferAliasResult::MayAlias)
        mayAlias.push_back(
            builder.getArrayAttr({builder.getI64IntegerAttr(lhs), builder.getI64IntegerAttr(rhs)}));
    }
  }

  SmallVector<Attribute> orderingScopes;
  for (const std::string &orderingScope : summary.orderingScopes)
    orderingScopes.push_back(builder.getStringAttr(orderingScope));

  return builder.getDictionaryAttr({
      builder.getNamedAttr("accesses", builder.getArrayAttr(arguments)),
      builder.getNamedAttr("may_alias", builder.getArrayAttr(mayAlias)),
      builder.getNamedAttr("ordering_scopes", builder.getArrayAttr(orderingScopes)),
      builder.getNamedAttr("synchronizes", builder.getBoolAttr(summary.synchronizes)),
      builder.getNamedAttr("unknown", builder.getBoolAttr(summary.unknown)),
  });
}

struct SummarizeEffectsPass : public PassWrapper<SummarizeEffectsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SummarizeEffectsPass)

  SummarizeEffectsPass() = default;
  SummarizeEffectsPass(const SummarizeEffectsPass &pass) : PassWrapper(pass) {}

  Option<bool> strict{*this, "strict",
                      llvm::cl::desc("Reject operations without complete effect semantics"),
                      llvm::cl::init(true)};

  StringRef getArgument() const final { return "ckl-summarize-effects"; }
  StringRef getDescription() const final {
    return "Infer whole-buffer interprocedural CKL effect summaries";
  }

  void runOnOperation() override {
    if (failed(deriveEffectSummaries(getOperation(), strict)))
      signalPassFailure();
  }
};

} // namespace

LogicalResult mlir::ckl::deriveEffectSummaries(ModuleOp module, bool strict) {
  SummaryAnalysis analysis(strict);
  bool failedAnalysis = false;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    FailureOr<KernelSummary> summary = analysis.summarize(function);
    if (failed(summary)) {
      failedAnalysis = true;
      continue;
    }
    function->setAttr("ckl.effect_summary",
                      materializeSummary(module.getContext(), function, *summary));
  }
  return failure(failedAnalysis);
}

void mlir::ckl::registerCKLPasses() {
  PassRegistration<SummarizeEffectsPass>();
  registerCKLGraphPasses();
}

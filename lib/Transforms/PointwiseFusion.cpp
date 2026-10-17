#include "ckl/Transforms/PointwiseFusion.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <optional>

using namespace mlir;
using namespace mlir::ckl;
namespace {
struct Pointwise {
  func::FuncOp function;
  Value index;
  scf::IfOp guard;
  StoreElementOp store;
  unsigned input = 0, output = 0;
  int64_t count = 0;
  arith::CmpIPredicate predicate;
};

// Recognize a small, structural contract rather than guessing from effect
// summaries: one x index, one full bounds guard, same-index accesses, one
// input buffer, one final output store, and pure arithmetic only.
std::optional<Pointwise> recognize(func::FuncOp function) {
  if (!function || !function->hasAttr("ckl.kernel") || function.isExternal() ||
      !llvm::hasSingleElement(function.getBody()) || function.getNumResults())
    return {};
  Pointwise p;
  p.function = function;
  unsigned buffers = 0;
  for (Type type : function.getArgumentTypes()) {
    if (auto buffer = dyn_cast<MemRefType>(type)) {
      // only allow one-dimensional, identity-layout, static-shape buffers in the default memory
      // space
      if (buffer.getRank() != 1 || !buffer.hasStaticShape() || !buffer.getLayout().isIdentity() ||
          buffer.getMemorySpace())
        return {};
      if (buffers && p.count != buffer.getDimSize(0))
        return {};
      p.count = buffer.getDimSize(0);
      ++buffers;
    }
  }
  if (buffers != 2 || p.count <= 0)
    // must have exactly one input and one output buffer, and the size must be known
    return {};
  for (Operation &op : function.getBody().front()) {
    if (auto index = dyn_cast<GlobalIdOp>(op)) {
      if (p.index || index.getAxis() != 0)
        return {};
      p.index = index.getResult();
    } else if (auto guard = dyn_cast<scf::IfOp>(op)) {
      if (p.guard || guard.getNumResults() || !guard.getElseRegion().empty())
        return {};
      p.guard = guard;
    } else if (!isa<arith::ConstantOp, arith::CmpIOp, func::ReturnOp>(op)) {
      return {};
    }
  }
  if (!p.index || !p.guard)
    return {};
  auto compare = p.guard.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getLhs() != p.index ||
      (compare.getPredicate() != arith::CmpIPredicate::slt &&
       compare.getPredicate() != arith::CmpIPredicate::ult))
    return {};
  auto bound = compare.getRhs().getDefiningOp<arith::ConstantIndexOp>();
  if (!bound || bound.value() != p.count)
    return {};
  p.predicate = compare.getPredicate();
  std::optional<unsigned> input;
  for (Operation &op : p.guard.getThenRegion().front()) {
    if (auto load = dyn_cast<LoadElementOp>(op)) {
      // The load must be from the input buffer, and the index must be the same as the global id.
      auto argument = dyn_cast<BlockArgument>(load.getResource());
      if (!argument || argument.getOwner() != &function.getBody().front() ||
          load.getIndices().size() != 1 || load.getIndices()[0] != p.index)
        return {};
      if (input && *input != argument.getArgNumber())
        return {};
      input = argument.getArgNumber();
    } else if (auto store = dyn_cast<StoreElementOp>(op)) {
      // The store must be to the output buffer, and the index must be the same as the global id.
      auto argument = dyn_cast<BlockArgument>(store.getResource());
      if (p.store || !argument || argument.getOwner() != &function.getBody().front() ||
          store.getIndices().size() != 1 || store.getIndices()[0] != p.index ||
          !isa<scf::YieldOp>(store->getNextNode()))
        return {};
      p.store = store;
      p.output = argument.getArgNumber();
    } else if (isa<scf::YieldOp>(op)) {
      continue;
    } else if (op.getName().getDialectNamespace() != "arith" || op.getNumRegions() ||
               !isMemoryEffectFree(&op)) {
      return {};
    }
    // Do not introduce new contraction/reassociation opportunities across a
    // materialized intermediate under relaxed floating-point contracts.
    if (auto flags = op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath"))
      if (flags.getValue() != arith::FastMathFlags::none)
        return {};
  }
  if (!input || !p.store || *input == p.output ||
      function.getArgument(*input).getType() != function.getArgument(p.output).getType())
    return {};
  p.input = *input;
  return p;
}

bool sameLaunch(DispatchOp a, DispatchOp b) {
  // The two dispatches must have the same launch configuration, no shared memory, and no
  // dependencies. The result values must be unused.
  return a.getDevice() == b.getDevice() && a.getGrid() == b.getGrid() &&
         a.getBlock() == b.getBlock() && a.getCapabilities() == b.getCapabilities() &&
         a.getSharedMemory() == 0 && b.getSharedMemory() == 0 && a.getGrid()[1] == 1 &&
         a.getGrid()[2] == 1 && a.getBlock()[1] == 1 && a.getBlock()[2] == 1 &&
         a.getDependencies().empty() && b.getDependencies().empty() && a.getResult().use_empty() &&
         b.getResult().use_empty();
}

bool fuse(ModuleOp module, DispatchOp first, DispatchOp second) {
  if (!sameLaunch(first, second))
    return false;
  auto producer =
      recognize(SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(first, first.getKernelAttr()));
  auto consumer =
      recognize(SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(second, second.getKernelAttr()));
  if (!producer || !consumer || producer->predicate != consumer->predicate ||
      producer->count != consumer->count)
    return false;
  Value temporary = first.getArguments()[producer->output];
  auto allocation = temporary.getDefiningOp<AllocOp>();
  if (!allocation || allocation->getBlock() != first->getBlock() ||
      second.getArguments()[consumer->input] != temporary)
    return false;
  FreeOp release;
  unsigned producers = 0, consumers = 0;
  for (OpOperand &use : temporary.getUses()) {
    if (use.getOwner() == first && use.getOperandNumber() == producer->output)
      ++producers;
    else if (use.getOwner() == second && use.getOperandNumber() == consumer->input)
      ++consumers;
    else if (auto free = dyn_cast<FreeOp>(use.getOwner())) {
      if (release || free->getBlock() != first->getBlock() || !second->isBeforeInBlock(free))
        return false;
      release = free;
    } else
      return false; // Escapes, views, or more than one reader/writer.
  }
  if (producers != 1 || consumers != 1 || !release)
    return false;

  // Keep all original input/output/scalar bindings except the eliminated
  // temporary. Deduplicate equal SSA operands so a chain's signature stays small.
  SmallVector<Value> actuals;
  auto add = [&](Value actual) {
    auto found = llvm::find(actuals, actual);
    if (found != actuals.end())
      return unsigned(found - actuals.begin());
    actuals.push_back(actual);
    return unsigned(actuals.size() - 1);
  };
  SmallVector<unsigned> producerArgs, consumerArgs;
  for (auto [i, actual] : llvm::enumerate(first.getArguments()))
    producerArgs.push_back(i == producer->output ? unsigned(-1) : add(actual));
  for (auto [i, actual] : llvm::enumerate(second.getArguments()))
    consumerArgs.push_back(i == consumer->input ? unsigned(-1) : add(actual));
  // Equal input/output actuals would collapse the two-buffer recognizer. Use
  // the original launches for these cases (the native alias guard also applies).
  if (actuals[producerArgs[producer->input]] == actuals[consumerArgs[consumer->output]])
    return false;
  SmallVector<Type> types;
  for (Value actual : actuals)
    types.push_back(actual.getType());
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto function = func::FuncOp::create(builder, first.getLoc(), "__ckl_fused",
                                       builder.getFunctionType(types, {}));
  function->setAttr("ckl.kernel", builder.getUnitAttr());
  auto steps = [](func::FuncOp function) {
    auto count = function->getAttrOfType<IntegerAttr>("ckl.fused_steps");
    return count ? count.getInt() : int64_t(1);
  };
  function->setAttr("ckl.fused_steps", builder.getI64IntegerAttr(steps(producer->function) +
                                                                 steps(consumer->function)));
  Block *entry = function.addEntryBlock();
  IRMapping pm, cm;
  for (auto [i, index] : llvm::enumerate(producerArgs))
    if (index != unsigned(-1))
      pm.map(producer->function.getArgument(i), entry->getArgument(index));
  for (auto [i, index] : llvm::enumerate(consumerArgs))
    if (index != unsigned(-1))
      cm.map(consumer->function.getArgument(i), entry->getArgument(index));
  builder.setInsertionPointToStart(entry);
  for (Operation &op : producer->function.getBody().front()) {
    if (&op == producer->guard.getOperation() || isa<func::ReturnOp>(op))
      continue;
    builder.clone(op, pm);
  }
  cm.map(consumer->index, pm.lookup(producer->index));
  for (Operation &op : consumer->function.getBody().front()) {
    if (&op == consumer->guard.getOperation() || isa<func::ReturnOp, GlobalIdOp>(op))
      continue;
    builder.clone(op, cm);
  }
  auto guard =
      scf::IfOp::create(builder, first.getLoc(), pm.lookup(producer->guard.getCondition()), false);
  builder.setInsertionPointToStart(&guard.getThenRegion().front());
  for (Operation &op : producer->guard.getThenRegion().front()) {
    if (isa<StoreElementOp, scf::YieldOp>(op))
      continue;
    builder.clone(op, pm);
  }
  Value intermediate = pm.lookup(producer->store.getValue());
  for (Operation &op : consumer->guard.getThenRegion().front()) {
    if (auto load = dyn_cast<LoadElementOp>(op))
      cm.map(load.getResult(), intermediate);
    else if (!isa<scf::YieldOp>(op))
      builder.clone(op, cm);
  }
  builder.setInsertionPointToEnd(entry);
  func::ReturnOp::create(builder, second.getLoc());
  std::string text;
  llvm::raw_string_ostream out(text);
  function->print(out, OpPrintingFlags().useLocalScope());
  std::string symbol =
      "kernel_fused_" +
      llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true).substr(0, 24);
  function.setSymName(symbol);
  // The function is already in the module; choose a unique name if an
  // identical chain appears more than once.
  for (unsigned suffix = 0;; ++suffix) {
    bool duplicate = false;
    for (auto other : module.getOps<func::FuncOp>())
      duplicate |= other != function && other.getSymName() == function.getSymName();
    if (!duplicate)
      break;
    function.setSymName(symbol + "_" + std::to_string(suffix));
  }
  builder.setInsertionPoint(second);
  auto fused = cast<DispatchOp>(builder.clone(*second.getOperation()));
  fused->setOperands(actuals);
  fused->setAttr("operandSegmentSizes", builder.getDenseI32ArrayAttr({int32_t(actuals.size()), 0}));
  fused.setKernelAttr(FlatSymbolRefAttr::get(module.getContext(), function.getSymName()));
  fused.setImplementationAttr(builder.getStringAttr(function.getSymName()));
  fused->removeAttr("artifact");
  first.erase();
  second.erase();
  release.erase();
  allocation.erase();
  return true;
}

struct PointwiseFusionPass : PassWrapper<PointwiseFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PointwiseFusionPass)
  StringRef getArgument() const final { return "ckl-fuse-pointwise"; }
  StringRef getDescription() const final {
    return "Fuse guarded unary pointwise chains, requiring a native disjoint-buffer fallback";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr("ckl.cuda.artifact")) {
      module.emitError("pointwise fusion must run before CUDA preparation");
      signalPassFailure();
      return;
    }
    bool changed = false, again;
    do {
      again = false;
      for (auto host : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
        if (host->hasAttr("ckl.kernel") || host.isExternal() ||
            !llvm::hasSingleElement(host.getBody()))
          continue;
        DispatchOp previous;
        for (Operation &op : host.getBody().front()) {
          if (auto dispatch = dyn_cast<DispatchOp>(op)) {
            if (previous && fuse(module, previous, dispatch)) {
              again = changed = true;
              break;
            }
            previous = dispatch;
          } else if (!isa<AllocOp, arith::ConstantOp>(op))
            previous = {};
        }
        if (again)
          break;
      }
    } while (again);
    if (changed) {
      module->setAttr("ckl.fusion.requires_disjoint", UnitAttr::get(&getContext()));
      for (auto function : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
        function->removeAttr("ckl.effect_summary");
        if (function->hasAttr("ckl.kernel") && SymbolTable::symbolKnownUseEmpty(function, module))
          function.erase();
      }
    }
  }
};
} // namespace

void mlir::ckl::registerCKLPointwiseFusionPass() { PassRegistration<PointwiseFusionPass>(); }

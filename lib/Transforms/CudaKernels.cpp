#include "ckl/Transforms/CudaKernels.h"
#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::ckl;

namespace {
bool numeric(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.isSignless() && (integer.getWidth() == 32 || integer.getWidth() == 64);
  return type.isF32() || type.isF64();
}

LogicalResult validateKernels(ModuleOp module) {
  unsigned count = 0;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (!function->hasAttr("ckl.kernel"))
      continue;
    ++count;
    if (function.isExternal() || function.getNumResults() ||
        !llvm::hasSingleElement(function.getBody()))
      return function.emitError("CUDA kernels require one defined entry block and no results");
    for (Type type : function.getArgumentTypes()) {
      if (auto buffer = dyn_cast<MemRefType>(type)) {
        if (!buffer.hasStaticShape() || !buffer.getLayout().isIdentity() ||
            buffer.getMemorySpace())
          return function.emitError(
              "CUDA direct ABI requires static identity-layout default-space memrefs");
        type = buffer.getElementType();
      }
      if (!numeric(type))
        return function.emitError("CUDA direct ABI supports only f32, f64, i32, and i64 arguments/elements");
    }
    WalkResult result = function.walk([&](Operation *operation) {
      if (operation == function.getOperation())
        return WalkResult::advance();
      bool supported = isa<LoadElementOp, StoreElementOp, ThreadIdOp, BlockIdOp,
                           BlockDimOp, GridDimOp, GlobalIdOp, func::ReturnOp,
                           scf::IfOp, scf::ForOp, scf::YieldOp>(operation) ||
                       operation->getName().getDialectNamespace() == "arith";
      if (!supported) {
        operation->emitError("unsupported operation in CKL CUDA kernel; legacy "
                             "whole-buffer operations have no device semantics");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
  }
  if (!count)
    return module.emitError("CUDA lowering requires at least one marked CKL kernel");
  return success();
}

std::string artifactIdentity(ModuleOp module, StringRef chip, StringRef features) {
  llvm::SHA256 hash;
  hash.update("ckl.cuda.v1;bare-ptr;index64;opt3;");
  hash.update(chip);
  hash.update(";");
  hash.update(features);
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (!function->hasAttr("ckl.kernel"))
      continue;
    OwningOpRef<func::FuncOp> copy(cast<func::FuncOp>(function->clone()));
    copy->getOperation()->removeAttr("ckl.effect_summary");
    std::string text;
    llvm::raw_string_ostream output(text);
    copy->getOperation()->print(output, OpPrintingFlags().useLocalScope());
    hash.update(text);
  }
  return "ckl.cuda." + llvm::toHex(hash.final(), /*LowerCase=*/true);
}

struct PrepareCudaPass : PassWrapper<PrepareCudaPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareCudaPass)
  PrepareCudaPass() = default;
  PrepareCudaPass(const PrepareCudaPass &other) : PassWrapper(other) {}
  Option<std::string> chip{*this, "chip", llvm::cl::desc("CUDA SM target"),
                           llvm::cl::init("sm_120")};
  Option<std::string> features{*this, "features", llvm::cl::desc("NVPTX target features"),
                               llvm::cl::init("+ptx87")};
  StringRef getArgument() const final { return "ckl-prepare-cuda"; }
  StringRef getDescription() const final {
    return "Prepare verified CKL CUDA artifact and direct-ABI dispatch metadata";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(validateKernels(module))) {
      signalPassFailure();
      return;
    }
    std::string artifact = artifactIdentity(module, chip, features);
    Builder builder(&getContext());
    bool invalid = false;
    module.walk([&](DispatchOp dispatch) {
      auto kernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          dispatch, dispatch.getKernelAttr());
      if (!kernel || !kernel->hasAttr("ckl.kernel")) {
        dispatch.emitError("CUDA dispatch must reference a marked CKL kernel");
        invalid = true;
        return;
      }
      if (dispatch.getDevice() != "unassigned:0" && dispatch.getDevice() != "cuda:0") {
        dispatch.emitError("initial CKL CUDA compilation supports only logical device 0");
        invalid = true;
        return;
      }
      for (int64_t dimension : dispatch.getGrid())
        if (dimension > UINT32_MAX) {
          dispatch.emitError("CUDA grid dimension exceeds the physical launch ABI");
          invalid = true;
        }
      uint64_t threads = 1;
      for (int64_t dimension : dispatch.getBlock()) {
        if (dimension > 1024 || threads > 1024 / static_cast<uint64_t>(dimension)) {
          dispatch.emitError("CUDA block exceeds 1024 threads");
          invalid = true;
          break;
        }
        threads *= dimension;
      }
      if (dispatch.getSharedMemoryAttr().getInt() > UINT32_MAX) {
        dispatch.emitError("CUDA shared-memory bytes exceed the physical launch ABI");
        invalid = true;
      }
      dispatch.setDeviceAttr(builder.getStringAttr("cuda:0"));
      dispatch.setCapabilitiesAttr(builder.getStrArrayAttr({"cuda.direct"}));
      dispatch.setArtifactAttr(builder.getStringAttr(artifact));
      dispatch.setImplementationAttr(
          builder.getStringAttr(artifact + "." + kernel.getSymName().str()));
    });
    if (invalid) {
      signalPassFailure();
      return;
    }
    module->setAttr("ckl.cuda.artifact", builder.getStringAttr(artifact));
    module->setAttr("ckl.cuda.chip", builder.getStringAttr(chip));
    module->setAttr("ckl.cuda.features", builder.getStringAttr(features));
  }
};

struct LowerKernelsToGPUPass : PassWrapper<LowerKernelsToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKernelsToGPUPass)
  StringRef getArgument() const final { return "ckl-lower-kernels-to-gpu"; }
  StringRef getDescription() const final {
    return "Extract prepared CKL bodies and lower their computation to a GPU module";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect, arith::ArithDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto artifact = module->getAttrOfType<StringAttr>("ckl.cuda.artifact");
    auto chip = module->getAttrOfType<StringAttr>("ckl.cuda.chip");
    auto features = module->getAttrOfType<StringAttr>("ckl.cuda.features");
    if (!artifact || !chip || !features || failed(validateKernels(module))) {
      module.emitError("kernel extraction requires a ckl-prepare-cuda module");
      signalPassFailure();
      return;
    }
    if (artifact.getValue() != artifactIdentity(module, chip.getValue(), features.getValue())) {
      module.emitError("prepared artifact identity is stale: kernel bodies or target changed");
      signalPassFailure();
      return;
    }
    for (Operation &operation : module.getBody()->getOperations())
      if (!isa<func::FuncOp>(operation)) {
        operation.emitError("kernel extraction expects prepared source functions, before graph lowering");
        signalPassFailure();
        return;
      }
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToEnd(module.getBody());
    auto gpuModule = gpu::GPUModuleOp::create(builder, module.getLoc(), "ckl_kernels");
    for (func::FuncOp function : functions) {
      if (!function->hasAttr("ckl.kernel"))
        continue;
      builder.setInsertionPointToStart(gpuModule.getBody());
      auto kernel = gpu::GPUFuncOp::create(builder, function.getLoc(),
                                          function.getSymName(), function.getFunctionType());
      kernel.setKernelAttr(builder.getUnitAttr());
      kernel.getBody().getBlocks().clear();
      IRMapping mapping;
      function.getBody().cloneInto(&kernel.getBody(), mapping);
      SmallVector<Operation *> operations;
      kernel.walk([&](Operation *operation) { operations.push_back(operation); });
      for (Operation *operation : operations) {
        builder.setInsertionPoint(operation);
        Value replacement;
        Location loc = operation->getLoc();
        if (auto load = dyn_cast<LoadElementOp>(operation))
          replacement = memref::LoadOp::create(builder, loc, load.getResource(), load.getIndices());
        else if (auto store = dyn_cast<StoreElementOp>(operation)) {
          memref::StoreOp::create(builder, loc, store.getValue(), store.getResource(), store.getIndices());
          store.erase();
          continue;
        } else if (isa<ThreadIdOp, BlockIdOp, BlockDimOp, GridDimOp, GlobalIdOp>(operation)) {
          int64_t axis = operation->getAttrOfType<IntegerAttr>("axis").getInt();
          auto dimension = static_cast<gpu::Dimension>(axis);
          if (isa<ThreadIdOp>(operation))
            replacement = gpu::ThreadIdOp::create(builder, loc, dimension);
          else if (isa<BlockIdOp>(operation))
            replacement = gpu::BlockIdOp::create(builder, loc, dimension);
          else if (isa<BlockDimOp>(operation))
            replacement = gpu::BlockDimOp::create(builder, loc, dimension);
          else if (isa<GridDimOp>(operation))
            replacement = gpu::GridDimOp::create(builder, loc, dimension);
          else {
            Value block = gpu::BlockIdOp::create(builder, loc, dimension);
            Value width = gpu::BlockDimOp::create(builder, loc, dimension);
            Value thread = gpu::ThreadIdOp::create(builder, loc, dimension);
            Value base = arith::MulIOp::create(builder, loc, block, width);
            replacement = arith::AddIOp::create(builder, loc, base, thread);
          }
        } else if (isa<func::ReturnOp>(operation)) {
          gpu::ReturnOp::create(builder, loc);
          operation->erase();
          continue;
        }
        if (replacement) {
          operation->getResult(0).replaceAllUsesWith(replacement);
          operation->erase();
        }
      }
    }
    for (func::FuncOp function : functions)
      function.erase();
  }
};
} // namespace

void mlir::ckl::registerCKLCudaKernelPasses() {
  PassRegistration<PrepareCudaPass>();
  PassRegistration<LowerKernelsToGPUPass>();
}

#include "ckl/Transforms/ExecutablePlan.h"

#include "ckl/Dialect/Exec/IR/CKLExecOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include <string>

using namespace mlir;
using namespace mlir::ckl;

namespace {

// The entry constructs an invocation through the runtime ABI. It does not launch
// kernels: ordinary execution and graph replay operate on the resulting plan.
class HostLowering {
public:
  HostLowering(ModuleOp output, func::FuncOp source)
      : output(output), source(source), b(output.getContext()), loc(source.getLoc()),
        ptr(LLVM::LLVMPointerType::get(output.getContext())) {}

  LogicalResult lower(exec::PlanOp plan, StringRef binary) {
    if (plan.getBackend() != "cuda")
      return plan.emitError("LLVM host lowering requires the CUDA backend");
    b.setInsertionPointToEnd(output.getBody());
    auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(b.getContext()), {ptr, ptr});
    auto entry = LLVM::LLVMFuncOp::create(b, loc, "ckl_entry", type);
    auto *block = entry.addEntryBlock(b);
    b.setInsertionPointToStart(block);
    frame = block->getArgument(0);
    arguments = block->getArgument(1);
    call("ckl_begin", {frame, bytes(binary, "ckl.binary"), number(binary.size())});

    for (auto heap : plan.getBody().front().getOps<exec::HeapOp>()) {
      if (heap.getDevice() != 0 || heap.getAddressSpace() != "global")
        return heap.emitError("LLVM host lowering requires device-0 global heaps");
      call("ckl_heap", {frame, number(heap.getId()), number(heap.getBytes())});
    }
    llvm::StringMap<uint64_t> resources;
    for (auto resource : plan.getBody().front().getOps<exec::ResourceOp>()) {
      if (resource.getDevice() != "cuda:0" || resource.getAddressSpace() != "global")
        return resource.emitError("LLVM host lowering requires cuda:0 global resources");
      uint64_t id = resources.size();
      resources[resource.getName()] = id;
      bool external = resource.getKind() == "external";
      Value address = LLVM::ZeroOp::create(b, loc, ptr);
      if (external) {
        auto argument = binding(resource.getName(), true);
        if (failed(argument))
          return failure();
        address = *argument;
      }
      call("ckl_resource", {frame, number(id),
           number(external ? 0 : resource.getHeapAttr().getInt()),
           number(external ? 0 : resource.getOffsetAttr().getInt()), address});
    }
    for (auto kernel : plan.getBody().front().getOps<exec::KernelOp>()) {
      if (kernel.getAbi() != "cuda.direct" || kernel.getDevice() != "cuda:0")
        return kernel.emitError("LLVM host lowering requires cuda.direct on logical device 0");
      auto grid = kernel.getGrid(), block = kernel.getBlock();
      call("ckl_kernel", {frame, bytes(kernel.getKernel().getLeafReference().getValue()),
           number(grid[0]), number(grid[1]), number(grid[2]), number(block[0]),
           number(block[1]), number(block[2]), number(kernel.getSharedMemory())});
      SmallVector<DictionaryAttr> slots(kernel.getArguments().getAsRange<DictionaryAttr>());
      llvm::sort(slots, [](DictionaryAttr a, DictionaryAttr b) {
        return a.getAs<IntegerAttr>("slot").getInt() < b.getAs<IntegerAttr>("slot").getInt();
      });
      for (auto slot : slots) {
        StringRef kind = slot.getAs<StringAttr>("kind").getValue();
        if (kind == "resource") {
          call("ckl_argument_resource", {frame,
               number(resources.at(slot.getAs<StringAttr>("resource").getValue()))});
          continue;
        }
        Value value;
        uint64_t size;
        if (kind == "bytes") {
          auto data = slot.getAs<DenseI8ArrayAttr>("value").asArrayRef();
          size = data.size();
          value = bytes(StringRef(reinterpret_cast<const char *>(data.data()), size));
        } else {
          auto type = slot.getAs<TypeAttr>("type").getValue();
          if (!type.isIntOrFloat() || type.getIntOrFloatBitWidth() % 8 ||
              type.getIntOrFloatBitWidth() > 64)
            return kernel.emitError("LLVM host lowering requires byte-sized scalars up to 64 bits");
          size = type.getIntOrFloatBitWidth() / 8;
          if (kind == "scalar") {
            auto argument = binding(slot.getAs<StringAttr>("name").getValue(), false, type);
            if (failed(argument))
              return failure();
            value = *argument;
          } else if (kind == "constant") {
            // Match the cuda.direct little-endian physical scalar ABI.
            auto attr = slot.get("value");
            uint64_t bits =
                isa<IntegerAttr>(attr)
                    ? cast<IntegerAttr>(attr).getValue().getZExtValue()
                    : cast<FloatAttr>(attr).getValue().bitcastToAPInt().getZExtValue();
            std::string data(size, '\0');
            for (uint64_t i = 0; i < size; ++i)
              data[i] = (bits >> (8 * i)) & 255;
            value = bytes(data);
          } else {
            return kernel.emitError("unsupported LLVM host argument binding");
          }
        }
        call("ckl_argument_value", {frame, value, number(size)});
      }
      for (int64_t predecessor : kernel.getDependencies())
        call("ckl_dependency", {frame, number(predecessor)});
      call("ckl_commit", {frame, number(kernel.getId())});
    }
    LLVM::ReturnOp::create(b, loc, ValueRange{});
    return success();
  }

private:
  Value number(uint64_t value) {
    return LLVM::ConstantOp::create(b, loc, b.getI64Type(), b.getI64IntegerAttr(value));
  }

  // Keep embedded bytes in MLIR globals; upstream LLVM translation owns their
  // conversion to LLVM IR, including binary strings with embedded NUL bytes.
  Value bytes(StringRef data, StringRef name = {}) {
    std::string storage = data.str();
    storage.push_back('\0');
    std::string symbol = name.empty() ? "ckl.data." + std::to_string(nextGlobal++) : name.str();
    LLVM::GlobalOp global;
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(output.getBody());
      global = LLVM::GlobalOp::create(b, loc,
          LLVM::LLVMArrayType::get(b.getI8Type(), storage.size()), true,
          LLVM::Linkage::Internal, symbol, b.getStringAttr(storage));
    }
    return LLVM::AddressOfOp::create(b, loc, global);
  }

  void call(StringRef name, ValueRange values) {
    auto function = output.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!function) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(output.getBody());
      auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(b.getContext()),
                                             llvm::to_vector(values.getTypes()));
      function = LLVM::LLVMFuncOp::create(b, loc, name, type);
    }
    LLVM::CallOp::create(b, loc, function, values);
  }

  FailureOr<Value> binding(StringRef name, bool buffer, Type scalarType = {}) {
    unsigned index;
    if (!name.consume_front("arg") || name.getAsInteger(10, index) ||
        index >= source.getNumArguments()) {
      source.emitError(
          "LLVM host lowering requires original argN bindings within the source signature");
      return failure();
    }
    Type type = source.getArgumentTypes()[index];
    if ((buffer && !isa<MemRefType>(type)) || (!buffer && type != scalarType)) {
      source.emitError("LLVM host binding type does not match the source signature");
      return failure();
    }
    Value offset = number(index);
    Value slot = LLVM::GEPOp::create(b, loc, ptr, ptr, arguments, ValueRange{offset});
    return LLVM::LoadOp::create(b, loc, ptr, slot).getResult();
  }

  ModuleOp output;
  func::FuncOp source;
  OpBuilder b;
  Location loc;
  Type ptr;
  Value frame, arguments;
  unsigned nextGlobal = 0;
};

struct LowerExecToLLVMPass : PassWrapper<LowerExecToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerExecToLLVMPass)
  StringRef getArgument() const final { return "ckl-lower-exec-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower a CUDA executable plan and embedded binary to the LLVM host runtime ABI";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<exec::PlanOp> plans(module.getOps<exec::PlanOp>());
    auto binary = module->getAttrOfType<StringAttr>("ckl_exec.binary");
    if (plans.size() != 1 || !binary || binary.getValue().empty()) {
      module.emitError(
          "LLVM host lowering requires one executable plan and non-empty ckl_exec.binary");
      return signalPassFailure();
    }
    auto source =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(plans[0], plans[0].getSource());
    if (!source || !source.isDeclaration() || source.getNumResults()) {
      plans[0].emitError(
          "LLVM host lowering requires a source function declaration with no results");
      return signalPassFailure();
    }
    // A dedicated host module prevents silently discarding unrelated operations.
    for (Operation &op : module.getBody()->getOperations())
      if (&op != plans[0].getOperation() && &op != source.getOperation()) {
        op.emitError("unexpected operation in the executable host module");
        return signalPassFailure();
      }
    OwningOpRef<ModuleOp> lowered = ModuleOp::create(module.getLoc());
    if (failed(HostLowering(*lowered, source).lower(plans[0], binary.getValue())))
      return signalPassFailure();
    module.getBodyRegion().takeBody(lowered->getBodyRegion());
    module->removeAttr("ckl_exec.binary");
  }
};
} // namespace

void mlir::ckl::registerCKLExecToLLVMPass() { PassRegistration<LowerExecToLLVMPass>(); }

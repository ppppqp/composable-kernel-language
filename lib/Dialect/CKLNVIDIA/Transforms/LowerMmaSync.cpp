#include "ckl/Dialect/CKLNVIDIA/Transforms/Passes.h"

#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIAOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::ckl::nvidia {
namespace {

class LowerMmaSyncPass : public PassWrapper<LowerMmaSyncPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMmaSyncPass)
  StringRef getArgument() const final { return "ckl-nvidia-lower-mma-sync"; }
  StringRef getDescription() const final {
    return "Expose fragment packing and lower fixed CKL NVIDIA MMA to NVGPU";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<CKLNVIDIADialect, ::mlir::nvgpu::NVGPUDialect>();
  }

  void runOnOperation() final {
    SmallVector<MmaSyncOp> operations;
    getOperation().walk([&](MmaSyncOp op) { operations.push_back(op); });
    IRRewriter rewriter(&getContext());
    auto f16 = rewriter.getF16Type();
    auto f32 = rewriter.getF32Type();
    auto lhsType = VectorType::get({4, 2}, f16);
    auto rhsType = VectorType::get({2, 2}, f16);
    auto accumulatorType = VectorType::get({2, 2}, f32);
    for (MmaSyncOp op : operations) {
      rewriter.setInsertionPoint(op);
      auto pack = [&](Value tile, VectorType type, StringRef role) {
        OperationState state(op.getLoc(), PackFragmentOp::getOperationName());
        state.addOperands(tile);
        state.addTypes(type);
        state.addAttribute("role", rewriter.getStringAttr(role));
        return rewriter.create(state)->getResult(0);
      };
      Value lhs = pack(op.getLhs(), lhsType, "lhs");
      Value rhs = pack(op.getRhs(), rhsType, "rhs");
      Value acc = pack(op.getAcc(), accumulatorType, "acc");

      auto mma = ::mlir::nvgpu::MmaSyncOp::create(
          rewriter, op.getLoc(), lhs, rhs, acc, ArrayRef<std::int64_t>{16, 8, 16});
      for (NamedAttribute attribute : op->getDiscardableAttrs())
        if (attribute.getName().getValue().starts_with("ckl."))
          mma->setAttr(attribute.getName(), attribute.getValue());
      Value fragment = mma.getRes();

      OperationState unpackState(op.getLoc(), UnpackFragmentOp::getOperationName());
      unpackState.addOperands(fragment);
      unpackState.addTypes(op.getResult().getType());
      unpackState.addAttribute("role", rewriter.getStringAttr("result"));
      Operation *unpack = rewriter.create(unpackState);
      rewriter.replaceOp(op, unpack->getResults());
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerMmaSyncPass() {
  return std::make_unique<LowerMmaSyncPass>();
}

void registerLowerMmaSyncPass() { PassRegistration<LowerMmaSyncPass>(); }

} // namespace mlir::ckl::nvidia

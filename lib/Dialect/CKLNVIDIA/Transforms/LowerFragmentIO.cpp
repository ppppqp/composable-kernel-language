#include "ckl/Dialect/CKLNVIDIA/Transforms/Passes.h"

#include "ckl/Dialect/CKL/IR/CKLOps.h"
#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIAOps.h"
#include "ckl/Extensions/NVIDIA/MmaSync.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::ckl::nvidia {
namespace {

Value constantIndex(IRRewriter &rewriter, Location location, std::int64_t value) {
  return arith::ConstantIndexOp::create(rewriter, location, value);
}

Value add(IRRewriter &rewriter, Location location, Value lhs, Value rhs) {
  return arith::AddIOp::create(rewriter, location, lhs, rhs);
}

Value multiply(IRRewriter &rewriter, Location location, Value lhs, std::int64_t rhs) {
  return arith::MulIOp::create(rewriter, location, lhs,
                              constantIndex(rewriter, location, rhs));
}

struct LaneCoordinates {
  Value group;
  Value thread;
};

LaneCoordinates getLaneCoordinates(IRRewriter &rewriter, Location location) {
  Value threadId = gpu::ThreadIdOp::create(rewriter, location, gpu::Dimension::x);
  Value lane = arith::RemUIOp::create(rewriter, location, threadId,
                                     constantIndex(rewriter, location, 32));
  return {arith::DivUIOp::create(rewriter, location, lane,
                                constantIndex(rewriter, location, 4)),
          arith::RemUIOp::create(rewriter, location, lane,
                                constantIndex(rewriter, location, 4))};
}

std::pair<Value, Value> getTileCoordinate(IRRewriter &rewriter, Location location,
                                          LaneCoordinates lane, StringRef role,
                                          std::int64_t value) {
  if (role == "lhs") {
    std::int64_t kHalf = value / 4;
    std::int64_t rowHalf = (value / 2) % 2;
    std::int64_t pair = value % 2;
    Value row = add(rewriter, location, lane.group,
                    constantIndex(rewriter, location, rowHalf * 8));
    Value column = add(rewriter, location, multiply(rewriter, location, lane.thread, 2),
                       constantIndex(rewriter, location, pair + kHalf * 8));
    return {row, column};
  }
  std::int64_t half = value / 2;
  std::int64_t pair = value % 2;
  if (role == "rhs") {
    Value row = add(rewriter, location, multiply(rewriter, location, lane.thread, 2),
                    constantIndex(rewriter, location, pair + half * 8));
    return {row, lane.group};
  }
  Value row = add(rewriter, location, lane.group,
                  constantIndex(rewriter, location, half * 8));
  Value column = add(rewriter, location, multiply(rewriter, location, lane.thread, 2),
                     constantIndex(rewriter, location, pair));
  return {row, column};
}

::ckl::core::Distribution expectedDistribution(StringRef role) {
  if (role == "lhs")
    return ::ckl::extensions::nvidia::makeMmaSyncM16N8K16F16LhsDistribution();
  if (role == "rhs")
    return ::ckl::extensions::nvidia::makeMmaSyncM16N8K16F16RhsDistribution();
  return ::ckl::extensions::nvidia::makeMmaSyncM16N8K16F32AccumulatorDistribution();
}

class LowerFragmentIOPass
    : public PassWrapper<LowerFragmentIOPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFragmentIOPass)
  StringRef getArgument() const final { return "ckl-nvidia-lower-fragment-io"; }
  StringRef getDescription() const final {
    return "Lower fixed MMA fragment packing to per-lane memory operations";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() final {
    SmallVector<PackFragmentOp> packs;
    SmallVector<::mlir::ckl::StoreTileOp> stores;
    getOperation().walk([&](PackFragmentOp op) { packs.push_back(op); });
    getOperation().walk([&](::mlir::ckl::StoreTileOp op) { stores.push_back(op); });
    IRRewriter rewriter(&getContext());

    for (PackFragmentOp pack : packs) {
      auto load = pack.getTile().getDefiningOp<::mlir::ckl::LoadTileOp>();
      if (!load) {
        pack.emitError("fixed fragment lowering requires a directly associated ckl.load_tile");
        signalPassFailure();
        return;
      }
      StringRef role = pack.getRole();
      if (load.getOffsets().size() != 2 ||
          load.getDistribution().getValue() !=
              ::ckl::core::serialize(expectedDistribution(role))) {
        load.emitError("distribution does not match the selected NVIDIA fragment role '")
            << role << "'";
        signalPassFailure();
        return;
      }
      rewriter.setInsertionPoint(pack);
      LaneCoordinates lane = getLaneCoordinates(rewriter, pack.getLoc());
      VectorType type = pack.getFragment().getType();
      Value fragment = arith::ConstantOp::create(rewriter, pack.getLoc(), type,
                                                 rewriter.getZeroAttr(type));
      const std::int64_t values = type.getNumElements();
      for (std::int64_t value = 0; value < values; ++value) {
        auto [row, column] = getTileCoordinate(rewriter, pack.getLoc(), lane, role, value);
        Value memoryRow = add(rewriter, pack.getLoc(), load.getOffsets()[0], row);
        Value memoryColumn = add(rewriter, pack.getLoc(), load.getOffsets()[1], column);
        Value scalar = memref::LoadOp::create(rewriter, pack.getLoc(), load.getSource(),
                                              ValueRange{memoryRow, memoryColumn});
        fragment = vector::InsertOp::create(rewriter, pack.getLoc(), scalar, fragment,
                                            ArrayRef<std::int64_t>{value / type.getShape()[1],
                                                                   value % type.getShape()[1]});
      }
      rewriter.replaceOp(pack, fragment);
      if (load->use_empty())
        rewriter.eraseOp(load);
    }

    for (::mlir::ckl::StoreTileOp store : stores) {
      auto unpack = store.getValue().getDefiningOp<UnpackFragmentOp>();
      if (!unpack)
        continue;
      if (store.getOffsets().size() != 2 ||
          store.getDistribution().getValue() != ::ckl::core::serialize(
                                                    expectedDistribution("result"))) {
        store.emitError("distribution does not match the selected NVIDIA result fragment");
        signalPassFailure();
        return;
      }
      rewriter.setInsertionPoint(store);
      LaneCoordinates lane = getLaneCoordinates(rewriter, store.getLoc());
      for (std::int64_t value = 0; value < 4; ++value) {
        auto [row, column] =
            getTileCoordinate(rewriter, store.getLoc(), lane, "result", value);
        Value memoryRow = add(rewriter, store.getLoc(), store.getOffsets()[0], row);
        Value memoryColumn = add(rewriter, store.getLoc(), store.getOffsets()[1], column);
        Value scalar = vector::ExtractOp::create(
            rewriter, store.getLoc(), unpack.getFragment(),
            ArrayRef<std::int64_t>{value / 2, value % 2});
        memref::StoreOp::create(rewriter, store.getLoc(), scalar, store.getTarget(),
                                ValueRange{memoryRow, memoryColumn});
      }
      rewriter.eraseOp(store);
      if (unpack->use_empty())
        rewriter.eraseOp(unpack);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerFragmentIOPass() {
  return std::make_unique<LowerFragmentIOPass>();
}

void registerLowerFragmentIOPass() { PassRegistration<LowerFragmentIOPass>(); }

} // namespace mlir::ckl::nvidia

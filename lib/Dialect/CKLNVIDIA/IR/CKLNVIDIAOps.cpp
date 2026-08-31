#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIAOps.h"

#include "ckl/Core/Layout/IndexMap.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::ckl;
using namespace mlir::ckl::nvidia;

#define GET_OP_CLASSES
#include "ckl/Dialect/CKLNVIDIA/IR/CKLNVIDIAOps.cpp.inc"

namespace {

LogicalResult verifyTile(Operation *op, TileType type, Type elementType,
                         ArrayRef<std::int64_t> shape, StringRef role) {
  if (type.getElementType() != elementType)
    return op->emitOpError() << role << " must have element type " << elementType;
  try {
    auto space = ::ckl::core::IndexSpace::deserialize(type.getSpace().getValue().str());
    if (space.rank() != shape.size())
      return op->emitOpError() << role << " must have rank " << shape.size();
    for (auto [axis, expected] : llvm::zip(space.axes(), shape))
      if (!axis.isStatic() || axis.extent != expected)
        return op->emitOpError() << role << " must have static shape " << shape;
  } catch (const std::exception &error) {
    return op->emitOpError() << "cannot inspect " << role << " tile: " << error.what();
  }
  return success();
}

} // namespace

LogicalResult MmaSyncOp::verify() {
  auto f16 = Float16Type::get(getContext());
  auto f32 = Float32Type::get(getContext());
  if (failed(verifyTile(*this, getLhs().getType(), f16, {16, 16}, "lhs")) ||
      failed(verifyTile(*this, getRhs().getType(), f16, {16, 8}, "rhs")) ||
      failed(verifyTile(*this, getAcc().getType(), f32, {16, 8}, "accumulator")) ||
      failed(verifyTile(*this, getResult().getType(), f32, {16, 8}, "result")))
    return failure();
  return success();
}

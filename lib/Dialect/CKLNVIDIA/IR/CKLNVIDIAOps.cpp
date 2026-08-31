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

LogicalResult PackFragmentOp::verify() {
  auto vector = getFragment().getType();
  StringRef role = getRole();
  Type elementType;
  ArrayRef<std::int64_t> tileShape;
  ArrayRef<std::int64_t> fragmentShape;
  const std::int64_t lhsTile[] = {16, 16};
  const std::int64_t matrixTile[] = {16, 8};
  const std::int64_t lhsFragment[] = {4, 2};
  const std::int64_t matrixFragment[] = {2, 2};
  if (role == "lhs") {
    elementType = Float16Type::get(getContext());
    tileShape = lhsTile;
    fragmentShape = lhsFragment;
  } else if (role == "rhs") {
    elementType = Float16Type::get(getContext());
    tileShape = matrixTile;
    fragmentShape = matrixFragment;
  } else if (role == "acc") {
    elementType = Float32Type::get(getContext());
    tileShape = matrixTile;
    fragmentShape = matrixFragment;
  } else {
    return emitOpError("role must be 'lhs', 'rhs', or 'acc'");
  }
  if (failed(verifyTile(*this, getTile().getType(), elementType, tileShape, role)))
    return failure();
  if (vector.getElementType() != elementType || vector.getShape() != fragmentShape)
    return emitOpError() << role << " fragment must have type vector<" << fragmentShape << "x"
                         << elementType << ">";
  return success();
}

LogicalResult UnpackFragmentOp::verify() {
  if (getRole() != "result")
    return emitOpError("role must be 'result'");
  auto f32 = Float32Type::get(getContext());
  const std::int64_t matrixTile[] = {16, 8};
  const std::int64_t matrixFragment[] = {2, 2};
  if (getFragment().getType().getElementType() != f32 ||
      getFragment().getType().getShape() != ArrayRef<std::int64_t>(matrixFragment))
    return emitOpError("result fragment must have type vector<2x2xf32>");
  return verifyTile(*this, getTile().getType(), f32, matrixTile, "result");
}

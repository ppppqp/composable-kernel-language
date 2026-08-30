#include "ckl/Dialect/CKL/IR/CKLAttrs.h"

#include "ckl/Core/Layout/IndexMap.h"
#include "ckl/Core/Layout/Distribution.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ckl;

#define GET_ATTRDEF_CLASSES
#include "ckl/Dialect/CKL/IR/CKLAttrs.cpp.inc"

void CKLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ckl/Dialect/CKL/IR/CKLAttrs.cpp.inc"
      >();
}

LogicalResult SpaceAttr::verify(function_ref<InFlightDiagnostic()> emitError, StringRef value) {
  try {
    ::ckl::core::IndexSpace space = ::ckl::core::IndexSpace::deserialize(value.str());
    if (space.serialize() != value)
      return emitError() << "space must use canonical CKLCore spelling '" << space.serialize()
                         << "'";
  } catch (const std::exception &error) {
    return emitError() << "invalid CKL index space: " << error.what();
  }
  return success();
}

LogicalResult IndexMapAttr::verify(function_ref<InFlightDiagnostic()> emitError, StringRef value) {
  try {
    ::ckl::core::IndexMap map = ::ckl::core::IndexMap::deserialize(value.str());
    if (map.serialize() != value)
      return emitError() << "index map must use canonical CKLCore spelling '" << map.serialize()
                         << "'";
  } catch (const std::exception &error) {
    return emitError() << "invalid CKL index map: " << error.what();
  }
  return success();
}

LogicalResult DistributionAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                       StringRef value) {
  try {
    ::ckl::core::Distribution distribution = ::ckl::core::deserializeDistribution(value.str());
    if (::ckl::core::serialize(distribution) != value)
      return emitError() << "distribution must use canonical CKLCore spelling";
    auto check = ::ckl::core::verifyDistribution(distribution);
    if (!check.valid)
      return emitError() << "invalid CKL distribution: " << check.message;
  } catch (const std::exception &error) {
    return emitError() << "invalid CKL distribution: " << error.what();
  }
  return success();
}

#include "ckl/Dialect/CKL/IR/CKLInterfaces.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"

#include "ckl/Dialect/CKL/IR/CKLInterfaces.cpp.inc"

using namespace mlir;

namespace {

std::optional<int64_t> constantInteger(Value value) {
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return std::nullopt;
  return integer.getSExtValue();
}

DenseI64ArrayAttr dimensions(MLIRContext *context, gpu::KernelDim3 values) {
  std::optional<int64_t> x = constantInteger(values.x);
  std::optional<int64_t> y = constantInteger(values.y);
  std::optional<int64_t> z = constantInteger(values.z);
  if (!x || !y || !z)
    return {};
  return DenseI64ArrayAttr::get(context, {*x, *y, *z});
}

struct GPULaunchFuncDispatchModel
    : public ckl::DispatchOpInterface::ExternalModel<GPULaunchFuncDispatchModel,
                                                     gpu::LaunchFuncOp> {
  SymbolRefAttr getKernelSymbol(Operation *operation) const {
    return cast<gpu::LaunchFuncOp>(operation).getKernel();
  }

  OperandRange getDispatchArguments(Operation *operation) const {
    return cast<gpu::LaunchFuncOp>(operation).getKernelOperands();
  }

  DenseI64ArrayAttr getGridDimensions(Operation *operation) const {
    auto launch = cast<gpu::LaunchFuncOp>(operation);
    return dimensions(operation->getContext(), launch.getGridSizeOperandValues());
  }

  DenseI64ArrayAttr getBlockDimensions(Operation *operation) const {
    auto launch = cast<gpu::LaunchFuncOp>(operation);
    return dimensions(operation->getContext(), launch.getBlockSizeOperandValues());
  }

  IntegerAttr getDynamicSharedMemory(Operation *operation) const {
    auto launch = cast<gpu::LaunchFuncOp>(operation);
    if (!launch.getDynamicSharedMemorySize())
      return IntegerAttr::get(IntegerType::get(operation->getContext(), 64), 0);
    std::optional<int64_t> value = constantInteger(launch.getDynamicSharedMemorySize());
    if (!value)
      return {};
    return IntegerAttr::get(IntegerType::get(operation->getContext(), 64), *value);
  }

  StringAttr getTargetDevice(Operation *operation) const {
    // e.g. "cuda:0"
    return operation->getAttrOfType<StringAttr>("ckl.device");
  }

  ArrayAttr getRequiredCapabilities(Operation *operation) const {
    if (auto capabilities = operation->getAttrOfType<ArrayAttr>("ckl.capabilities"))
      return capabilities;
    return ArrayAttr::get(operation->getContext(), {});
  }

  StringAttr getImplementationIdentity(Operation *operation) const {
    // e.g. compiler cache hash
    return operation->getAttrOfType<StringAttr>("ckl.implementation");
  }

  StringAttr getArtifactIdentity(Operation *operation) const {
    if (auto artifact = operation->getAttrOfType<StringAttr>("ckl.artifact"))
      return artifact;
    return getImplementationIdentity(operation);
  }

  StringAttr getArgumentABI(Operation *operation) const {
    // e.g. cuda.direct
    return operation->getAttrOfType<StringAttr>("ckl.abi");
  }

  OperandRange getExplicitDependencies(Operation *operation) const {
    return cast<gpu::LaunchFuncOp>(operation).getAsyncDependencies();
  }
};

} // namespace

void mlir::ckl::registerProducerInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, gpu::GPUDialect *) {
    gpu::LaunchFuncOp::attachInterface<GPULaunchFuncDispatchModel>(*context);
  });
}

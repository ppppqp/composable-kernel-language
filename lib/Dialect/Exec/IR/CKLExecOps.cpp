#include "ckl/Dialect/Exec/IR/CKLExecOps.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::ckl::exec;

namespace {

LogicalResult verifyDimensions(Operation *operation, ArrayRef<int64_t> dimensions,
                               StringRef name) {
  if (dimensions.size() != 3)
    return operation->emitOpError() << "requires exactly three " << name << " dimensions";
  if (llvm::any_of(dimensions, [](int64_t value) { return value <= 0; }))
    return operation->emitOpError() << "requires positive " << name << " dimensions";
  return success();
}

} // namespace

LogicalResult HeapOp::verify() {
  if (getIdAttr().getInt() < 0 || getBytesAttr().getInt() <= 0)
    return emitOpError("requires a non-negative id and positive byte size");
  int64_t alignment = getAlignmentAttr().getInt();
  if (alignment <= 0 || (alignment & (alignment - 1)))
    return emitOpError("requires power-of-two alignment");
  if (getDeviceAttr().getInt() < 0)
    return emitOpError("requires a non-negative device ordinal");
  return success();
}

LogicalResult ResourceOp::verify() {
  if (getName().empty() || (getKind() != "external" && getKind() != "temporary"))
    return emitOpError("requires a name and external or temporary kind");
  int64_t alignment = getAlignmentAttr().getInt();
  if (alignment <= 0 || (alignment & (alignment - 1)))
    return emitOpError("requires power-of-two alignment");
  if (getKind() == "temporary") {
    if (!getBytesAttr() || !getHeapAttr() || !getOffsetAttr())
      return emitOpError("temporary resource requires bytes, heap, and offset");
    if (getBytesAttr().getInt() <= 0 || getHeapAttr().getInt() < 0 ||
        getOffsetAttr().getInt() < 0)
      return emitOpError("temporary resource requires positive bytes and non-negative placement");
  } else if (getHeapAttr() || getOffsetAttr()) {
    return emitOpError("external resource cannot have a heap placement");
  }
  return success();
}

LogicalResult KernelOp::verify() {
  if (getIdAttr().getInt() < 0)
    return emitOpError("requires a non-negative id");
  if (getImplementation().empty() || getArtifact().empty() || getAbi().empty() ||
      getDevice().empty())
    return emitOpError("requires implementation, artifact, ABI, and device identities");
  if (failed(verifyDimensions(getOperation(), getGrid(), "grid")) ||
      failed(verifyDimensions(getOperation(), getBlock(), "block")))
    return failure();
  if (getSharedMemoryAttr().getInt() < 0)
    return emitOpError("requires non-negative shared memory");
  llvm::DenseSet<int64_t> seen;
  for (int64_t dependency : getDependencies())
    if (dependency < 0 || dependency >= getIdAttr().getInt())
      return emitOpError("dependencies must reference earlier kernel ids");
    else if (!seen.insert(dependency).second)
      return emitOpError("contains a duplicate dependency");
  return success();
}

LogicalResult PlanOp::verify() {
  if (getBody().empty())
    return emitOpError("requires a body block");
  if (getName().empty() || getBackend().empty())
    return emitOpError("requires plan and backend identities");
  llvm::DenseSet<int64_t> heaps;
  llvm::DenseSet<int64_t> kernels;
  llvm::DenseSet<StringRef> resources;
  llvm::DenseMap<int64_t, int64_t> heapBytes;
  for (Operation &operation : getBody().front()) {
    if (auto heap = dyn_cast<HeapOp>(operation)) {
      if (!heaps.insert(heap.getIdAttr().getInt()).second)
        return heap.emitOpError("has a duplicate heap id");
      heapBytes[heap.getIdAttr().getInt()] = heap.getBytesAttr().getInt();
    } else if (auto resource = dyn_cast<ResourceOp>(operation)) {
      if (!resources.insert(resource.getName()).second)
        return resource.emitOpError("has a duplicate resource name");
    } else if (auto kernel = dyn_cast<KernelOp>(operation)) {
      if (!kernels.insert(kernel.getIdAttr().getInt()).second)
        return kernel.emitOpError("has a duplicate kernel id");
    } else {
      return operation.emitOpError("is not an executable-plan operation");
    }
  }
  for (ResourceOp resource : getBody().front().getOps<ResourceOp>())
    if (resource.getHeapAttr()) {
      int64_t heap = resource.getHeapAttr().getInt();
      if (!heaps.contains(heap))
        return resource.emitOpError("references an unknown heap");
      int64_t offset = resource.getOffsetAttr().getInt();
      int64_t bytes = resource.getBytesAttr().getInt();
      if (offset % resource.getAlignmentAttr().getInt() != 0)
        return resource.emitOpError("heap offset does not satisfy resource alignment");
      if (offset > heapBytes[heap] || bytes > heapBytes[heap] - offset)
        return resource.emitOpError("placement exceeds its heap");
    }
  for (KernelOp kernel : getBody().front().getOps<KernelOp>()) {
    for (int64_t dependency : kernel.getDependencies())
      if (!kernels.contains(dependency))
        return kernel.emitOpError("references an unknown kernel dependency");
    llvm::DenseSet<int64_t> slots;
    for (Attribute attribute : kernel.getArguments()) {
      auto argument = dyn_cast<DictionaryAttr>(attribute);
      if (!argument)
        return kernel.emitOpError("requires dictionary argument descriptors");
      auto slot = argument.getAs<IntegerAttr>("slot");
      auto logicalIndex = argument.getAs<IntegerAttr>("logical_index");
      auto kind = argument.getAs<StringAttr>("kind");
      auto type = argument.getAs<TypeAttr>("type");
      if (!slot || slot.getInt() < 0 || !logicalIndex || logicalIndex.getInt() < 0 || !kind ||
          !type)
        return kernel.emitOpError(
            "argument descriptor requires non-negative slot and logical index plus a type");
      if (!slots.insert(slot.getInt()).second)
        return kernel.emitOpError("contains a duplicate physical argument slot");
      if (kind.getValue() == "resource") {
        auto resource = argument.getAs<StringAttr>("resource");
        if (!resource || !resources.contains(resource.getValue()))
          return kernel.emitOpError("argument references an unknown resource");
        if (kernel.getAbi() == "cuda.direct") {
          auto packing = argument.getAs<StringAttr>("packing");
          if (!packing || packing.getValue() != "direct_pointer")
            return kernel.emitOpError(
                "cuda.direct resource arguments require direct_pointer packing");
        } else if (kernel.getAbi() == "rocm.bare_ptr") {
          auto packing = argument.getAs<StringAttr>("packing");
          if (!packing || packing.getValue() != "bare_pointer")
            return kernel.emitOpError(
                "rocm.bare_ptr resource arguments require bare_pointer packing");
        }
      } else if (kind.getValue() == "scalar") {
        auto name = argument.getAs<StringAttr>("name");
        if (!name || name.getValue().empty())
          return kernel.emitOpError("scalar argument requires a non-empty binding name");
      } else if (kind.getValue() == "constant") {
        auto value = dyn_cast_or_null<TypedAttr>(argument.get("value"));
        if (!value || value.getType() != type.getValue())
          return kernel.emitOpError("constant argument requires a value matching its type");
      } else if (kind.getValue() == "bytes") {
        auto value = dyn_cast_or_null<DenseI8ArrayAttr>(argument.get("value"));
        auto vector = dyn_cast<VectorType>(type.getValue());
        if (!value || value.empty() || !vector || vector.getRank() != 1 ||
            !vector.getElementType().isInteger(8) ||
            static_cast<int64_t>(value.size()) != vector.getNumElements())
          return kernel.emitOpError(
              "bytes argument requires a non-empty vector<Nxi8> type and N byte values");
      } else {
        return kernel.emitOpError("contains an unsupported physical argument kind");
      }
    }
    for (int64_t slot = 0, end = kernel.getArguments().size(); slot < end; ++slot)
      if (!slots.contains(slot))
        return kernel.emitOpError("physical argument slots must be contiguous");
  }
  return success();
}

#define GET_OP_CLASSES
#include "ckl/Dialect/Exec/IR/CKLExecOps.cpp.inc"

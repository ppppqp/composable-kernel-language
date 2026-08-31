#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include <cuda.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void checkCuda(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS)
    return;
  const char *name = nullptr;
  const char *description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(std::string(operation) + " failed: " +
                           (name ? name : "unknown") + " (" +
                           (description ? description : "no description") + ")");
}

std::vector<char> readCubin(mlir::ModuleOp module) {
  std::vector<char> cubin;
  module.walk([&](mlir::gpu::BinaryOp binary) {
    if (!cubin.empty())
      throw std::runtime_error("expected exactly one GPU binary");
    if (binary.getObjects().size() != 1)
      throw std::runtime_error("expected exactly one GPU object");
    auto object = llvm::dyn_cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]);
    if (!object || object.getFormat() != mlir::gpu::CompilationTarget::Binary)
      throw std::runtime_error("expected a binary GPU object");
    llvm::StringRef bytes = object.getObject().getValue();
    cubin.assign(bytes.begin(), bytes.end());
  });
  if (cubin.empty())
    throw std::runtime_error("no GPU binary found");
  cubin.push_back(0);
  return cubin;
}

struct MemRef2D {
  CUdeviceptr allocated;
  CUdeviceptr aligned;
  int64_t offset;
  int64_t size0;
  int64_t size1;
  int64_t stride0;
  int64_t stride1;
};

void appendDescriptor(std::vector<void *> &parameters, MemRef2D &descriptor) {
  parameters.push_back(&descriptor.allocated);
  parameters.push_back(&descriptor.aligned);
  parameters.push_back(&descriptor.offset);
  parameters.push_back(&descriptor.size0);
  parameters.push_back(&descriptor.size1);
  parameters.push_back(&descriptor.stride0);
  parameters.push_back(&descriptor.stride1);
}

uint16_t integerHalf(unsigned value) {
  static constexpr std::array<uint16_t, 4> values = {
      0x0000, 0x3c00, 0x4000, 0x4200};
  return values.at(value);
}

} // namespace

int main(int argc, char **argv) try {
  if (argc != 2)
    throw std::runtime_error("usage: ckl-nvidia-runtime-validation <binary.mlir>");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module)
    throw std::runtime_error("failed to parse serialized GPU module");
  std::vector<char> cubin = readCubin(*module);

  checkCuda(cuInit(0), "cuInit");
  CUdevice device;
  checkCuda(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext cudaContext;
  checkCuda(cuCtxCreate(&cudaContext, 0, device), "cuCtxCreate");
  CUmodule cudaModule;
  checkCuda(cuModuleLoadData(&cudaModule, cubin.data()), "cuModuleLoadData");
  CUfunction kernel;
  checkCuda(cuModuleGetFunction(&kernel, cudaModule, "kernel"),
            "cuModuleGetFunction");

  std::vector<uint16_t> a(16 * 16);
  std::vector<uint16_t> b(16 * 8);
  std::vector<float> c(16 * 8);
  std::vector<float> d(16 * 8, -1.0f);
  for (int m = 0; m < 16; ++m)
    for (int k = 0; k < 16; ++k)
      a[m * 16 + k] = integerHalf(1 + m % 3);
  for (int k = 0; k < 16; ++k)
    for (int n = 0; n < 8; ++n)
      b[k * 8 + n] = integerHalf(1 + n % 2);
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n)
      c[m * 8 + n] = static_cast<float>((m + n) % 4);

  MemRef2D aDesc{}, bDesc{}, cDesc{}, dDesc{};
  checkCuda(cuMemAlloc(&aDesc.allocated, a.size() * sizeof(uint16_t)), "cuMemAlloc(a)");
  checkCuda(cuMemAlloc(&bDesc.allocated, b.size() * sizeof(uint16_t)), "cuMemAlloc(b)");
  checkCuda(cuMemAlloc(&cDesc.allocated, c.size() * sizeof(float)), "cuMemAlloc(c)");
  checkCuda(cuMemAlloc(&dDesc.allocated, d.size() * sizeof(float)), "cuMemAlloc(d)");
  aDesc.aligned = aDesc.allocated;
  bDesc.aligned = bDesc.allocated;
  cDesc.aligned = cDesc.allocated;
  dDesc.aligned = dDesc.allocated;
  aDesc.size0 = 16; aDesc.size1 = 16; aDesc.stride0 = 16; aDesc.stride1 = 1;
  bDesc.size0 = 16; bDesc.size1 = 8;  bDesc.stride0 = 8;  bDesc.stride1 = 1;
  cDesc.size0 = 16; cDesc.size1 = 8;  cDesc.stride0 = 8;  cDesc.stride1 = 1;
  dDesc.size0 = 16; dDesc.size1 = 8;  dDesc.stride0 = 8;  dDesc.stride1 = 1;

  checkCuda(cuMemcpyHtoD(aDesc.aligned, a.data(), a.size() * sizeof(uint16_t)),
            "cuMemcpyHtoD(a)");
  checkCuda(cuMemcpyHtoD(bDesc.aligned, b.data(), b.size() * sizeof(uint16_t)),
            "cuMemcpyHtoD(b)");
  checkCuda(cuMemcpyHtoD(cDesc.aligned, c.data(), c.size() * sizeof(float)),
            "cuMemcpyHtoD(c)");
  checkCuda(cuMemcpyHtoD(dDesc.aligned, d.data(), d.size() * sizeof(float)),
            "cuMemcpyHtoD(d)");

  std::vector<void *> parameters;
  appendDescriptor(parameters, aDesc);
  appendDescriptor(parameters, bDesc);
  appendDescriptor(parameters, cDesc);
  appendDescriptor(parameters, dDesc);
  checkCuda(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr,
                           parameters.data(), nullptr),
            "cuLaunchKernel");
  checkCuda(cuCtxSynchronize(), "cuCtxSynchronize");
  checkCuda(cuMemcpyDtoH(d.data(), dDesc.aligned, d.size() * sizeof(float)),
            "cuMemcpyDtoH(d)");

  for (int m = 0; m < 16; ++m) {
    for (int n = 0; n < 8; ++n) {
      float expected = 16.0f * static_cast<float>(1 + m % 3) *
                           static_cast<float>(1 + n % 2) + c[m * 8 + n];
      if (std::fabs(d[m * 8 + n] - expected) > 1.0e-4f)
        throw std::runtime_error("result mismatch at (" + std::to_string(m) +
                                 ", " + std::to_string(n) + "): got " +
                                 std::to_string(d[m * 8 + n]) + ", expected " +
                                 std::to_string(expected));
    }
  }

  checkCuda(cuMemFree(aDesc.allocated), "cuMemFree(a)");
  checkCuda(cuMemFree(bDesc.allocated), "cuMemFree(b)");
  checkCuda(cuMemFree(cDesc.allocated), "cuMemFree(c)");
  checkCuda(cuMemFree(dDesc.allocated), "cuMemFree(d)");
  checkCuda(cuModuleUnload(cudaModule), "cuModuleUnload");
  checkCuda(cuCtxDestroy(cudaContext), "cuCtxDestroy");
  std::cout << "validated 128 results\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

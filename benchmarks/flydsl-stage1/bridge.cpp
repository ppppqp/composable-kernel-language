#include "host.cpp.inc"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace mlir::ckl::runtime;

namespace {

thread_local std::string lastError;

struct Handle {
  explicit Handle(int device) : runtime(device) {}

  HipRuntime runtime;
  HipStream stream;
  ArtifactRegistry artifacts;
  std::vector<HipBuffer> retained;
  HipPlan plan;
  HipGraphExecutable graph;
};

template <typename Function> int protect(Function &&function) {
  try {
    function();
    lastError.clear();
    return 0;
  } catch (const std::exception &error) {
    lastError = error.what();
    return -1;
  }
}

} // namespace

extern "C" {

const char *ckl_flydsl_stage1_last_error() { return lastError.c_str(); }

void *ckl_flydsl_stage1_create(const char *bundleDirectory, int device, std::uintptr_t stream,
                               std::uint64_t input, std::uint64_t gamma, std::uint64_t dy,
                               std::uint64_t rstd, std::uint64_t dx, std::uint64_t dweight,
                               std::uint64_t partial, std::int32_t rows, std::int32_t columns,
                               std::int32_t programs) {
  try {
    if (!bundleDirectory || rows <= 0 || columns <= 0 || programs <= 0)
      throw std::invalid_argument("invalid FlyDSL Stage 1 handle arguments");
    auto handle = std::make_unique<Handle>(device);
    handle->stream = handle->runtime.importStream(stream);
    const std::size_t matrixBytes =
        static_cast<std::size_t>(rows) * columns * sizeof(float);
    const std::size_t vectorBytes = static_cast<std::size_t>(columns) * sizeof(float);
    const std::size_t partialBytes =
        static_cast<std::size_t>(programs) * columns * sizeof(float);

    HipBuffer inputBuffer = handle->runtime.importBuffer(input, matrixBytes);
    HipBuffer gammaBuffer = handle->runtime.importBuffer(gamma, vectorBytes);
    HipBuffer dyBuffer = handle->runtime.importBuffer(dy, matrixBytes);
    HipBuffer rstdBuffer =
        handle->runtime.importBuffer(rstd, static_cast<std::size_t>(rows) * sizeof(float));
    HipBuffer dxBuffer = handle->runtime.importBuffer(dx, matrixBytes);
    HipBuffer dweightBuffer = handle->runtime.importBuffer(dweight, vectorBytes);
    HipBuffer partialBuffer = handle->runtime.importBuffer(partial, partialBytes);

    auto generated = ckl_generated::build_flydsl_launch_rmsnorm_bwd_two_stage(
        handle->runtime, dweightBuffer, partialBuffer, dxBuffer, dyBuffer, gammaBuffer,
        inputBuffer, rstdBuffer, rows);
    handle->artifacts =
        ckl_generated::load_flydsl_launch_rmsnorm_bwd_two_stage_artifacts(bundleDirectory);
    handle->retained = std::move(generated.retained);
    handle->plan = handle->runtime.resolve(generated.plan, handle->artifacts);
    handle->graph = handle->runtime.instantiate(handle->plan);
    lastError.clear();
    return handle.release();
  } catch (const std::exception &error) {
    lastError = error.what();
    return nullptr;
  }
}

int ckl_flydsl_stage1_launch_ordinary(void *opaque) {
  if (!opaque) {
    lastError = "null FlyDSL Stage 1 handle";
    return -1;
  }
  auto &handle = *static_cast<Handle *>(opaque);
  return protect([&] { handle.runtime.launchOrdinary(handle.plan, handle.stream); });
}

int ckl_flydsl_stage1_launch_graph(void *opaque) {
  if (!opaque) {
    lastError = "null FlyDSL Stage 1 handle";
    return -1;
  }
  auto &handle = *static_cast<Handle *>(opaque);
  return protect([&] { handle.graph.launch(handle.stream); });
}

void ckl_flydsl_stage1_destroy(void *opaque) { delete static_cast<Handle *>(opaque); }

} // extern "C"

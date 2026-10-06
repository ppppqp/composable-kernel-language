#include "ckl/Runtime/HipRuntime.h"

#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace mlir::ckl::runtime;

namespace {

KernelInvocation addBiasInvocation(const std::string &artifact, std::uint64_t values, int elements,
                                   float bias) {
  KernelArguments arguments;
  arguments.add(values).add(elements).add(bias);
  return {artifact,
          "add_bias",
          "rocm.bare_ptr",
          0,
          {(static_cast<unsigned>(elements) + 255) / 256, 1, 1},
          {256, 1, 1},
          0,
          std::move(arguments)};
}

void requireResult(const HipBuffer &buffer, const std::vector<float> &initial, float expectedBias) {
  std::vector<float> result(initial.size());
  buffer.copyToHost(result.data(), result.size() * sizeof(float));
  for (std::size_t index = 0; index < result.size(); ++index) {
    float expected = initial[index] + expectedBias;
    if (!std::isfinite(result[index]) || std::abs(result[index] - expected) > 1e-5f)
      throw std::runtime_error("HIP execution produced an incorrect result");
  }
}

} // namespace

int main(int argc, char **argv) try {
  if (argc != 2) {
    std::cerr << "usage: ckl-hip-runtime-validation <hsaco>\n";
    return 1;
  }

  int deviceCount = 0;
  hipError_t countResult = hipGetDeviceCount(&deviceCount);
  if (countResult != hipSuccess || deviceCount == 0) {
    std::cout << "CKL HIP runtime validation skipped: no accessible HIP device\n";
    return 77;
  }

  constexpr int elements = 1024;
  constexpr std::size_t bytes = elements * sizeof(float);
  std::vector<float> initial(elements);
  for (int index = 0; index < elements; ++index)
    initial[index] = static_cast<float>(index % 29) / 29.0f;

  HipRuntime runtime;
  HipStream stream = runtime.createStream();
  HipBuffer owned = runtime.allocate(bytes);
  HipBuffer borrowed = runtime.importBuffer(owned.address(), owned.size());
  if (borrowed.slice(sizeof(float), sizeof(float)).address() != owned.address() + sizeof(float))
    throw std::runtime_error("borrowed HIP buffer slice lost its byte offset");

  ArtifactRegistry artifacts;
  artifacts.add(KernelArtifact::readFile("flydsl.test", "rocm.hsaco", "test", argv[1]));
  ExecutionPlan plan;
  auto first = plan.addKernel(addBiasInvocation("flydsl.test", borrowed.address(), elements, 1.0f));
  plan.addKernel(addBiasInvocation("flydsl.test", borrowed.address(), elements, 2.0f), {first});

  owned.copyFromHost(initial.data(), bytes);
  runtime.launchOrdinary(plan, artifacts, stream);
  stream.synchronize();
  requireResult(owned, initial, 3.0f);

  owned.copyFromHost(initial.data(), bytes);
  auto graph = runtime.getOrCreateGraph("flydsl.test.plan", plan, artifacts);
  auto graphAgain = runtime.getOrCreateGraph("flydsl.test.plan", plan, artifacts);
  if (graph != graphAgain || graph->nodeCount() != 2 || runtime.cachedGraphCount() != 1)
    throw std::runtime_error("HIP graph cache did not reuse the executable plan");
  graph->launch(stream);
  stream.synchronize();
  requireResult(owned, initial, 3.0f);

  std::cout << "device=" << runtime.deviceName()
            << ",ordinary_correct=true,graph_correct=true,borrowed_resources=true\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ckl-hip-runtime-validation: " << error.what() << '\n';
  return 1;
}

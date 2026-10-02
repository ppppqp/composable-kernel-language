#include "ckl/Runtime/NvidiaRuntime.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace mlir::ckl::runtime;

int main(int argc, char **argv) try {
  if (argc != 2)
    throw std::invalid_argument("usage: ckl-generated-cubin-validation <cubin>");

  constexpr std::int64_t elements = 257;
  NvidiaRuntime runtime;
  NvidiaStream stream = runtime.createStream();
  NvidiaBuffer buffer = runtime.allocate(elements * sizeof(float));
  // Model a framework-owned allocation: the imported view is borrowed and CKL will not free it.
  NvidiaBuffer imported = runtime.importBuffer(buffer.address(), buffer.size());
  std::vector<float> input(elements);
  for (std::int64_t index = 0; index < elements; ++index)
    input[index] = static_cast<float>(index) * 0.25f;
  buffer.copyFromHost(input.data(), buffer.size());

  std::uint64_t pointer = imported.address();
  std::int64_t offset = 0;
  std::int64_t stride = 1;
  KernelArguments arguments;
  arguments.add(pointer)
      .add(pointer)
      .add(offset)
      .add(elements)
      .add(stride)
      .add(elements);
  ArtifactRegistry artifacts;
  artifacts.add(
      KernelArtifact::readFile("producer-object", "cuda.cubin", "sm_test", argv[1]));
  KernelInvocation invocation{"producer-object",
                              "increment",
                              "mlir.strided-memref.v1",
                              0,
                              {static_cast<unsigned>((elements + 255) / 256), 1, 1},
                              {256, 1, 1},
                              0,
                              std::move(arguments)};
  ExecutionPlan plan;
  plan.addKernel(std::move(invocation));
  NvidiaGraphExecutable graph = runtime.instantiate(plan, artifacts);
  graph.launch(stream);
  stream.synchronize();

  std::vector<float> output(elements);
  buffer.copyToHost(output.data(), buffer.size());
  for (std::int64_t index = 0; index < elements; ++index)
    if (std::abs(output[index] - (input[index] + 1.0f)) > 1e-6f)
      throw std::runtime_error("compiler-generated CUBIN result mismatch at index " +
                               std::to_string(index));
  std::cout << "validated compiler-generated CUBIN over " << elements << " elements\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ckl-generated-cubin-validation: " << error.what() << '\n';
  return 1;
}

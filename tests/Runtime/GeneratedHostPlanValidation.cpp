#include "ckl/Runtime/NvidiaRuntime.h"

#include "generated-host-plan.inc"

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace mlir::ckl::runtime;

namespace {

double checksum(const NvidiaBuffer &buffer, std::size_t elements) {
  std::vector<float> host(elements);
  buffer.copyToHost(host.data(), host.size() * sizeof(float));
  return std::accumulate(host.begin(), host.end(), 0.0);
}

bool close(double lhs, double rhs) {
  return std::abs(lhs - rhs) <= std::max(1e-3, std::abs(lhs) * 1e-5);
}

} // namespace

int main(int argc, char **argv) try {
  if (argc != 2)
    throw std::invalid_argument("usage: ckl-generated-host-plan-validation <cubin>");
  constexpr std::size_t elements = 257;
  NvidiaRuntime runtime;
  NvidiaModule module = runtime.loadCubinFile(argv[1]);
  NvidiaStream stream = runtime.createStream();
  NvidiaBuffer inputA = runtime.allocate(elements * sizeof(float));
  NvidiaBuffer inputB = runtime.allocate(elements * sizeof(float));
  NvidiaBuffer output = runtime.allocate(elements * sizeof(float));
  std::vector<float> hostA(elements), hostB(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    hostA[index] = 0.25f + static_cast<float>(index % 17) / 17.0f;
    hostB[index] = 0.5f + static_cast<float>(index % 31) / 31.0f;
  }
  inputA.copyFromHost(hostA.data(), inputA.size());
  inputB.copyFromHost(hostB.data(), inputB.size());

  ckl_generated::host_graphPlan generated =
      ckl_generated::build_host_graph(runtime, module, inputA, inputB, output);
  if (generated.plan.nodes().size() != 5 || generated.heaps.empty())
    throw std::runtime_error("generated host plan has an unexpected shape");

  output.fillZero();
  runtime.launchOrdinary(generated.plan, stream);
  stream.synchronize();
  double ordinary = checksum(output, elements);
  output.fillZero();
  NvidiaGraphExecutable graph = runtime.instantiate(generated.plan);
  graph.launch(stream);
  stream.synchronize();
  double graphed = checksum(output, elements);
  if (!close(ordinary, graphed))
    throw std::runtime_error("generated ordinary and graph plans disagree");
  std::cout << "nodes=5,heaps=" << generated.heaps.size() << ",checksum=" << graphed
            << ",correct=true\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ckl-generated-host-plan-validation: " << error.what() << '\n';
  return 1;
}

#include "ckl/Runtime/ExecutionPlan.h"

#include <array>
#include <iostream>
#include <stdexcept>

using namespace mlir::ckl::runtime;

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Callable> void requireInvalid(Callable callable, const char *message) {
  try {
    callable();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(message);
}

KernelInvocation invocation(const char *entry) {
  KernelArguments arguments;
  arguments.add(std::uint64_t{0x1000}).add(std::int32_t{16});
  return {"producer-object", entry, "cuda.direct", 0, {1, 1, 1}, {32, 1, 1}, 0,
          std::move(arguments)};
}

} // namespace

int main() try {
  std::vector<std::byte> object{std::byte{0x7f}, std::byte{0x01}};
  ArtifactRegistry artifacts;
  artifacts.add(KernelArtifact("producer-object", "cuda.cubin", "sm_120", object));
  require(artifacts.size() == 1 && artifacts.lookup("producer-object").size() == object.size(),
          "artifact registry did not preserve the producer object");
  requireInvalid(
      [&] { artifacts.add(KernelArtifact("producer-object", "cuda.cubin", "sm_120", object)); },
      "duplicate artifact identity was accepted");

  ExecutionPlan plan;
  auto first = plan.addKernel(invocation("first"));
  plan.addKernel(invocation("second"), {first});
  ExecutionPlan repeated = plan.repeat(3);
  require(repeated.nodes().size() == 6, "execution plan repeat produced the wrong node count");
  require(repeated.nodes()[2].dependencies.size() == 1 &&
              repeated.nodes()[2].dependencies[0] == 1,
          "execution plan repeat did not connect iteration boundaries");
  requireInvalid([&] { plan.addKernel(invocation("bad"), {99}); },
                 "forward dependency was accepted");

  std::cout << "artifact_registry=true,physical_plan=true,repeat=true\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ckl-execution-plan-test: " << error.what() << '\n';
  return 1;
}

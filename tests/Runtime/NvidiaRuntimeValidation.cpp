#include "ckl/Runtime/NvidiaRuntime.h"

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace mlir::ckl::runtime;

namespace {

void check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS)
    return;
  const char *description = nullptr;
  cuGetErrorString(result, &description);
  throw std::runtime_error(std::string(operation) + ": " +
                           (description ? description : "unknown CUDA error"));
}

struct Options {
  std::string cubin;
  int elements = 2048;
  int cycles = 200;
  int warmup = 20;
  int repeats = 9;
};

int parsePositive(const char *value, const char *option) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0 || parsed > (1L << 30))
    throw std::invalid_argument(std::string(option) + " requires a positive integer");
  return static_cast<int>(parsed);
}

Options parseOptions(int argc, char **argv) {
  if (argc < 2)
    throw std::invalid_argument("usage: ckl-nvidia-runtime-validation <cubin> [options]");
  Options options;
  options.cubin = argv[1];
  for (int index = 2; index < argc; index += 2) {
    if (index + 1 == argc)
      throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    std::string_view option(argv[index]);
    if (option == "--elements")
      options.elements = parsePositive(argv[index + 1], argv[index]);
    else if (option == "--cycles")
      options.cycles = parsePositive(argv[index + 1], argv[index]);
    else if (option == "--warmup")
      options.warmup = parsePositive(argv[index + 1], argv[index]);
    else if (option == "--repeats")
      options.repeats = parsePositive(argv[index + 1], argv[index]);
    else
      throw std::invalid_argument(std::string("unknown option ") + argv[index]);
  }
  return options;
}

KernelLaunch fieldLaunch(NvidiaFunction function, std::uint64_t input, std::uint64_t output,
                         int elements, float bias) {
  unsigned blocks = (static_cast<unsigned>(elements) + 255) / 256;
  KernelArguments arguments;
  arguments.add(input).add(output).add(elements).add(bias);
  return {std::move(function), {blocks, 1, 1}, {256, 1, 1}, 0, std::move(arguments)};
}

KernelLaunch joinLaunch(NvidiaFunction function, std::uint64_t first, std::uint64_t second,
                        std::uint64_t output, int elements) {
  unsigned blocks = (static_cast<unsigned>(elements) + 255) / 256;
  KernelArguments arguments;
  arguments.add(first).add(second).add(output).add(elements);
  return {std::move(function), {blocks, 1, 1}, {256, 1, 1}, 0, std::move(arguments)};
}

struct Buffers {
  Buffers(NvidiaRuntime &runtime, int elements)
      : a0(runtime.allocate(elements * sizeof(float))),
        a1(runtime.allocate(elements * sizeof(float))),
        b0(runtime.allocate(elements * sizeof(float))),
        b1(runtime.allocate(elements * sizeof(float))),
        output(runtime.allocate(elements * sizeof(float))), initialA(elements), initialB(elements) {
    for (int index = 0; index < elements; ++index) {
      initialA[index] = 0.25f + static_cast<float>(index % 251) / 251.0f;
      initialB[index] = 0.5f + static_cast<float>(index % 127) / 127.0f;
    }
  }

  void reset() {
    a0.copyFromHost(initialA.data(), a0.size());
    b0.copyFromHost(initialB.data(), b0.size());
    a1.fillZero();
    b1.fillZero();
    output.fillZero();
  }

  double checksum() const {
    std::vector<float> host(initialA.size());
    output.copyToHost(host.data(), output.size());
    return std::accumulate(host.begin(), host.end(), 0.0);
  }

  NvidiaBuffer a0;
  NvidiaBuffer a1;
  NvidiaBuffer b0;
  NvidiaBuffer b1;
  NvidiaBuffer output;
  std::vector<float> initialA;
  std::vector<float> initialB;
};

NvidiaPlan makePlan(const NvidiaModule &module, const Buffers &buffers, int elements,
                    float firstBias = 0.00001f) {
  NvidiaFunction field = module.function("field");
  NvidiaFunction join = module.function("join");
  NvidiaPlan plan;
  auto a0 = plan.addKernel(
      fieldLaunch(field, buffers.a0.address(), buffers.a1.address(), elements, firstBias));
  auto b0 = plan.addKernel(
      fieldLaunch(field, buffers.b0.address(), buffers.b1.address(), elements, 0.00002f));
  auto join0 = plan.addKernel(joinLaunch(join, buffers.a1.address(), buffers.b1.address(),
                                         buffers.output.address(), elements),
                              {a0, b0});
  auto a1 = plan.addKernel(
      fieldLaunch(field, buffers.a1.address(), buffers.a0.address(), elements, 0.00001f), {a0});
  auto b1 = plan.addKernel(
      fieldLaunch(field, buffers.b1.address(), buffers.b0.address(), elements, 0.00002f), {b0});
  plan.addKernel(joinLaunch(join, buffers.a0.address(), buffers.b0.address(),
                            buffers.output.address(), elements),
                 {join0, a1, b1});
  return plan;
}

class ManualGraph {
public:
  ManualGraph(const std::string &cubin, const Buffers &buffers, int elements) {
    check(cuModuleLoad(&module_, cubin.c_str()), "cuModuleLoad");
    CUfunction field{};
    CUfunction join{};
    check(cuModuleGetFunction(&field, module_, "field"), "cuModuleGetFunction(field)");
    check(cuModuleGetFunction(&join, module_, "join"), "cuModuleGetFunction(join)");
    check(cuGraphCreate(&graph_, 0), "cuGraphCreate");
    auto a0 = addField(field, {}, buffers.a0.address(), buffers.a1.address(), elements, 0.00001f);
    auto b0 = addField(field, {}, buffers.b0.address(), buffers.b1.address(), elements, 0.00002f);
    auto join0 = addJoin(join, {a0, b0}, buffers.a1.address(), buffers.b1.address(),
                         buffers.output.address(), elements);
    auto a1 = addField(field, {a0}, buffers.a1.address(), buffers.a0.address(), elements, 0.00001f);
    auto b1 = addField(field, {b0}, buffers.b1.address(), buffers.b0.address(), elements, 0.00002f);
    addJoin(join, {join0, a1, b1}, buffers.a0.address(), buffers.b0.address(),
            buffers.output.address(), elements);
    check(cuGraphInstantiate(&executable_, graph_, 0), "cuGraphInstantiate");
    check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
  }

  ~ManualGraph() {
    if (stream_)
      cuStreamDestroy(stream_);
    if (executable_)
      cuGraphExecDestroy(executable_);
    if (graph_)
      cuGraphDestroy(graph_);
    if (module_)
      cuModuleUnload(module_);
  }

  void launch() { check(cuGraphLaunch(executable_, stream_), "cuGraphLaunch(manual)"); }
  void synchronize() { check(cuStreamSynchronize(stream_), "cuStreamSynchronize(manual)"); }

private:
  CUgraphNode addField(CUfunction function, std::vector<CUgraphNode> dependencies,
                       std::uint64_t input, std::uint64_t output, int elements, float bias) {
    void *arguments[] = {&input, &output, &elements, &bias};
    CUDA_KERNEL_NODE_PARAMS parameters{};
    parameters.func = function;
    parameters.gridDimX = (static_cast<unsigned>(elements) + 255) / 256;
    parameters.gridDimY = parameters.gridDimZ = 1;
    parameters.blockDimX = 256;
    parameters.blockDimY = parameters.blockDimZ = 1;
    parameters.kernelParams = arguments;
    CUgraphNode node{};
    check(cuGraphAddKernelNode(&node, graph_, dependencies.data(), dependencies.size(),
                               &parameters),
          "cuGraphAddKernelNode(field)");
    return node;
  }

  CUgraphNode addJoin(CUfunction function, std::vector<CUgraphNode> dependencies,
                      std::uint64_t first, std::uint64_t second, std::uint64_t output,
                      int elements) {
    void *arguments[] = {&first, &second, &output, &elements};
    CUDA_KERNEL_NODE_PARAMS parameters{};
    parameters.func = function;
    parameters.gridDimX = (static_cast<unsigned>(elements) + 255) / 256;
    parameters.gridDimY = parameters.gridDimZ = 1;
    parameters.blockDimX = 256;
    parameters.blockDimY = parameters.blockDimZ = 1;
    parameters.kernelParams = arguments;
    CUgraphNode node{};
    check(cuGraphAddKernelNode(&node, graph_, dependencies.data(), dependencies.size(),
                               &parameters),
          "cuGraphAddKernelNode(join)");
    return node;
  }

  CUmodule module_{};
  CUgraph graph_{};
  CUgraphExec executable_{};
  CUstream stream_{};
};

struct Result {
  double milliseconds;
  double checksum;
};

template <typename Execute, typename Synchronize>
Result benchmark(Buffers &buffers, int cycles, int warmup, int repeats, Execute execute,
                 Synchronize synchronize) {
  buffers.reset();
  for (int index = 0; index < warmup; ++index)
    execute();
  synchronize();

  std::vector<double> samples;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    buffers.reset();
    auto begin = std::chrono::steady_clock::now();
    for (int cycle = 0; cycle < cycles; ++cycle)
      execute();
    synchronize();
    auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  std::sort(samples.begin(), samples.end());
  return {samples[samples.size() / 2], buffers.checksum()};
}

bool close(double lhs, double rhs) {
  return std::isfinite(lhs) && std::isfinite(rhs) &&
         std::abs(lhs - rhs) <= std::max(1e-3, std::abs(lhs) * 1e-5);
}

} // namespace

int main(int argc, char **argv) try {
  Options options = parseOptions(argc, argv);
  NvidiaRuntime runtime;
  NvidiaStream stream = runtime.createStream();
  NvidiaModule module = runtime.loadCubinFile(options.cubin);
  Buffers buffers(runtime, options.elements);
  NvidiaPlan plan = makePlan(module, buffers, options.elements);

  auto cached = runtime.getOrCreateGraph("multi-field-v1", plan);
  auto cachedAgain = runtime.getOrCreateGraph("multi-field-v1", plan);
  if (cached != cachedAgain || runtime.cachedGraphCount() != 1 || cached->nodeCount() != 6)
    throw std::runtime_error("graph cache did not reuse the executable topology");

  Result ordinary = benchmark(
      buffers, options.cycles, options.warmup, options.repeats,
      [&] { runtime.launchOrdinary(plan, stream); }, [&] { stream.synchronize(); });
  Result graph = benchmark(buffers, options.cycles, options.warmup, options.repeats,
                           [&] { cached->launch(stream); }, [&] { stream.synchronize(); });
  ManualGraph manual(options.cubin, buffers, options.elements);
  Result manualResult = benchmark(buffers, options.cycles, options.warmup, options.repeats,
                                  [&] { manual.launch(); }, [&] { manual.synchronize(); });

  if (!close(ordinary.checksum, graph.checksum) || !close(ordinary.checksum, manualResult.checksum))
    throw std::runtime_error("ordinary, CKL graph, and manual graph checksums differ");

  NvidiaPlan updated = makePlan(module, buffers, options.elements, 0.00003f);
  NvidiaGraphExecutable updateExecutable = runtime.instantiate(plan);
  updateExecutable.updateKernel(0, updated.nodes()[0].launch);
  Result updatedOrdinary = benchmark(
      buffers, 1, 1, 1, [&] { runtime.launchOrdinary(updated, stream); },
      [&] { stream.synchronize(); });
  Result updatedGraph = benchmark(buffers, 1, 1, 1, [&] { updateExecutable.launch(stream); },
                                  [&] { stream.synchronize(); });
  if (!close(updatedOrdinary.checksum, updatedGraph.checksum))
    throw std::runtime_error("updated graph parameters produced an incorrect result");

  double parity = graph.milliseconds / manualResult.milliseconds;
  if (options.repeats > 1 && (parity < 0.95 || parity > 1.05))
    throw std::runtime_error("CKL graph overhead is not within 5% of the manual graph baseline");

  std::cout << "device,mode,elements,cycles,median_ms,checksum,correct\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << runtime.deviceName() << ",ordinary," << options.elements << ',' << options.cycles
            << ',' << ordinary.milliseconds << ',' << ordinary.checksum << ",true\n";
  std::cout << runtime.deviceName() << ",ckl_graph," << options.elements << ',' << options.cycles
            << ',' << graph.milliseconds << ',' << graph.checksum << ",true\n";
  std::cout << runtime.deviceName() << ",manual_graph," << options.elements << ',' << options.cycles
            << ',' << manualResult.milliseconds << ',' << manualResult.checksum << ",true\n";
  std::cout << "cache_entries=" << runtime.cachedGraphCount() << ",update_correct=true,parity="
            << parity << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ckl-nvidia-runtime-validation: " << error.what() << '\n';
  return 1;
}

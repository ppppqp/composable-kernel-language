#include "ckl/Core/MemoryPlanner.h"
#include "ckl/Runtime/NvidiaRuntime.h"

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

using namespace mlir::ckl::planning;
using namespace mlir::ckl::runtime;

namespace {

struct Options {
  std::string cubin;
  int elements = 2048;
  int cycles = 192;
  int warmup = 10;
  int repeats = 7;
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
    throw std::invalid_argument("usage: ckl-milestone4-validation <cubin> [options]");
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

KernelLaunch fieldLaunch(const NvidiaFunction &function, std::uint64_t input,
                         std::uint64_t output, int elements, float bias) {
  KernelArguments arguments;
  arguments.add(input).add(output).add(elements).add(bias);
  unsigned blocks = (static_cast<unsigned>(elements) + 255) / 256;
  return {function, {blocks, 1, 1}, {256, 1, 1}, 0, std::move(arguments)};
}

KernelLaunch joinLaunch(const NvidiaFunction &function, std::uint64_t first,
                        std::uint64_t second, std::uint64_t output, int elements) {
  KernelArguments arguments;
  arguments.add(first).add(second).add(output).add(elements);
  unsigned blocks = (static_cast<unsigned>(elements) + 255) / 256;
  return {function, {blocks, 1, 1}, {256, 1, 1}, 0, std::move(arguments)};
}

MemoryPlan makeMemoryPlan(std::size_t bytes) {
  // The second temporary intentionally follows the first. Reuse introduces no new edge.
  std::vector<GraphEdge> edges = {{0, 1}, {1, 2}, {1, 4}, {2, 3}, {3, 4}};
  std::vector<Resource> resources = {
      {"first-temporary", bytes, 256, "global", 0, ResourceKind::Temporary, {0, 1}},
      {"second-temporary", bytes, 256, "global", 0, ResourceKind::Temporary, {2, 3}},
  };
  return MemoryPlanner().plan(5, edges, resources);
}

struct BufferSet {
  BufferSet(NvidiaRuntime &runtime, int elements, const MemoryPlan *memoryPlan)
      : inputA(runtime.allocate(elements * sizeof(float))),
        inputB(runtime.allocate(elements * sizeof(float))),
        outputA(runtime.allocate(elements * sizeof(float))),
        outputB(runtime.allocate(elements * sizeof(float))),
        output(runtime.allocate(elements * sizeof(float))), initialA(elements), initialB(elements) {
    std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(float);
    if (memoryPlan) {
      if (memoryPlan->heaps.size() != 1)
        throw std::runtime_error("Milestone 4 validation expects one device heap");
      arena = runtime.allocate(memoryPlan->heaps[0].bytes);
      const Assignment *first = memoryPlan->find(0);
      const Assignment *second = memoryPlan->find(1);
      if (!first || !second)
        throw std::runtime_error("temporary memory plan is incomplete");
      temporaryA = arena.slice(first->offset, first->bytes);
      temporaryB = arena.slice(second->offset, second->bytes);
    } else {
      temporaryA = runtime.allocate(bytes);
      temporaryB = runtime.allocate(bytes);
    }
    for (int index = 0; index < elements; ++index) {
      initialA[index] = 0.25f + static_cast<float>(index % 251) / 251.0f;
      initialB[index] = 0.5f + static_cast<float>(index % 127) / 127.0f;
    }
  }

  void reset() {
    inputA.copyFromHost(initialA.data(), inputA.size());
    inputB.copyFromHost(initialB.data(), inputB.size());
    outputA.fillZero();
    outputB.fillZero();
    output.fillZero();
  }

  double checksum() const {
    std::vector<float> host(initialA.size());
    output.copyToHost(host.data(), output.size());
    return std::accumulate(host.begin(), host.end(), 0.0);
  }

  NvidiaBuffer inputA, inputB, temporaryA, temporaryB, outputA, outputB, output, arena;
  std::vector<float> initialA, initialB;
};

NvidiaPlan makePlan(const NvidiaModule &module, const BufferSet &buffers, int elements) {
  NvidiaFunction field = module.function("field");
  NvidiaFunction join = module.function("join");
  NvidiaPlan plan;
  auto first = plan.addKernel(fieldLaunch(field, buffers.inputA.address(),
                                          buffers.temporaryA.address(), elements, 0.00001f));
  auto firstOutput = plan.addKernel(fieldLaunch(field, buffers.temporaryA.address(),
                                                buffers.outputA.address(), elements, 0.00001f),
                                    {first});
  auto second = plan.addKernel(fieldLaunch(field, buffers.inputB.address(),
                                           buffers.temporaryB.address(), elements, 0.00002f),
                               {firstOutput});
  auto secondOutput = plan.addKernel(fieldLaunch(field, buffers.temporaryB.address(),
                                                 buffers.outputB.address(), elements, 0.00002f),
                                     {second});
  plan.addKernel(joinLaunch(join, buffers.outputA.address(), buffers.outputB.address(),
                            buffers.output.address(), elements),
                 {firstOutput, secondOutput});
  return plan;
}

bool close(double lhs, double rhs) {
  return std::isfinite(lhs) && std::isfinite(rhs) &&
         std::abs(lhs - rhs) <= std::max(1e-3, std::abs(lhs) * 1e-5);
}

struct Candidate {
  int batch = 1;
  int inFlight = 1;
  std::size_t nodes = 0;
  std::size_t temporaryBefore = 0;
  std::size_t temporaryAfter = 0;
  double setupMilliseconds = 0;
  double milliseconds = 0;
  double checksum = 0;
  bool correct = false;
};

Candidate measureCandidate(NvidiaRuntime &runtime, const NvidiaModule &module,
                           const MemoryPlan &memoryPlan, const Options &options, int batch,
                           int inFlight, double reference) {
  std::vector<BufferSet> buffers;
  std::vector<NvidiaStream> streams;
  buffers.reserve(inFlight);
  streams.reserve(inFlight);
  for (int index = 0; index < inFlight; ++index) {
    buffers.emplace_back(runtime, options.elements, &memoryPlan);
    streams.push_back(runtime.createStream());
  }

  auto setupBegin = std::chrono::steady_clock::now();
  std::vector<NvidiaGraphExecutable> executables;
  executables.reserve(inFlight);
  std::size_t nodes = 0;
  for (int index = 0; index < inFlight; ++index) {
    NvidiaPlan repeated = makePlan(module, buffers[index], options.elements).repeat(batch);
    nodes = repeated.nodes().size();
    executables.push_back(runtime.instantiate(repeated));
  }
  auto setupEnd = std::chrono::steady_clock::now();

  auto execute = [&](int logicalIterations) {
    int submissions = logicalIterations / batch;
    for (int submission = 0; submission < submissions; ++submission) {
      int instance = submission % inFlight;
      executables[instance].launch(streams[instance]);
    }
    for (NvidiaStream &stream : streams)
      stream.synchronize();
  };

  for (BufferSet &set : buffers)
    set.reset();
  execute(options.warmup * batch);

  std::vector<double> samples;
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    for (BufferSet &set : buffers)
      set.reset();
    auto begin = std::chrono::steady_clock::now();
    execute(options.cycles);
    auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  std::sort(samples.begin(), samples.end());

  double checksum = buffers.front().checksum();
  bool correct = close(checksum, reference);
  for (const BufferSet &set : buffers)
    correct = correct && close(set.checksum(), reference);
  std::size_t bytes = static_cast<std::size_t>(options.elements) * sizeof(float);
  return {batch,
          inFlight,
          nodes,
          2 * bytes * static_cast<std::size_t>(inFlight),
          memoryPlan.plannedBytes * static_cast<std::size_t>(inFlight),
          std::chrono::duration<double, std::milli>(setupEnd - setupBegin).count(),
          samples[samples.size() / 2],
          checksum,
          correct};
}

} // namespace

int main(int argc, char **argv) try {
  Options options = parseOptions(argc, argv);
  NvidiaRuntime runtime;
  NvidiaModule module = runtime.loadCubinFile(options.cubin);
  std::size_t bytes = static_cast<std::size_t>(options.elements) * sizeof(float);
  MemoryPlan memoryPlan = makeMemoryPlan(bytes);
  if (memoryPlan.baselineBytes != 2 * bytes || memoryPlan.plannedBytes != bytes)
    throw std::runtime_error("temporary planner did not achieve the expected 50% reduction");
  if (memoryPlan.find(0)->offset != memoryPlan.find(1)->offset)
    throw std::runtime_error("ordered temporaries did not share an arena range");

  BufferSet baselineBuffers(runtime, options.elements, nullptr);
  baselineBuffers.reset();
  NvidiaStream baselineStream = runtime.createStream();
  NvidiaPlan baselinePlan = makePlan(module, baselineBuffers, options.elements);
  runtime.launchOrdinary(baselinePlan, baselineStream);
  baselineStream.synchronize();
  double reference = baselineBuffers.checksum();

  std::vector<Candidate> candidates;
  for (int batch : {1, 2, 4, 8}) {
    if (batch > options.cycles || options.cycles % batch)
      continue;
    for (int inFlight : {1, 2, 4}) {
      if (inFlight > options.cycles / batch)
        continue;
      candidates.push_back(measureCandidate(runtime, module, memoryPlan, options, batch,
                                            inFlight, reference));
    }
  }
  if (candidates.empty())
    throw std::runtime_error("no valid tuning candidates");
  if (std::any_of(candidates.begin(), candidates.end(),
                  [](const Candidate &candidate) { return !candidate.correct; }))
    throw std::runtime_error("a measured Milestone 4 candidate produced an incorrect result");
  auto selected = std::min_element(candidates.begin(), candidates.end(),
                                   [](const Candidate &lhs, const Candidate &rhs) {
                                     return lhs.milliseconds < rhs.milliseconds;
                                   });

  std::cout << "device,batch,in_flight,nodes,temp_before_bytes,temp_after_bytes,setup_ms,"
               "median_ms,logical_iterations,checksum,correct,selected\n";
  std::cout << std::fixed << std::setprecision(6);
  for (auto candidate = candidates.begin(); candidate != candidates.end(); ++candidate)
    std::cout << runtime.deviceName() << ',' << candidate->batch << ',' << candidate->inFlight
              << ',' << candidate->nodes << ',' << candidate->temporaryBefore << ','
              << candidate->temporaryAfter << ',' << candidate->setupMilliseconds << ','
              << candidate->milliseconds << ',' << options.cycles << ',' << candidate->checksum
              << ",true," << (candidate == selected ? "true" : "false") << '\n';
  std::cout << "# decision,selected_batch=" << selected->batch
            << ",selected_in_flight=" << selected->inFlight
            << ",reason=lowest measured median,planner_reason="
            << memoryPlan.decisions[1].reason << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ckl-milestone4-validation: " << error.what() << '\n';
  return 1;
}

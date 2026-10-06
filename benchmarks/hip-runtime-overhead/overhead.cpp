#include "ckl/Runtime/HipRuntime.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace mlir::ckl::runtime;

namespace {

void check(hipError_t result, const char *operation) {
  if (result != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
}

#define HIP_CHECK(expression) check((expression), #expression)

struct Options {
  std::filesystem::path hsaco;
  int device = 0;
  int elements = 256;
  int warmup = 100;
  int iterations = 1000;
  int samples = 11;
  std::vector<int> nodeCounts{2, 4, 8, 16};
};

int parsePositive(const char *value, const char *option) {
  char *end = nullptr;
  long result = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || result <= 0 || result > (1L << 30))
    throw std::invalid_argument(std::string(option) + " requires a positive integer");
  return static_cast<int>(result);
}

int parseNonNegative(const char *value, const char *option) {
  char *end = nullptr;
  long result = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || result < 0 || result > (1L << 30))
    throw std::invalid_argument(std::string(option) + " requires a non-negative integer");
  return static_cast<int>(result);
}

std::vector<int> parseNodes(std::string_view value) {
  std::vector<int> result;
  std::size_t start = 0;
  while (start < value.size()) {
    std::size_t comma = value.find(',', start);
    std::string token(value.substr(start, comma == std::string_view::npos ? value.size() - start
                                                                         : comma - start));
    result.push_back(parsePositive(token.c_str(), "--nodes"));
    if (comma == std::string_view::npos)
      break;
    start = comma + 1;
  }
  if (result.empty())
    throw std::invalid_argument("--nodes requires at least one node count");
  return result;
}

Options parseOptions(int argc, char **argv) {
  if (argc < 2)
    throw std::invalid_argument(
        "usage: ckl-hip-runtime-overhead <hsaco> [--device N] [--elements N] "
        "[--warmup N] [--iterations N] [--samples N] [--nodes 2,4,8,16]");
  Options options;
  options.hsaco = argv[1];
  for (int index = 2; index < argc; index += 2) {
    if (index + 1 == argc)
      throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    std::string_view option(argv[index]);
    if (option == "--device") {
      options.device = parseNonNegative(argv[index + 1], argv[index]);
    } else if (option == "--elements") {
      options.elements = parsePositive(argv[index + 1], argv[index]);
    } else if (option == "--warmup") {
      options.warmup = parsePositive(argv[index + 1], argv[index]);
    } else if (option == "--iterations") {
      options.iterations = parsePositive(argv[index + 1], argv[index]);
    } else if (option == "--samples") {
      options.samples = parsePositive(argv[index + 1], argv[index]);
    } else if (option == "--nodes") {
      options.nodeCounts = parseNodes(argv[index + 1]);
    } else {
      throw std::invalid_argument(std::string("unknown option ") + argv[index]);
    }
  }
  return options;
}

std::vector<char> readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open HSACO: " + path.string());
  std::streamsize size = input.tellg();
  if (size <= 0)
    throw std::runtime_error("HSACO is empty: " + path.string());
  input.seekg(0);
  std::vector<char> bytes(static_cast<std::size_t>(size));
  if (!input.read(bytes.data(), size))
    throw std::runtime_error("failed to read HSACO: " + path.string());
  return bytes;
}

template <typename Handle, hipError_t (*Destroy)(Handle)> class HipHandle {
public:
  HipHandle() = default;
  explicit HipHandle(Handle handle) : handle_(handle) {}
  ~HipHandle() {
    if (handle_)
      (void)Destroy(handle_);
  }
  HipHandle(const HipHandle &) = delete;
  HipHandle &operator=(const HipHandle &) = delete;
  HipHandle(HipHandle &&other) noexcept : handle_(other.handle_) { other.handle_ = {}; }
  HipHandle &operator=(HipHandle &&other) noexcept {
    if (this != &other) {
      if (handle_)
        (void)Destroy(handle_);
      handle_ = other.handle_;
      other.handle_ = {};
    }
    return *this;
  }
  Handle get() const { return handle_; }

private:
  Handle handle_{};
};

using Module = HipHandle<hipModule_t, hipModuleUnload>;
using Stream = HipHandle<hipStream_t, hipStreamDestroy>;
using Event = HipHandle<hipEvent_t, hipEventDestroy>;
using Graph = HipHandle<hipGraph_t, hipGraphDestroy>;
using GraphExec = HipHandle<hipGraphExec_t, hipGraphExecDestroy>;

struct DeviceMemory {
  explicit DeviceMemory(std::size_t bytes) {
    HIP_CHECK(hipMalloc(&pointer, bytes));
  }
  ~DeviceMemory() {
    if (pointer)
      (void)hipFree(pointer);
  }
  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;
  void *pointer = nullptr;
};

struct RawLaunch {
  hipFunction_t function{};
  float *values = nullptr;
  int elements = 0;
  float bias = 0.0f;

  void enqueue(hipStream_t stream) {
    void *arguments[] = {&values, &elements, &bias};
    HIP_CHECK(hipModuleLaunchKernel(function, (static_cast<unsigned>(elements) + 255) / 256, 1, 1,
                                    256, 1, 1, 0, stream, arguments, nullptr));
  }

  hipKernelNodeParams graphParameters() {
    argumentPointers[0] = &values;
    argumentPointers[1] = &elements;
    argumentPointers[2] = &bias;
    hipKernelNodeParams parameters{};
    parameters.func = reinterpret_cast<void *>(function);
    parameters.gridDim = dim3((static_cast<unsigned>(elements) + 255) / 256, 1, 1);
    parameters.blockDim = dim3(256, 1, 1);
    parameters.kernelParams = argumentPointers;
    return parameters;
  }

  void *argumentPointers[3]{};
};

struct ManualGraph {
  Graph graph;
  GraphExec executable;
};

ManualGraph makeManualGraph(RawLaunch &launch, int nodes) {
  hipGraph_t rawGraph{};
  HIP_CHECK(hipGraphCreate(&rawGraph, 0));
  ManualGraph result{Graph(rawGraph), GraphExec()};
  hipGraphNode_t predecessor{};
  for (int index = 0; index < nodes; ++index) {
    hipKernelNodeParams parameters = launch.graphParameters();
    hipGraphNode_t node{};
    const hipGraphNode_t *dependencies = index ? &predecessor : nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&node, result.graph.get(), dependencies, index ? 1 : 0,
                                    &parameters));
    predecessor = node;
  }
  hipGraphExec_t rawExecutable{};
  HIP_CHECK(hipGraphInstantiate(&rawExecutable, result.graph.get(), nullptr, nullptr, 0));
  result.executable = GraphExec(rawExecutable);
  return result;
}

KernelInvocation invocation(std::uint64_t address, int elements, float bias, int device) {
  KernelArguments arguments;
  arguments.add(address).add(elements).add(bias);
  return {"stage0.add_bias", "add_bias", "rocm.bare_ptr", device,
          {(static_cast<unsigned>(elements) + 255) / 256, 1, 1}, {256, 1, 1}, 0,
          std::move(arguments)};
}

ExecutionPlan makePlan(std::uint64_t address, int elements, float bias, int nodes, int device) {
  ExecutionPlan plan;
  ExecutionPlan::NodeId predecessor = 0;
  for (int index = 0; index < nodes; ++index) {
    std::vector<ExecutionPlan::NodeId> dependencies;
    if (index)
      dependencies.push_back(predecessor);
    predecessor =
        plan.addKernel(invocation(address, elements, bias, device), std::move(dependencies));
  }
  return plan;
}

template <typename Enqueue>
double gpuTimeUs(hipStream_t stream, int iterations, Enqueue &&enqueue) {
  hipEvent_t rawStart{}, rawEnd{};
  HIP_CHECK(hipEventCreate(&rawStart));
  Event start(rawStart);
  HIP_CHECK(hipEventCreate(&rawEnd));
  Event end(rawEnd);
  HIP_CHECK(hipEventRecord(start.get(), stream));
  for (int iteration = 0; iteration < iterations; ++iteration)
    enqueue();
  HIP_CHECK(hipEventRecord(end.get(), stream));
  HIP_CHECK(hipEventSynchronize(end.get()));
  float milliseconds = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&milliseconds, start.get(), end.get()));
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

template <typename Enqueue>
double cpuTimeUs(hipStream_t stream, int iterations, Enqueue &&enqueue) {
  HIP_CHECK(hipStreamSynchronize(stream));
  auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration)
    enqueue();
  auto end = std::chrono::steady_clock::now();
  HIP_CHECK(hipStreamSynchronize(stream));
  return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

void reset(void *pointer, std::size_t bytes, hipStream_t stream) {
  HIP_CHECK(hipMemsetAsync(pointer, 0, bytes, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}

template <typename Enqueue>
void validate(const char *mode, Enqueue &&enqueue, void *pointer, int elements, int nodes,
              hipStream_t stream) {
  reset(pointer, static_cast<std::size_t>(elements) * sizeof(float), stream);
  enqueue();
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<float> values(elements);
  HIP_CHECK(hipMemcpy(values.data(), pointer, values.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (float value : values) {
    if (!std::isfinite(value) || std::abs(value - 1.0f) > 1e-5f)
      throw std::runtime_error(std::string(mode) + " produced an incorrect result for " +
                               std::to_string(nodes) + " nodes");
  }
}

void emit(const std::string &device, const std::string &architecture, const char *mode, int nodes,
          int elements, const char *metric, int sample, double valueUs) {
  std::cout << device << ',' << architecture << ',' << mode << ',' << nodes << ',' << elements
            << ',' << metric << ',' << sample << ',' << valueUs << ",true\n";
}

struct Mode {
  const char *name;
  std::function<void()> enqueue;
};

} // namespace

int main(int argc, char **argv) try {
  Options options = parseOptions(argc, argv);
  HIP_CHECK(hipSetDevice(options.device));

  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, options.device));
  std::string device = properties.name;
  std::string architecture = properties.gcnArchName;
  if (auto colon = architecture.find(':'); colon != std::string::npos)
    architecture.resize(colon);

  std::vector<char> hsaco = readFile(options.hsaco);
  auto setupStart = std::chrono::steady_clock::now();
  hipModule_t rawModule{};
  HIP_CHECK(hipModuleLoadData(&rawModule, hsaco.data()));
  Module module(rawModule);
  hipFunction_t function{};
  HIP_CHECK(hipModuleGetFunction(&function, module.get(), "add_bias"));
  auto setupEnd = std::chrono::steady_clock::now();

  hipStream_t rawStream{};
  HIP_CHECK(hipStreamCreateWithFlags(&rawStream, hipStreamNonBlocking));
  Stream stream(rawStream);
  std::size_t bytes = static_cast<std::size_t>(options.elements) * sizeof(float);
  DeviceMemory allocation(bytes);

  HipRuntime runtime(options.device);
  HipStream cklStream = runtime.importStream(reinterpret_cast<std::uintptr_t>(stream.get()));
  HipBuffer cklBuffer = runtime.importBuffer(reinterpret_cast<std::uint64_t>(allocation.pointer), bytes);
  ArtifactRegistry artifacts;
  artifacts.add(KernelArtifact("stage0.add_bias", "rocm.hsaco", architecture,
                               std::vector<std::byte>(
                                   reinterpret_cast<const std::byte *>(hsaco.data()),
                                   reinterpret_cast<const std::byte *>(hsaco.data() + hsaco.size()))));

  std::cout << "device,architecture,mode,nodes,elements,metric,sample,value_us,correct\n";
  emit(device, architecture, "direct", 0, options.elements, "module_load_us", 0,
       std::chrono::duration<double, std::micro>(setupEnd - setupStart).count());

  // Exclude the HIP runtime's process-wide first-graph initialization from either implementation's
  // graph construction measurement.
  {
    RawLaunch warmup{function, static_cast<float *>(allocation.pointer), options.elements, 1.0f};
    ManualGraph graphApiWarmup = makeManualGraph(warmup, 1);
  }

  for (int nodes : options.nodeCounts) {
    float bias = 1.0f / static_cast<float>(nodes);
    RawLaunch raw{function, static_cast<float *>(allocation.pointer), options.elements, bias};

    ExecutionPlan plan =
        makePlan(cklBuffer.address(), options.elements, bias, nodes, options.device);
    auto resolveStart = std::chrono::steady_clock::now();
    HipPlan resolved = runtime.resolve(plan, artifacts);
    auto resolveEnd = std::chrono::steady_clock::now();
    emit(device, architecture, "ckl_ordinary", nodes, options.elements, "resolve_us", 0,
         std::chrono::duration<double, std::micro>(resolveEnd - resolveStart).count());

    auto manualStart = std::chrono::steady_clock::now();
    ManualGraph manual = makeManualGraph(raw, nodes);
    auto manualEnd = std::chrono::steady_clock::now();
    emit(device, architecture, "manual_graph", nodes, options.elements, "instantiate_us", 0,
         std::chrono::duration<double, std::micro>(manualEnd - manualStart).count());

    auto cklGraphStart = std::chrono::steady_clock::now();
    HipGraphExecutable cklGraph = runtime.instantiate(resolved);
    auto cklGraphEnd = std::chrono::steady_clock::now();
    emit(device, architecture, "ckl_graph", nodes, options.elements, "instantiate_us", 0,
         std::chrono::duration<double, std::micro>(cklGraphEnd - cklGraphStart).count());

    std::vector<Mode> modes{
        {"direct",
         [&] {
           for (int node = 0; node < nodes; ++node)
             raw.enqueue(stream.get());
         }},
        {"ckl_ordinary", [&] { runtime.launchOrdinary(resolved, cklStream); }},
        {"manual_graph",
         [&] { HIP_CHECK(hipGraphLaunch(manual.executable.get(), stream.get())); }},
        {"ckl_graph", [&] { cklGraph.launch(cklStream); }},
    };

    for (Mode &mode : modes) {
      validate(mode.name, mode.enqueue, allocation.pointer, options.elements, nodes, stream.get());
      for (int iteration = 0; iteration < options.warmup; ++iteration)
        mode.enqueue();
      HIP_CHECK(hipStreamSynchronize(stream.get()));
    }

    // Rotate the first mode so temperature and clock drift do not consistently favor one path.
    for (int sample = 0; sample < options.samples; ++sample) {
      for (std::size_t offset = 0; offset < modes.size(); ++offset) {
        Mode &mode = modes[(static_cast<std::size_t>(sample) + offset) % modes.size()];
        reset(allocation.pointer, bytes, stream.get());
        emit(device, architecture, mode.name, nodes, options.elements, "gpu_replay_us", sample,
             gpuTimeUs(stream.get(), options.iterations, mode.enqueue));
        reset(allocation.pointer, bytes, stream.get());
        emit(device, architecture, mode.name, nodes, options.elements, "cpu_submit_us", sample,
             cpuTimeUs(stream.get(), options.iterations, mode.enqueue));
      }
    }
  }
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ckl-hip-runtime-overhead: " << error.what() << '\n';
  return 1;
}

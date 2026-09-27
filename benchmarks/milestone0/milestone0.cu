#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

#define CUDA_CHECK(expression) check((expression), #expression)

struct Options {
  int elements = 1 << 11;
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
  Options options;
  for (int index = 1; index < argc; index += 2) {
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

class Stream {
public:
  explicit Stream(unsigned flags = cudaStreamNonBlocking) { CUDA_CHECK(cudaStreamCreateWithFlags(&value_, flags)); }
  ~Stream() { cudaStreamDestroy(value_); }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  operator cudaStream_t() const { return value_; }

private:
  cudaStream_t value_{};
};

class Event {
public:
  explicit Event(unsigned flags = cudaEventDefault) { CUDA_CHECK(cudaEventCreateWithFlags(&value_, flags)); }
  ~Event() { cudaEventDestroy(value_); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  operator cudaEvent_t() const { return value_; }

private:
  cudaEvent_t value_{};
};

class GraphExec {
public:
  GraphExec() = default;
  explicit GraphExec(cudaGraph_t graph) {
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&value_, graph, 0));
  }
  ~GraphExec() {
    if (value_)
      cudaGraphExecDestroy(value_);
  }
  GraphExec(const GraphExec &) = delete;
  GraphExec &operator=(const GraphExec &) = delete;
  GraphExec(GraphExec &&other) noexcept : value_(other.value_) { other.value_ = nullptr; }
  GraphExec &operator=(GraphExec &&other) noexcept {
    std::swap(value_, other.value_);
    return *this;
  }
  operator cudaGraphExec_t() const { return value_; }

private:
  cudaGraphExec_t value_{};
};

class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    CUDA_CHECK(cudaMalloc(&value_, elements * sizeof(float)));
  }
  ~DeviceBuffer() { cudaFree(value_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  float *get() const { return value_; }
  std::size_t bytes() const { return elements_ * sizeof(float); }

private:
  float *value_{};
  std::size_t elements_{};
};

__global__ void stencilKernel(const float *input, float *output, int size, float bias) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= size)
    return;
  int left = index == 0 ? 0 : index - 1;
  int right = index + 1 == size ? size - 1 : index + 1;
  output[index] = 0.25f * input[left] + 0.5f * input[index] + 0.25f * input[right] + bias;
}

__global__ void fieldKernel(const float *input, float *output, int size, float bias) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= size)
    return;
  float value = input[index];
#pragma unroll 1
  for (int iteration = 0; iteration < 128; ++iteration)
    value = fmaf(value, 0.99991f, bias) + 0.00001f * __sinf(value);
  output[index] = value;
}

__global__ void transformKernel(const float *input, float *output, int size, float scale,
                                float bias) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < size) {
    float value = input[index];
    output[index] = scale * value + bias / (1.0f + value * value);
  }
}

__global__ void joinKernel(const float *first, const float *second, float *output, int size) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < size)
    output[index] = first[index] * 0.625f + second[index] * 0.375f;
}

struct Buffers {
  explicit Buffers(int elements)
      : a0(elements), a1(elements), b0(elements), b1(elements), scratch(elements), output(elements),
        initialA(elements), initialB(elements) {
    for (int index = 0; index < elements; ++index) {
      initialA[index] = 0.25f + static_cast<float>(index % 251) / 251.0f;
      initialB[index] = 0.5f + static_cast<float>(index % 127) / 127.0f;
    }
  }

  void reset() {
    CUDA_CHECK(cudaMemcpy(a0.get(), initialA.data(), a0.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b0.get(), initialB.data(), b0.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(a1.get(), 0, a1.bytes()));
    CUDA_CHECK(cudaMemset(b1.get(), 0, b1.bytes()));
    CUDA_CHECK(cudaMemset(scratch.get(), 0, scratch.bytes()));
    CUDA_CHECK(cudaMemset(output.get(), 0, output.bytes()));
    // The benchmark streams are non-blocking, so legacy-default-stream
    // ordering does not protect these initialization operations.
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  double checksum(int elements) const {
    std::vector<float> host(elements);
    CUDA_CHECK(cudaMemcpy(host.data(), output.get(), output.bytes(), cudaMemcpyDeviceToHost));
    return std::accumulate(host.begin(), host.end(), 0.0);
  }

  DeviceBuffer a0;
  DeviceBuffer a1;
  DeviceBuffer b0;
  DeviceBuffer b1;
  DeviceBuffer scratch;
  DeviceBuffer output;
  std::vector<float> initialA;
  std::vector<float> initialB;
};

enum class Workload { MultiField, LinearStencil };
enum class Mode { Sequential, Capture, Streams, Explicit };

const char *name(Workload workload) {
  return workload == Workload::MultiField ? "multi_field" : "linear_stencil";
}

const char *name(Mode mode) {
  switch (mode) {
  case Mode::Sequential:
    return "sequential";
  case Mode::Capture:
    return "capture";
  case Mode::Streams:
    return "streams";
  case Mode::Explicit:
    return "explicit";
  }
  return "unknown";
}

dim3 gridFor(int elements) {
  constexpr int threads = 256;
  return dim3((elements + threads - 1) / threads);
}

void enqueueMultiFieldSequential(Buffers &buffers, int elements, cudaStream_t stream) {
  constexpr int threads = 256;
  dim3 grid = gridFor(elements);
  fieldKernel<<<grid, threads, 0, stream>>>(buffers.a0.get(), buffers.a1.get(), elements, 0.00001f);
  fieldKernel<<<grid, threads, 0, stream>>>(buffers.b0.get(), buffers.b1.get(), elements, 0.00002f);
  joinKernel<<<grid, threads, 0, stream>>>(buffers.a1.get(), buffers.b1.get(), buffers.output.get(),
                                           elements);
  fieldKernel<<<grid, threads, 0, stream>>>(buffers.a1.get(), buffers.a0.get(), elements, 0.00001f);
  fieldKernel<<<grid, threads, 0, stream>>>(buffers.b1.get(), buffers.b0.get(), elements, 0.00002f);
  joinKernel<<<grid, threads, 0, stream>>>(buffers.a0.get(), buffers.b0.get(), buffers.output.get(),
                                           elements);
  CUDA_CHECK(cudaGetLastError());
}

void enqueueLinearSequential(Buffers &buffers, int elements, cudaStream_t stream) {
  constexpr int threads = 256;
  dim3 grid = gridFor(elements);
  stencilKernel<<<grid, threads, 0, stream>>>(buffers.a0.get(), buffers.a1.get(), elements, 0.001f);
  transformKernel<<<grid, threads, 0, stream>>>(buffers.a1.get(), buffers.scratch.get(), elements,
                                                0.999f, 0.0005f);
  stencilKernel<<<grid, threads, 0, stream>>>(buffers.scratch.get(), buffers.a0.get(), elements,
                                               0.001f);
  transformKernel<<<grid, threads, 0, stream>>>(buffers.a0.get(), buffers.output.get(), elements,
                                                0.999f, 0.0005f);
  CUDA_CHECK(cudaGetLastError());
}

void enqueueSequential(Workload workload, Buffers &buffers, int elements, cudaStream_t stream) {
  if (workload == Workload::MultiField)
    enqueueMultiFieldSequential(buffers, elements, stream);
  else
    enqueueLinearSequential(buffers, elements, stream);
}

GraphExec captureGraph(Workload workload, Buffers &buffers, int elements, cudaStream_t stream) {
  cudaGraph_t graph{};
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  enqueueSequential(workload, buffers, elements, stream);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  GraphExec executable(graph);
  CUDA_CHECK(cudaGraphDestroy(graph));
  return executable;
}

cudaGraphNode_t addStencil(cudaGraph_t graph, const std::vector<cudaGraphNode_t> &dependencies,
                           const float *input, float *output, int elements, float bias) {
  void *arguments[] = {&input, &output, &elements, &bias};
  cudaKernelNodeParams parameters{};
  parameters.func = reinterpret_cast<void *>(stencilKernel);
  parameters.gridDim = gridFor(elements);
  parameters.blockDim = dim3(256);
  parameters.kernelParams = arguments;
  cudaGraphNode_t node{};
  CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, dependencies.data(), dependencies.size(),
                                    &parameters));
  return node;
}

cudaGraphNode_t addField(cudaGraph_t graph, const std::vector<cudaGraphNode_t> &dependencies,
                         const float *input, float *output, int elements, float bias) {
  void *arguments[] = {&input, &output, &elements, &bias};
  cudaKernelNodeParams parameters{};
  parameters.func = reinterpret_cast<void *>(fieldKernel);
  parameters.gridDim = gridFor(elements);
  parameters.blockDim = dim3(256);
  parameters.kernelParams = arguments;
  cudaGraphNode_t node{};
  CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, dependencies.data(), dependencies.size(),
                                    &parameters));
  return node;
}

cudaGraphNode_t addTransform(cudaGraph_t graph, const std::vector<cudaGraphNode_t> &dependencies,
                             const float *input, float *output, int elements, float scale,
                             float bias) {
  void *arguments[] = {&input, &output, &elements, &scale, &bias};
  cudaKernelNodeParams parameters{};
  parameters.func = reinterpret_cast<void *>(transformKernel);
  parameters.gridDim = gridFor(elements);
  parameters.blockDim = dim3(256);
  parameters.kernelParams = arguments;
  cudaGraphNode_t node{};
  CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, dependencies.data(), dependencies.size(),
                                    &parameters));
  return node;
}

cudaGraphNode_t addJoin(cudaGraph_t graph, const std::vector<cudaGraphNode_t> &dependencies,
                        const float *first, const float *second, float *output, int elements) {
  void *arguments[] = {&first, &second, &output, &elements};
  cudaKernelNodeParams parameters{};
  parameters.func = reinterpret_cast<void *>(joinKernel);
  parameters.gridDim = gridFor(elements);
  parameters.blockDim = dim3(256);
  parameters.kernelParams = arguments;
  cudaGraphNode_t node{};
  CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, dependencies.data(), dependencies.size(),
                                    &parameters));
  return node;
}

GraphExec explicitGraph(Workload workload, Buffers &buffers, int elements) {
  cudaGraph_t graph{};
  CUDA_CHECK(cudaGraphCreate(&graph, 0));
  if (workload == Workload::MultiField) {
    auto a0 = addField(graph, {}, buffers.a0.get(), buffers.a1.get(), elements, 0.00001f);
    auto b0 = addField(graph, {}, buffers.b0.get(), buffers.b1.get(), elements, 0.00002f);
    auto join0 = addJoin(graph, {a0, b0}, buffers.a1.get(), buffers.b1.get(), buffers.output.get(),
                         elements);
    auto a1 = addField(graph, {a0}, buffers.a1.get(), buffers.a0.get(), elements, 0.00001f);
    auto b1 = addField(graph, {b0}, buffers.b1.get(), buffers.b0.get(), elements, 0.00002f);
    addJoin(graph, {a1, b1, join0}, buffers.a0.get(), buffers.b0.get(), buffers.output.get(),
            elements);
  } else {
    auto first = addStencil(graph, {}, buffers.a0.get(), buffers.a1.get(), elements, 0.001f);
    auto second = addTransform(graph, {first}, buffers.a1.get(), buffers.scratch.get(), elements,
                               0.999f, 0.0005f);
    auto third = addStencil(graph, {second}, buffers.scratch.get(), buffers.a0.get(), elements,
                            0.001f);
    addTransform(graph, {third}, buffers.a0.get(), buffers.output.get(), elements, 0.999f, 0.0005f);
  }
  GraphExec executable(graph);
  CUDA_CHECK(cudaGraphDestroy(graph));
  return executable;
}

class ManualStreams {
public:
  ManualStreams() : aDone0_(cudaEventDisableTiming), aDone1_(cudaEventDisableTiming),
                    bDone0_(cudaEventDisableTiming), bDone1_(cudaEventDisableTiming),
                    cycleStart_(cudaEventDisableTiming), cycleComplete_(cudaEventDisableTiming) {}

  void enqueue(Workload workload, Buffers &buffers, int elements, int cycles) {
    if (workload == Workload::LinearStencil) {
      for (int cycle = 0; cycle < cycles; ++cycle)
        enqueueLinearSequential(buffers, elements, join_);
      return;
    }

    constexpr int threads = 256;
    dim3 grid = gridFor(elements);
    CUDA_CHECK(cudaEventRecord(cycleStart_, join_));
    CUDA_CHECK(cudaStreamWaitEvent(a_, cycleStart_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(b_, cycleStart_, 0));
    for (int cycle = 0; cycle < cycles; ++cycle) {
      fieldKernel<<<grid, threads, 0, a_>>>(buffers.a0.get(), buffers.a1.get(), elements, 0.00001f);
      CUDA_CHECK(cudaEventRecord(aDone0_, a_));
      fieldKernel<<<grid, threads, 0, a_>>>(buffers.a1.get(), buffers.a0.get(), elements, 0.00001f);
      CUDA_CHECK(cudaEventRecord(aDone1_, a_));

      fieldKernel<<<grid, threads, 0, b_>>>(buffers.b0.get(), buffers.b1.get(), elements, 0.00002f);
      CUDA_CHECK(cudaEventRecord(bDone0_, b_));
      fieldKernel<<<grid, threads, 0, b_>>>(buffers.b1.get(), buffers.b0.get(), elements, 0.00002f);
      CUDA_CHECK(cudaEventRecord(bDone1_, b_));

      CUDA_CHECK(cudaStreamWaitEvent(join_, aDone0_, 0));
      CUDA_CHECK(cudaStreamWaitEvent(join_, bDone0_, 0));
      joinKernel<<<grid, threads, 0, join_>>>(buffers.a1.get(), buffers.b1.get(),
                                              buffers.output.get(), elements);
      CUDA_CHECK(cudaStreamWaitEvent(join_, aDone1_, 0));
      CUDA_CHECK(cudaStreamWaitEvent(join_, bDone1_, 0));
      joinKernel<<<grid, threads, 0, join_>>>(buffers.a0.get(), buffers.b0.get(),
                                              buffers.output.get(), elements);
      CUDA_CHECK(cudaEventRecord(cycleComplete_, join_));
      CUDA_CHECK(cudaStreamWaitEvent(a_, cycleComplete_, 0));
      CUDA_CHECK(cudaStreamWaitEvent(b_, cycleComplete_, 0));
    }
    CUDA_CHECK(cudaGetLastError());
  }

  cudaStream_t timingStream() const { return join_; }

private:
  Stream a_;
  Stream b_;
  Stream join_;
  Event aDone0_;
  Event aDone1_;
  Event bDone0_;
  Event bDone1_;
  Event cycleStart_;
  Event cycleComplete_;
};

struct Result {
  double medianMs{};
  double p95Ms{};
  double checksum{};
};

Result benchmark(Workload workload, Mode mode, Buffers &buffers, const Options &options) {
  Stream stream;
  ManualStreams streams;
  GraphExec executable;
  if (mode == Mode::Capture)
    executable = captureGraph(workload, buffers, options.elements, stream);
  else if (mode == Mode::Explicit)
    executable = explicitGraph(workload, buffers, options.elements);

  auto execute = [&](int cycles) {
    if (mode == Mode::Sequential) {
      for (int cycle = 0; cycle < cycles; ++cycle)
        enqueueSequential(workload, buffers, options.elements, stream);
    } else if (mode == Mode::Streams) {
      streams.enqueue(workload, buffers, options.elements, cycles);
    } else {
      for (int cycle = 0; cycle < cycles; ++cycle)
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
    }
  };

  buffers.reset();
  execute(options.warmup);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> samples;
  samples.reserve(options.repeats);
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    buffers.reset();
    Event begin;
    Event end;
    cudaStream_t timing = mode == Mode::Streams ? streams.timingStream()
                                                 : static_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaEventRecord(begin, timing));
    execute(options.cycles);
    // A graph or a manually coordinated schedule may execute work on streams
    // other than the launch/timing stream.  Fence the whole device before
    // placing the end marker so validation and timing cover every leaf.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(end, timing));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    samples.push_back(milliseconds);
  }

  std::sort(samples.begin(), samples.end());
  std::size_t p95Index = static_cast<std::size_t>(std::ceil(samples.size() * 0.95)) - 1;
  return {samples[samples.size() / 2], samples[p95Index], buffers.checksum(options.elements)};
}

bool checksumsMatch(double expected, double actual) {
  double tolerance = std::max(1e-3, std::abs(expected) * 1e-5);
  return std::isfinite(actual) && std::abs(expected - actual) <= tolerance;
}

std::string deviceName() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::string result(properties.name);
  std::replace(result.begin(), result.end(), ',', ';');
  return result;
}

} // namespace

int main(int argc, char **argv) try {
  Options options = parseOptions(argc, argv);
  std::string device = deviceName();
  Buffers buffers(options.elements);

  std::cout << "device,workload,mode,elements,cycles,warmup,repeats,median_ms,p95_ms,checksum,correct\n";
  std::cout << std::fixed << std::setprecision(6);
  for (Workload workload : {Workload::MultiField, Workload::LinearStencil}) {
    double reference = 0.0;
    for (Mode mode : {Mode::Sequential, Mode::Capture, Mode::Streams, Mode::Explicit}) {
      Result result = benchmark(workload, mode, buffers, options);
      if (mode == Mode::Sequential)
        reference = result.checksum;
      bool correct = checksumsMatch(reference, result.checksum);
      std::cout << device << ',' << name(workload) << ',' << name(mode) << ',' << options.elements
                << ',' << options.cycles << ',' << options.warmup << ',' << options.repeats << ','
                << result.medianMs << ',' << result.p95Ms << ',' << result.checksum << ','
                << (correct ? "true" : "false") << '\n';
    }
  }
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ckl-milestone0: " << error.what() << '\n';
  return 1;
}

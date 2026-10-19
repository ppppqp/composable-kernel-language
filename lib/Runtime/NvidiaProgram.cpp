#include "ckl/Runtime/NvidiaProgram.h"

#include <cuda.h>
#include <cudaProfiler.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>

using namespace mlir::ckl::runtime;

namespace mlir::ckl::runtime::detail {
struct NvidiaProgramState {
  using Entry = void (*)(void **);
  Entry entry;
  std::vector<uint64_t> bufferBytes;
  std::vector<std::string> scalarTypes;
  std::shared_ptr<NvidiaProgramState> aliasFallback;
  std::mutex moduleMutex;
  std::map<uint64_t, NvidiaModule> loadedModules;
};

struct NvidiaInvocationState {
  std::shared_ptr<NvidiaProgramState> program;
  std::shared_ptr<NvidiaProgramDevice> device;
  NvidiaStream stream;
  NvidiaModule module;
  std::vector<NvidiaBuffer> heaps;
  std::vector<std::shared_ptr<NvidiaProgramBuffer>> buffers;
  std::vector<uint64_t> resources;
  NvidiaPlan plan;
  KernelLaunch pending;
  std::vector<std::size_t> dependencies;
  std::map<uint64_t, std::size_t> nodes;
  std::map<std::pair<int, uint64_t>, NvidiaGraphExecutable> executionGraphs;
  std::mutex runMutex;
  std::string error;
  NvidiaInvocationState(std::shared_ptr<NvidiaProgramState> program,
                        std::shared_ptr<NvidiaProgramDevice> device)
      : program(std::move(program)), device(device), stream(device->runtime.createStream()) {}
};
} // namespace mlir::ckl::runtime::detail

using namespace mlir::ckl::runtime::detail;
namespace {
std::atomic<uint64_t> nextDeviceId{0};
// Exceptions never unwind through JIT frames. Each callback records failure and
// subsequent callbacks become no-ops; binding reports the error after entry returns.
template <typename F> void callback(void *opaque, F action) {
  auto &frame = *static_cast<NvidiaInvocationState *>(opaque);
  if (!frame.error.empty())
    return;
  try {
    action(frame);
  } catch (const std::exception &error) {
    frame.error = error.what();
  }
}
extern "C" void ckl_begin(void *frame, const void *binary, uint64_t bytes) {
  callback(frame, [&](NvidiaInvocationState &f) {
    std::lock_guard<std::mutex> lock(f.program->moduleMutex);
    auto &module = f.program->loadedModules[f.device->id];
    if (!module)
      module = f.device->runtime.loadCubin(binary, bytes);
    f.module = module;
  });
}
extern "C" void ckl_heap(void *frame, uint64_t id, uint64_t bytes) {
  callback(frame, [&](NvidiaInvocationState &f) {
    if (f.heaps.size() <= id)
      f.heaps.resize(id + 1);
    f.heaps[id] = f.device->runtime.allocate(bytes);
  });
}
extern "C" void ckl_resource(void *frame, uint64_t id, uint64_t heapId, uint64_t offset, void *external) {
  callback(frame, [&](NvidiaInvocationState &f) {
    if (f.resources.size() <= id)
      f.resources.resize(id + 1);
    f.resources[id] =
        external ? *static_cast<uint64_t *>(external) : f.heaps.at(heapId).address() + offset;
  });
}
extern "C" void ckl_kernel(void *frame, const char *name, uint64_t gx, uint64_t gy, uint64_t gz, uint64_t bx,
            uint64_t by, uint64_t bz, uint64_t shared) {
  callback(frame, [&](NvidiaInvocationState &f) {
    f.pending = {f.module.function(name),
                 {unsigned(gx), unsigned(gy), unsigned(gz)},
                 {unsigned(bx), unsigned(by), unsigned(bz)},
                 unsigned(shared),
                 {}};
    f.dependencies.clear();
  });
}
extern "C" void ckl_argument_resource(void *frame, uint64_t id, uint64_t offset) {
  callback(frame, [&](NvidiaInvocationState &f) { f.pending.arguments.add(f.resources.at(id) + offset); });
}
extern "C" void ckl_argument_value(void *frame, const void *value, uint64_t bytes) {
  callback(frame, [&](NvidiaInvocationState &f) { f.pending.arguments.addBytes(value, bytes); });
}
extern "C" void ckl_dependency(void *frame, uint64_t id) {
  callback(frame, [&](NvidiaInvocationState &f) { f.dependencies.push_back(f.nodes.at(id)); });
}
extern "C" void ckl_commit(void *frame, uint64_t id) {
  callback(frame, [&](NvidiaInvocationState &f) {
    f.nodes[id] = f.plan.addKernel(std::move(f.pending), std::move(f.dependencies));
  });
}

void validateExecution(NvidiaInvocationState &f, int mode, uint64_t batch) {
  if (mode < 0 || mode > 2 || !batch || f.plan.empty() || batch > 1000000 / f.plan.nodes().size())
    throw std::invalid_argument("invalid execution mode or batch exceeds one million kernel nodes");
}

NvidiaGraphExecutable &executionGraph(NvidiaInvocationState &f, int mode, uint64_t batch) {
  auto &graph = f.executionGraphs[{mode, batch}];
  if (!graph) {
    NvidiaPlan plan = batch == 1 ? f.plan : f.plan.repeat(batch);
    graph =
        mode == 2 ? f.device->runtime.capture(plan, f.stream) : f.device->runtime.instantiate(plan);
  }
  return graph;
}

void execute(NvidiaInvocationState &f, int mode, uint64_t batch) {
  if (mode)
    executionGraph(f, mode, batch).launch(f.stream);
  else
    for (uint64_t i = 0; i < batch; ++i)
      f.device->runtime.launchOrdinary(f.plan, f.stream);
}
} // namespace

NvidiaProgramDevice::NvidiaProgramDevice() : id(nextDeviceId++) {}

NvidiaProgram::NvidiaProgram(uintptr_t entry, std::vector<uint64_t> bufferBytes,
                             std::vector<std::string> scalarTypes,
                             std::shared_ptr<NvidiaProgram> fallback) {
  if (!entry || bufferBytes.size() != scalarTypes.size())
    throw std::invalid_argument("invalid host entry or argument metadata");
  state_ = std::make_shared<NvidiaProgramState>();
  state_->entry = reinterpret_cast<NvidiaProgramState::Entry>(entry);
  state_->bufferBytes = std::move(bufferBytes);
  state_->scalarTypes = std::move(scalarTypes);
  state_->aliasFallback = fallback ? fallback->state_ : nullptr;
}

const std::vector<uint64_t> &NvidiaProgram::bufferBytes() const { return state_->bufferBytes; }
const std::vector<std::string> &NvidiaProgram::scalarTypes() const { return state_->scalarTypes; }

NvidiaInvocation::NvidiaInvocation(std::unique_ptr<NvidiaInvocationState> state)
    : state_(std::move(state)) {}
NvidiaInvocation::~NvidiaInvocation() = default;
std::size_t NvidiaInvocation::nodeCount() const { return state_->plan.nodes().size(); }

std::shared_ptr<NvidiaInvocation> NvidiaProgram::bind(
    std::shared_ptr<NvidiaProgramDevice> d,
    const std::vector<NvidiaProgramArgument> &arguments) const {
  auto p = state_;
  if (!d)
    throw std::invalid_argument("program binding requires a device");
  const auto count = arguments.size();
  if (count != p->bufferBytes.size())
    throw std::invalid_argument("incorrect program argument count");
  std::vector<uint64_t> addresses(count), scalarStorage(count);
  std::vector<void *> packed(count);
  std::vector<std::shared_ptr<NvidiaProgramBuffer>> buffers;
  try {
    for (size_t i = 0; i < count; ++i) {
      if (p->bufferBytes[i]) {
        auto buffer = std::get<std::shared_ptr<NvidiaProgramBuffer>>(arguments[i]);
        if (!buffer || buffer->device != d || buffer->buffer.size() < p->bufferBytes[i])
          throw std::invalid_argument("buffer device or size does not match compiled signature");
        addresses[i] = buffer->buffer.address();
        packed[i] = &addresses[i];
        buffers.push_back(std::move(buffer));
      } else {
        const auto &type = p->scalarTypes[i];
        auto store = [&](auto value) { std::memcpy(&scalarStorage[i], &value, sizeof(value)); };
        if (type == "i32") store(std::get<int32_t>(arguments[i]));
        else if (type == "i64") store(std::get<int64_t>(arguments[i]));
        else if (type == "f32") store(std::get<float>(arguments[i]));
        else if (type == "f64") store(std::get<double>(arguments[i]));
        else throw std::invalid_argument("unsupported scalar binding type");
        packed[i] = &scalarStorage[i];
      }
    }
  } catch (const std::bad_variant_access &) {
    throw std::invalid_argument("argument type does not match compiled signature");
  }
  // Scalarized intermediates change ordering between threads when external
  // ranges overlap (including shifted views). Keep the original whole-kernel
  // boundaries for those bindings. Compare differences to avoid overflow.
  if (p->aliasFallback) {
    bool overlap = false;
    for (uint64_t i = 0; i < count; ++i)
      for (uint64_t j = i + 1; j < count; ++j) {
        if (!p->bufferBytes[i] || !p->bufferBytes[j])
          continue;
        overlap |= addresses[i] <= addresses[j] ? addresses[j] - addresses[i] < p->bufferBytes[i]
                                                : addresses[i] - addresses[j] < p->bufferBytes[j];
      }
    if (overlap)
      p = p->aliasFallback;
  }
  auto invocation = std::make_unique<NvidiaInvocationState>(p, d);
  invocation->buffers = std::move(buffers);
  void *frame = invocation.get();
  void **values = packed.data();
  void *entryArguments[] = {&frame, &values};
  p->entry(entryArguments);
  if (!invocation->error.empty())
    throw std::runtime_error(invocation->error);
  return std::shared_ptr<NvidiaInvocation>(new NvidiaInvocation(std::move(invocation)));
}

void NvidiaInvocation::run(int mode, uint64_t batch) {
  auto &f = *state_;
  std::lock_guard<std::mutex> lock(f.runMutex);
  validateExecution(f, mode, batch);
  execute(f, mode, batch);
  f.stream.synchronize();
}

std::array<double, 3> NvidiaInvocation::benchmark(int mode, uint64_t iterations,
                                                uint64_t batch) {
  auto &f = *state_;
  std::array<double, 3> result{};
  if (!iterations)
    throw std::invalid_argument("benchmark iterations must be positive");
  std::lock_guard<std::mutex> lock(f.runMutex);
  validateExecution(f, mode, batch);
  using Clock = std::chrono::steady_clock;
  f.stream.synchronize();
  auto setupStart = Clock::now();
  if (mode)
    executionGraph(f, mode, batch);
  result[2] = std::chrono::duration<double, std::milli>(Clock::now() - setupStart).count();
  auto check = [](CUresult error) {
    if (error != CUDA_SUCCESS) {
      const char *message = nullptr;
      cuGetErrorString(error, &message);
      throw std::runtime_error(message ? message : "CUDA event measurement failed");
    }
  };
  struct Event {
    CUevent value = nullptr;
    ~Event() {
      if (value)
        cuEventDestroy(value);
    }
  } start, stop;
  check(cuEventCreate(&start.value, CU_EVENT_DEFAULT));
  check(cuEventCreate(&stop.value, CU_EVENT_DEFAULT));
  auto stream = reinterpret_cast<CUstream>(f.stream.nativeHandle());
  check(cuEventRecord(start.value, stream));
  auto wallStart = Clock::now();
  for (uint64_t i = 0; i < iterations; ++i) {
    execute(f, mode, batch);
  }
  check(cuEventRecord(stop.value, stream));
  check(cuEventSynchronize(stop.value));
  auto wallStop = Clock::now();
  float elapsed;
  check(cuEventElapsedTime(&elapsed, start.value, stop.value));
  result[0] = elapsed * 1000.0 / iterations / batch;
  result[1] = std::chrono::duration<double, std::micro>(wallStop - wallStart).count() /
              iterations / batch;
  return result;
}

std::array<NvidiaRuntimeSymbol, 8> mlir::ckl::runtime::nvidiaProgramRuntimeSymbols() {
  return {{{"ckl_begin", reinterpret_cast<uintptr_t>(&ckl_begin)},
           {"ckl_heap", reinterpret_cast<uintptr_t>(&ckl_heap)},
           {"ckl_resource", reinterpret_cast<uintptr_t>(&ckl_resource)},
           {"ckl_kernel", reinterpret_cast<uintptr_t>(&ckl_kernel)},
           {"ckl_argument_resource", reinterpret_cast<uintptr_t>(&ckl_argument_resource)},
           {"ckl_argument_value", reinterpret_cast<uintptr_t>(&ckl_argument_value)},
           {"ckl_dependency", reinterpret_cast<uintptr_t>(&ckl_dependency)},
           {"ckl_commit", reinterpret_cast<uintptr_t>(&ckl_commit)}}};
}

bool mlir::ckl::runtime::nvidiaDeviceAvailable() {
  int count = 0;
  return cuInit(0) == CUDA_SUCCESS && cuDeviceGetCount(&count) == CUDA_SUCCESS && count > 0;
}
void mlir::ckl::runtime::nvidiaProfilerStart() {
  if (cuProfilerStart() != CUDA_SUCCESS)
    throw std::runtime_error("CUDA profiler start failed");
}
void mlir::ckl::runtime::nvidiaProfilerStop() {
  if (cuProfilerStop() != CUDA_SUCCESS)
    throw std::runtime_error("CUDA profiler stop failed");
}

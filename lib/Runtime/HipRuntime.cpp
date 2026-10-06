#include "ckl/Runtime/HipRuntime.h"

#include <hip/hip_runtime_api.h>

#include <fstream>
#include <unordered_map>

using namespace mlir::ckl::runtime;

namespace {

[[noreturn]] void throwHipError(hipError_t result, const char *operation) {
  const char *description = hipGetErrorString(result);
  throw HipError(operation, static_cast<int>(result),
                 description ? description : "unknown HIP error");
}

void check(hipError_t result, const char *operation) {
  if (result != hipSuccess)
    throwHipError(result, operation);
}

template <typename T> T checkedCast(void *value) { return reinterpret_cast<T>(value); }

} // namespace

namespace mlir::ckl::runtime::hip_detail {

struct ContextState {
  hipDevice_t device{};
  int deviceOrdinal = 0;

  void makeCurrent() const { check(hipSetDevice(deviceOrdinal), "hipSetDevice"); }
  void makeCurrentNoThrow() const noexcept { (void)hipSetDevice(deviceOrdinal); }
};

struct ModuleState {
  std::shared_ptr<ContextState> context;
  hipModule_t module{};

  ~ModuleState() {
    if (module) {
      context->makeCurrentNoThrow();
      (void)hipModuleUnload(module);
    }
  }
};

struct StreamState {
  std::shared_ptr<ContextState> context;
  hipStream_t stream{};
  bool owned = true;

  ~StreamState() {
    if (stream && owned) {
      context->makeCurrentNoThrow();
      (void)hipStreamDestroy(stream);
    }
  }
};

struct BufferState {
  std::shared_ptr<ContextState> context;
  void *pointer = nullptr;
  std::size_t bytes = 0;
  bool owned = true;

  ~BufferState() {
    if (pointer && owned) {
      context->makeCurrentNoThrow();
      (void)hipFree(pointer);
    }
  }
};

struct GraphState {
  std::shared_ptr<ContextState> context;
  hipGraph_t graph{};
  hipGraphExec_t executable{};
  std::vector<hipGraphNode_t> nodes;

  ~GraphState() {
    context->makeCurrentNoThrow();
    if (executable)
      (void)hipGraphExecDestroy(executable);
    if (graph)
      (void)hipGraphDestroy(graph);
  }
};

} // namespace mlir::ckl::runtime::hip_detail

HipError::HipError(std::string operation, int result, std::string description)
    : std::runtime_error(std::move(operation) + " failed: " + std::move(description)),
      result_(result) {}

void HipStream::synchronize() const {
  if (!state_)
    throw std::invalid_argument("cannot synchronize an empty HIP stream");
  state_->context->makeCurrent();
  check(hipStreamSynchronize(state_->stream), "hipStreamSynchronize");
}

std::uint64_t HipBuffer::address() const {
  if (!state_)
    throw std::invalid_argument("empty HIP buffer has no address");
  return reinterpret_cast<std::uint64_t>(static_cast<std::byte *>(state_->pointer) + offset_);
}

std::size_t HipBuffer::size() const { return state_ ? bytes_ : 0; }

void HipBuffer::copyFromHost(const void *source, std::size_t bytes, std::size_t offset) const {
  if (!state_ || offset > bytes_ || bytes > bytes_ - offset)
    throw std::out_of_range("host-to-device copy exceeds HIP buffer");
  state_->context->makeCurrent();
  check(hipMemcpy(static_cast<std::byte *>(state_->pointer) + offset_ + offset, source, bytes,
                  hipMemcpyHostToDevice),
        "hipMemcpyHostToDevice");
}

void HipBuffer::copyToHost(void *destination, std::size_t bytes, std::size_t offset) const {
  if (!state_ || offset > bytes_ || bytes > bytes_ - offset)
    throw std::out_of_range("device-to-host copy exceeds HIP buffer");
  state_->context->makeCurrent();
  check(hipMemcpy(destination, static_cast<std::byte *>(state_->pointer) + offset_ + offset, bytes,
                  hipMemcpyDeviceToHost),
        "hipMemcpyDeviceToHost");
}

void HipBuffer::fillZero() const {
  if (!state_)
    throw std::invalid_argument("cannot clear an empty HIP buffer");
  state_->context->makeCurrent();
  check(hipMemset(static_cast<std::byte *>(state_->pointer) + offset_, 0, bytes_), "hipMemset");
}

HipBuffer HipBuffer::slice(std::size_t offset, std::size_t bytes) const {
  if (!state_ || !bytes || offset > bytes_ || bytes > bytes_ - offset)
    throw std::out_of_range("HIP buffer slice exceeds its parent view");
  return HipBuffer(state_, offset_ + offset, bytes);
}

HipFunction HipModule::function(const std::string &name) const {
  if (!state_)
    throw std::invalid_argument("cannot resolve a function from an empty HIP module");
  state_->context->makeCurrent();
  hipFunction_t function{};
  check(hipModuleGetFunction(&function, state_->module, name.c_str()), "hipModuleGetFunction");
  HipFunction result;
  result.handle_ = reinterpret_cast<void *>(function);
  result.module_ = state_;
  return result;
}

HipPlan::NodeId HipPlan::addKernel(HipKernelLaunch launch, std::vector<NodeId> dependencies) {
  if (!launch.function)
    throw std::invalid_argument("HIP plan node requires a resolved function");
  if (!launch.grid.x || !launch.grid.y || !launch.grid.z || !launch.block.x || !launch.block.y ||
      !launch.block.z)
    throw std::invalid_argument("HIP plan node requires positive launch dimensions");
  NodeId id = nodes_.size();
  for (NodeId dependency : dependencies)
    if (dependency >= id)
      throw std::invalid_argument("HIP kernel dependencies must reference earlier plan nodes");
  nodes_.push_back({std::move(launch), std::move(dependencies)});
  return id;
}

HipPlan HipPlan::repeat(std::size_t iterations) const {
  if (!iterations)
    throw std::invalid_argument("HIP plan repeat count must be nonzero");
  if (empty())
    throw std::invalid_argument("cannot repeat an empty HIP plan");

  std::vector<bool> hasSuccessor(nodes_.size(), false);
  for (const Node &node : nodes_)
    for (NodeId dependency : node.dependencies)
      hasSuccessor[dependency] = true;
  std::vector<NodeId> leaves;
  for (NodeId node = 0; node < nodes_.size(); ++node)
    if (!hasSuccessor[node])
      leaves.push_back(node);

  HipPlan result;
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    NodeId offset = result.nodes_.size();
    for (NodeId nodeId = 0; nodeId < nodes_.size(); ++nodeId) {
      std::vector<NodeId> dependencies;
      dependencies.reserve(nodes_[nodeId].dependencies.size() + leaves.size());
      for (NodeId dependency : nodes_[nodeId].dependencies)
        dependencies.push_back(offset + dependency);
      if (iteration && nodes_[nodeId].dependencies.empty())
        for (NodeId leaf : leaves)
          dependencies.push_back(offset - nodes_.size() + leaf);
      result.addKernel(nodes_[nodeId].launch, std::move(dependencies));
    }
  }
  return result;
}

HipRuntime::HipRuntime(int deviceOrdinal) {
  check(hipInit(0), "hipInit");
  auto state = std::make_shared<hip_detail::ContextState>();
  state->deviceOrdinal = deviceOrdinal;
  check(hipDeviceGet(&state->device, deviceOrdinal), "hipDeviceGet");
  state->makeCurrent();
  context_ = std::move(state);
}

HipStream HipRuntime::createStream() const {
  context_->makeCurrent();
  auto state = std::make_shared<hip_detail::StreamState>();
  state->context = context_;
  check(hipStreamCreateWithFlags(&state->stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
  return HipStream(std::move(state));
}

HipStream HipRuntime::importStream(std::uintptr_t nativeHandle) const {
  auto state = std::make_shared<hip_detail::StreamState>();
  state->context = context_;
  state->stream = reinterpret_cast<hipStream_t>(nativeHandle);
  state->owned = false;
  return HipStream(std::move(state));
}

HipBuffer HipRuntime::allocate(std::size_t bytes) const {
  if (!bytes)
    throw std::invalid_argument("HIP allocation size must be nonzero");
  context_->makeCurrent();
  auto state = std::make_shared<hip_detail::BufferState>();
  state->context = context_;
  state->bytes = bytes;
  check(hipMalloc(&state->pointer, bytes), "hipMalloc");
  return HipBuffer(std::move(state), 0, bytes);
}

HipBuffer HipRuntime::importBuffer(std::uint64_t address, std::size_t bytes) const {
  if (!address || !bytes)
    throw std::invalid_argument("imported HIP buffer requires an address and size");
  auto state = std::make_shared<hip_detail::BufferState>();
  state->context = context_;
  state->pointer = reinterpret_cast<void *>(address);
  state->bytes = bytes;
  state->owned = false;
  return HipBuffer(std::move(state), 0, bytes);
}

// load a HIP module from raw HSACO data in memory.
HipModule HipRuntime::loadHsaco(const void *data, std::size_t bytes) const {
  if (!data || !bytes)
    throw std::invalid_argument("cannot load an empty HSACO");
  context_->makeCurrent();
  auto state = std::make_shared<hip_detail::ModuleState>();
  state->context = context_;
  check(hipModuleLoadData(&state->module, data), "hipModuleLoadData");
  return HipModule(std::move(state));
}

HipModule HipRuntime::loadHsacoFile(const std::filesystem::path &path) const {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open HSACO: " + path.string());
  std::streamsize size = input.tellg();
  if (size <= 0)
    throw std::runtime_error("HSACO is empty: " + path.string());
  input.seekg(0);
  std::vector<char> data(static_cast<std::size_t>(size));
  if (!input.read(data.data(), size))
    throw std::runtime_error("failed to read HSACO: " + path.string());
  return loadHsaco(data.data(), data.size());
}

/*
Resolve an execution plan into a HIP plan by loading the required artifacts and resolving the entry
points into HIP functions. The resolved plan can be launched directly or instantiated into a graph
executable.
*/
HipPlan HipRuntime::resolve(const ExecutionPlan &plan, const ArtifactRegistry &artifacts) const {
  if (plan.empty())
    throw std::invalid_argument("cannot resolve an empty execution plan");

  std::unordered_map<std::string, HipModule> modules;
  HipPlan resolved;
  for (const ExecutionPlan::Node &node : plan.nodes()) {
    // For each node in the execution plan, resolve the kernel invocation into a HIP kernel launch
    const KernelInvocation &invocation = node.invocation;
    if (invocation.deviceOrdinal != context_->deviceOrdinal)
      throw std::invalid_argument("kernel invocation targets a different HIP device");
    if (invocation.abi != "rocm.bare_ptr")
      throw std::invalid_argument("HIP executor does not support kernel ABI: " + invocation.abi);
    auto found = modules.find(invocation.artifact);
    if (found == modules.end()) {
      // Load the artifact from the registry and create a HIP module for it
      const KernelArtifact &artifact = artifacts.lookup(invocation.artifact);
      if (artifact.format() != "rocm.hsaco")
        throw std::invalid_argument("HIP executor does not support artifact format: " +
                                    artifact.format());
      found =
          modules.emplace(invocation.artifact, loadHsaco(artifact.data(), artifact.size())).first;
    }
    HipKernelLaunch launch{found->second.function(invocation.entryPoint), invocation.grid,
                           invocation.block, invocation.sharedMemoryBytes, invocation.arguments};
    resolved.addKernel(std::move(launch), node.dependencies);
  }
  return resolved;
}

void HipRuntime::launchOrdinary(const ExecutionPlan &plan, const ArtifactRegistry &artifacts,
                                const HipStream &stream) const {
  launchOrdinary(resolve(plan, artifacts), stream);
}

HipGraphExecutable HipRuntime::instantiate(const ExecutionPlan &plan,
                                           const ArtifactRegistry &artifacts) const {
  return instantiate(resolve(plan, artifacts));
}

std::shared_ptr<HipGraphExecutable>
HipRuntime::getOrCreateGraph(const std::string &key, const ExecutionPlan &plan,
                             const ArtifactRegistry &artifacts) {
  if (key.empty())
    throw std::invalid_argument("graph cache key must not be empty");
  if (auto found = graphCache_.find(key); found != graphCache_.end())
    return found->second;
  auto executable = std::make_shared<HipGraphExecutable>(instantiate(plan, artifacts));
  graphCache_.emplace(key, executable);
  return executable;
}

void HipRuntime::launchOrdinary(const HipPlan &plan, const HipStream &stream) const {
  if (!stream.state_)
    throw std::invalid_argument("ordinary launch requires a HIP stream");
  if (stream.state_->context != context_)
    throw std::invalid_argument("ordinary launch stream belongs to a different context");
  context_->makeCurrent();
  for (const HipPlan::Node &node : plan.nodes()) {
    const HipKernelLaunch &launch = node.launch;
    if (!launch.function.module_ || launch.function.module_->context != context_)
      throw std::invalid_argument("ordinary launch function belongs to a different context");
    std::vector<void *> arguments = launch.arguments.rawPointers();
    // launch every node in topological order to one stream.
    check(hipModuleLaunchKernel(checkedCast<hipFunction_t>(launch.function.handle_), launch.grid.x,
                                launch.grid.y, launch.grid.z, launch.block.x, launch.block.y,
                                launch.block.z, launch.sharedMemoryBytes, stream.state_->stream,
                                arguments.data(), nullptr),
          "hipModuleLaunchKernel");
  }
}

HipGraphExecutable HipRuntime::instantiate(const HipPlan &plan) const {
  // create a HIP graph executable from a resolved HIP plan.
  if (plan.empty())
    throw std::invalid_argument("cannot instantiate an empty HIP plan");
  context_->makeCurrent();
  auto state = std::make_shared<hip_detail::GraphState>();
  state->context = context_;
  check(hipGraphCreate(&state->graph, 0), "hipGraphCreate");
  state->nodes.reserve(plan.nodes().size());
  for (const HipPlan::Node &node : plan.nodes()) {
    if (!node.launch.function.module_ || node.launch.function.module_->context != context_)
      throw std::invalid_argument("graph launch function belongs to a different context");
    std::vector<hipGraphNode_t> dependencies;
    dependencies.reserve(node.dependencies.size());
    for (HipPlan::NodeId dependency : node.dependencies)
      dependencies.push_back(state->nodes.at(dependency));
    std::vector<void *> arguments = node.launch.arguments.rawPointers();

    hipKernelNodeParams parameters{};
    parameters.func = node.launch.function.handle_;
    parameters.gridDim = dim3(node.launch.grid.x, node.launch.grid.y, node.launch.grid.z);
    parameters.blockDim = dim3(node.launch.block.x, node.launch.block.y, node.launch.block.z);
    parameters.sharedMemBytes = node.launch.sharedMemoryBytes;
    parameters.kernelParams = arguments.data();
    hipGraphNode_t graphNode{};
    check(hipGraphAddKernelNode(&graphNode, state->graph, dependencies.data(), dependencies.size(),
                                &parameters),
          "hipGraphAddKernelNode");
    state->nodes.push_back(graphNode);
  }
  check(hipGraphInstantiate(&state->executable, state->graph, nullptr, nullptr, 0),
        "hipGraphInstantiate");
  return HipGraphExecutable(std::move(state));
}

std::shared_ptr<HipGraphExecutable> HipRuntime::getOrCreateGraph(const std::string &key,
                                                                 const HipPlan &plan) {
  if (key.empty())
    throw std::invalid_argument("graph cache key must not be empty");
  if (auto found = graphCache_.find(key); found != graphCache_.end())
    return found->second;
  auto executable = std::make_shared<HipGraphExecutable>(instantiate(plan));
  graphCache_.emplace(key, executable);
  return executable;
}

void HipRuntime::synchronize() const {
  context_->makeCurrent();
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

std::string HipRuntime::deviceName() const {
  char name[256]{};
  check(hipDeviceGetName(name, sizeof(name), context_->device), "hipDeviceGetName");
  return name;
}

void HipGraphExecutable::launch(const HipStream &stream) const {
  // Launch a HIP graph executable on a stream. The graph must be instantiated first.
  if (!state_ || !stream.state_)
    throw std::invalid_argument("graph launch requires an executable and stream");
  if (stream.state_->context != state_->context)
    throw std::invalid_argument("graph launch stream belongs to a different context");
  state_->context->makeCurrent();
  check(hipGraphLaunch(state_->executable, stream.state_->stream), "hipGraphLaunch");
}

void HipGraphExecutable::updateKernel(HipPlan::NodeId node, const HipKernelLaunch &launch) {
  if (!state_ || node >= state_->nodes.size())
    throw std::out_of_range("graph kernel update references an unknown node");
  if (!launch.function)
    throw std::invalid_argument("graph kernel update requires a resolved function");
  if (!launch.function.module_ || launch.function.module_->context != state_->context)
    throw std::invalid_argument("updated graph function belongs to a different context");
  state_->context->makeCurrent();
  std::vector<void *> arguments = launch.arguments.rawPointers();

  hipKernelNodeParams parameters{};
  parameters.func = launch.function.handle_;
  parameters.gridDim = dim3(launch.grid.x, launch.grid.y, launch.grid.z);
  parameters.blockDim = dim3(launch.block.x, launch.block.y, launch.block.z);
  parameters.sharedMemBytes = launch.sharedMemoryBytes;
  parameters.kernelParams = arguments.data();
  check(hipGraphExecKernelNodeSetParams(state_->executable, state_->nodes[node], &parameters),
        "hipGraphExecKernelNodeSetParams");
}

std::size_t HipGraphExecutable::nodeCount() const { return state_ ? state_->nodes.size() : 0; }

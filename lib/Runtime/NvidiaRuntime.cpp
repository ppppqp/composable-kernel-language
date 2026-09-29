
#include "ckl/Runtime/NvidiaRuntime.h"

#include <cuda.h>

#include <fstream>

using namespace mlir::ckl::runtime;

namespace {

[[noreturn]] void throwDriverError(CUresult result, const char *operation) {
  const char *name = nullptr;
  const char *description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  std::string message = name ? name : "unknown CUDA error";
  if (description) {
    message += ": ";
    message += description;
  }
  throw DriverError(operation, static_cast<int>(result), std::move(message));
}

void check(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS)
    throwDriverError(result, operation);
}

template <typename T> T checkedCast(void *value) { return reinterpret_cast<T>(value); }

} // namespace

namespace mlir::ckl::runtime::detail {

struct ContextState {
  CUdevice device{};
  CUcontext context{};

  ~ContextState() {
    if (context)
      cuDevicePrimaryCtxRelease(device);
  }

  void makeCurrent() const { check(cuCtxSetCurrent(context), "cuCtxSetCurrent"); }
};

struct ModuleState {
  std::shared_ptr<ContextState> context;
  CUmodule module{};

  ~ModuleState() {
    if (module) {
      context->makeCurrent();
      cuModuleUnload(module);
    }
  }
};

struct StreamState {
  std::shared_ptr<ContextState> context;
  CUstream stream{};

  ~StreamState() {
    if (stream) {
      context->makeCurrent();
      cuStreamDestroy(stream);
    }
  }
};

struct BufferState {
  std::shared_ptr<ContextState> context;
  CUdeviceptr pointer{};
  std::size_t bytes{};

  ~BufferState() {
    if (pointer) {
      context->makeCurrent();
      cuMemFree(pointer);
    }
  }
};

struct GraphState {
  std::shared_ptr<ContextState> context;
  CUgraph graph{};
  CUgraphExec executable{};
  std::vector<CUgraphNode> nodes;

  ~GraphState() {
    context->makeCurrent();
    if (executable)
      cuGraphExecDestroy(executable);
    if (graph)
      cuGraphDestroy(graph);
  }
};

} // namespace mlir::ckl::runtime::detail

DriverError::DriverError(std::string operation, int result, std::string description)
    : std::runtime_error(std::move(operation) + " failed: " + std::move(description)),
      result_(result) {}

std::vector<void *> KernelArguments::rawPointers() const {
  std::vector<void *> pointers;
  pointers.reserve(values_.size());
  for (const std::vector<std::byte> &value : values_)
    pointers.push_back(const_cast<std::byte *>(value.data()));
  return pointers;
}

void NvidiaStream::synchronize() const {
  if (!state_)
    throw std::invalid_argument("cannot synchronize an empty NVIDIA stream");
  state_->context->makeCurrent();
  check(cuStreamSynchronize(state_->stream), "cuStreamSynchronize");
}

std::uint64_t NvidiaBuffer::address() const {
  if (!state_)
    throw std::invalid_argument("empty NVIDIA buffer has no address");
  return static_cast<std::uint64_t>(state_->pointer + offset_);
}

std::size_t NvidiaBuffer::size() const { return state_ ? bytes_ : 0; }

void NvidiaBuffer::copyFromHost(const void *source, std::size_t bytes, std::size_t offset) const {
  if (!state_ || offset > bytes_ || bytes > bytes_ - offset)
    throw std::out_of_range("host-to-device copy exceeds NVIDIA buffer");
  state_->context->makeCurrent();
  check(cuMemcpyHtoD(state_->pointer + offset_ + offset, source, bytes), "cuMemcpyHtoD");
}

void NvidiaBuffer::copyToHost(void *destination, std::size_t bytes, std::size_t offset) const {
  if (!state_ || offset > bytes_ || bytes > bytes_ - offset)
    throw std::out_of_range("device-to-host copy exceeds NVIDIA buffer");
  state_->context->makeCurrent();
  check(cuMemcpyDtoH(destination, state_->pointer + offset_ + offset, bytes), "cuMemcpyDtoH");
}

void NvidiaBuffer::fillZero() const {
  if (!state_)
    throw std::invalid_argument("cannot clear an empty NVIDIA buffer");
  state_->context->makeCurrent();
  check(cuMemsetD8(state_->pointer + offset_, 0, bytes_), "cuMemsetD8");
}

NvidiaBuffer NvidiaBuffer::slice(std::size_t offset, std::size_t bytes) const {
  if (!state_ || !bytes || offset > bytes_ || bytes > bytes_ - offset)
    throw std::out_of_range("NVIDIA buffer slice exceeds its parent view");
  return NvidiaBuffer(state_, offset_ + offset, bytes);
}

NvidiaFunction NvidiaModule::function(const std::string &name) const {
  if (!state_)
    throw std::invalid_argument("cannot resolve a function from an empty NVIDIA module");
  state_->context->makeCurrent();
  CUfunction function{};
  check(cuModuleGetFunction(&function, state_->module, name.c_str()), "cuModuleGetFunction");
  NvidiaFunction result;
  result.handle_ = reinterpret_cast<void *>(function);
  result.module_ = state_;
  return result;
}

NvidiaPlan::NodeId NvidiaPlan::addKernel(KernelLaunch launch, std::vector<NodeId> dependencies) {
  if (!launch.function)
    throw std::invalid_argument("kernel plan node requires a resolved function");
  if (!launch.grid.x || !launch.grid.y || !launch.grid.z || !launch.block.x || !launch.block.y ||
      !launch.block.z)
    throw std::invalid_argument("kernel plan node requires positive launch dimensions");
  NodeId id = nodes_.size();
  for (NodeId dependency : dependencies)
    if (dependency >= id)
      // prevent cycles
      throw std::invalid_argument("kernel dependencies must reference earlier plan nodes");
  nodes_.push_back({std::move(launch), std::move(dependencies)});
  return id;
}

NvidiaPlan NvidiaPlan::repeat(std::size_t iterations) const {
  if (!iterations)
    throw std::invalid_argument("NVIDIA plan repeat count must be nonzero");
  if (empty())
    throw std::invalid_argument("cannot repeat an empty NVIDIA plan");

  std::vector<bool> hasSuccessor(nodes_.size(), false);
  for (const Node &node : nodes_)
    for (NodeId dependency : node.dependencies)
      hasSuccessor[dependency] = true;
  std::vector<NodeId> leaves;
  for (NodeId node = 0; node < nodes_.size(); ++node)
    if (!hasSuccessor[node])
      leaves.push_back(node);

  NvidiaPlan result;
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

NvidiaRuntime::NvidiaRuntime(int deviceOrdinal) {
  check(cuInit(0), "cuInit");
  auto state = std::make_shared<detail::ContextState>();
  check(cuDeviceGet(&state->device, deviceOrdinal), "cuDeviceGet");
  check(cuDevicePrimaryCtxRetain(&state->context, state->device), "cuDevicePrimaryCtxRetain");
  state->makeCurrent();
  context_ = std::move(state);
}

NvidiaStream NvidiaRuntime::createStream() const {
  context_->makeCurrent();
  auto state = std::make_shared<detail::StreamState>();
  state->context = context_;
  check(cuStreamCreate(&state->stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
  return NvidiaStream(std::move(state));
}

NvidiaBuffer NvidiaRuntime::allocate(std::size_t bytes) const {
  if (!bytes)
    throw std::invalid_argument("NVIDIA allocation size must be nonzero");
  context_->makeCurrent();
  auto state = std::make_shared<detail::BufferState>();
  state->context = context_;
  state->bytes = bytes;
  check(cuMemAlloc(&state->pointer, bytes), "cuMemAlloc");
  return NvidiaBuffer(std::move(state), 0, bytes);
}

NvidiaModule NvidiaRuntime::loadCubin(const void *data, std::size_t bytes) const {
  if (!data || !bytes)
    throw std::invalid_argument("cannot load an empty CUBIN");
  context_->makeCurrent();
  auto state = std::make_shared<detail::ModuleState>();
  state->context = context_;
  check(cuModuleLoadData(&state->module, data), "cuModuleLoadData");
  return NvidiaModule(std::move(state));
}

NvidiaModule NvidiaRuntime::loadCubinFile(const std::filesystem::path &path) const {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open CUBIN: " + path.string());
  std::streamsize size = input.tellg();
  if (size <= 0)
    throw std::runtime_error("CUBIN is empty: " + path.string());
  input.seekg(0);
  std::vector<char> data(static_cast<std::size_t>(size));
  if (!input.read(data.data(), size))
    throw std::runtime_error("failed to read CUBIN: " + path.string());
  return loadCubin(data.data(), data.size());
}

void NvidiaRuntime::launchOrdinary(const NvidiaPlan &plan, const NvidiaStream &stream) const {
  if (!stream.state_)
    throw std::invalid_argument("ordinary launch requires an NVIDIA stream");
  if (stream.state_->context != context_)
    throw std::invalid_argument("ordinary launch stream belongs to a different context");
  context_->makeCurrent();
  for (const NvidiaPlan::Node &node : plan.nodes()) {
    // serialize the launch of every node in topological order to one stream.
    const KernelLaunch &launch = node.launch;
    if (!launch.function.module_ || launch.function.module_->context != context_)
      throw std::invalid_argument("ordinary launch function belongs to a different context");
    std::vector<void *> arguments = launch.arguments.rawPointers();
    check(cuLaunchKernel(checkedCast<CUfunction>(launch.function.handle_), launch.grid.x,
                         launch.grid.y, launch.grid.z, launch.block.x, launch.block.y,
                         launch.block.z, launch.sharedMemoryBytes, stream.state_->stream,
                         arguments.data(), nullptr),
          "cuLaunchKernel");
  }
}

NvidiaGraphExecutable NvidiaRuntime::instantiate(const NvidiaPlan &plan) const {
  if (plan.empty())
    throw std::invalid_argument("cannot instantiate an empty NVIDIA plan");
  context_->makeCurrent();
  auto state = std::make_shared<detail::GraphState>();
  state->context = context_;
  check(cuGraphCreate(&state->graph, 0), "cuGraphCreate");
  state->nodes.reserve(plan.nodes().size());
  for (const NvidiaPlan::Node &node : plan.nodes()) {
    // for every plan node
    if (!node.launch.function.module_ || node.launch.function.module_->context != context_)
      throw std::invalid_argument("graph launch function belongs to a different context");
    std::vector<CUgraphNode> dependencies;
    dependencies.reserve(node.dependencies.size());
    for (NvidiaPlan::NodeId dependency : node.dependencies)
      // convert plan predecessors to graph node handles
      dependencies.push_back(state->nodes.at(dependency));
    std::vector<void *> arguments = node.launch.arguments.rawPointers();

    // construct CUDA_KERNEL_NODE_PARAMS
    CUDA_KERNEL_NODE_PARAMS parameters{};
    parameters.func = checkedCast<CUfunction>(node.launch.function.handle_);
    parameters.gridDimX = node.launch.grid.x;
    parameters.gridDimY = node.launch.grid.y;
    parameters.gridDimZ = node.launch.grid.z;
    parameters.blockDimX = node.launch.block.x;
    parameters.blockDimY = node.launch.block.y;
    parameters.blockDimZ = node.launch.block.z;
    parameters.sharedMemBytes = node.launch.sharedMemoryBytes;
    parameters.kernelParams = arguments.data();
    CUgraphNode graphNode{};

    // call cuGraphAddKernelNode to add the node to the graph
    check(cuGraphAddKernelNode(&graphNode, state->graph, dependencies.data(), dependencies.size(),
                               &parameters),
          "cuGraphAddKernelNode");
    state->nodes.push_back(graphNode);
  }
  // produces a reusable CUgraphExec from the graph, which can be launched on a stream.
  // can be launched by graph.launch(stream)
  check(cuGraphInstantiate(&state->executable, state->graph, 0), "cuGraphInstantiate");
  return NvidiaGraphExecutable(std::move(state));
}

std::shared_ptr<NvidiaGraphExecutable> NvidiaRuntime::getOrCreateGraph(const std::string &key,
                                                                       const NvidiaPlan &plan) {
  if (key.empty())
    throw std::invalid_argument("graph cache key must not be empty");
  if (auto found = graphCache_.find(key); found != graphCache_.end())
    return found->second;
  auto executable = std::make_shared<NvidiaGraphExecutable>(instantiate(plan));
  graphCache_.emplace(key, executable);
  return executable;
}

void NvidiaRuntime::synchronize() const {
  context_->makeCurrent();
  check(cuCtxSynchronize(), "cuCtxSynchronize");
}

std::string NvidiaRuntime::deviceName() const {
  char name[256]{};
  check(cuDeviceGetName(name, sizeof(name), context_->device), "cuDeviceGetName");
  return name;
}

void NvidiaGraphExecutable::launch(const NvidiaStream &stream) const {
  if (!state_ || !stream.state_)
    throw std::invalid_argument("graph launch requires an executable and stream");
  if (stream.state_->context != state_->context)
    throw std::invalid_argument("graph launch stream belongs to a different context");
  state_->context->makeCurrent();
  check(cuGraphLaunch(state_->executable, stream.state_->stream), "cuGraphLaunch");
}

void NvidiaGraphExecutable::updateKernel(NvidiaPlan::NodeId node, const KernelLaunch &launch) {
  // update the parameters of a kernel node in the graph executable
  // it does not require re-instantiating the graph, but the new parameters must be compatible with
  // the original node
  if (!state_ || node >= state_->nodes.size())
    throw std::out_of_range("graph kernel update references an unknown node");
  if (!launch.function)
    throw std::invalid_argument("graph kernel update requires a resolved function");
  if (!launch.function.module_ || launch.function.module_->context != state_->context)
    throw std::invalid_argument("updated graph function belongs to a different context");
  state_->context->makeCurrent();
  std::vector<void *> arguments = launch.arguments.rawPointers();

  // recreates the CUDA_KERNEL_NODE_PARAMS for the updated kernel node
  CUDA_KERNEL_NODE_PARAMS parameters{};
  parameters.func = checkedCast<CUfunction>(launch.function.handle_);
  parameters.gridDimX = launch.grid.x;
  parameters.gridDimY = launch.grid.y;
  parameters.gridDimZ = launch.grid.z;
  parameters.blockDimX = launch.block.x;
  parameters.blockDimY = launch.block.y;
  parameters.blockDimZ = launch.block.z;
  parameters.sharedMemBytes = launch.sharedMemoryBytes;
  parameters.kernelParams = arguments.data();

  // update the kernel node parameters
  check(cuGraphExecKernelNodeSetParams(state_->executable, state_->nodes[node], &parameters),
        "cuGraphExecKernelNodeSetParams");
}

std::size_t NvidiaGraphExecutable::nodeCount() const { return state_ ? state_->nodes.size() : 0; }

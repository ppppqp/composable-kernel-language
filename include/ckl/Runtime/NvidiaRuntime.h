/*
Wrapper for the NVIDIA runtime API.
*/

#ifndef CKL_RUNTIME_NVIDIARUNTIME_H
#define CKL_RUNTIME_NVIDIARUNTIME_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::ckl::runtime {

namespace detail {
struct ContextState;
struct ModuleState;
struct StreamState;
struct BufferState;
struct GraphState;
} // namespace detail

struct Dim3 {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;
};

class DriverError : public std::runtime_error {
public:
  DriverError(std::string operation, int result, std::string description);
  int result() const { return result_; }

private:
  int result_;
};

class NvidiaStream {
public:
  NvidiaStream() = default;
  void synchronize() const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  friend class NvidiaRuntime;
  friend class NvidiaGraphExecutable;
  explicit NvidiaStream(std::shared_ptr<detail::StreamState> state) : state_(std::move(state)) {}
  std::shared_ptr<detail::StreamState> state_;
};

class NvidiaBuffer {
public:
  NvidiaBuffer() = default;
  std::uint64_t address() const;
  std::size_t size() const;
  void copyFromHost(const void *source, std::size_t bytes, std::size_t offset = 0) const;
  void copyToHost(void *destination, std::size_t bytes, std::size_t offset = 0) const;
  void fillZero() const;
  /// Return a bounded view that shares ownership of this allocation.
  NvidiaBuffer slice(std::size_t offset, std::size_t bytes) const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  friend class NvidiaRuntime;
  explicit NvidiaBuffer(std::shared_ptr<detail::BufferState> state, std::size_t offset = 0,
                        std::size_t bytes = 0)
      : state_(std::move(state)), offset_(offset), bytes_(bytes) {}
  std::shared_ptr<detail::BufferState> state_;
  std::size_t offset_ = 0;
  std::size_t bytes_ = 0;
};

class NvidiaFunction {
public:
  NvidiaFunction() = default;
  explicit operator bool() const { return handle_ != nullptr; }

private:
  friend class NvidiaModule;
  friend class NvidiaRuntime;
  friend class NvidiaGraphExecutable;
  void *handle_ = nullptr;
  std::shared_ptr<detail::ModuleState> module_;
};

class NvidiaModule {
public:
  NvidiaModule() = default;
  NvidiaFunction function(const std::string &name) const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  friend class NvidiaRuntime;
  explicit NvidiaModule(std::shared_ptr<detail::ModuleState> state) : state_(std::move(state)) {}
  std::shared_ptr<detail::ModuleState> state_;
};

class KernelArguments {
public:
  KernelArguments() = default;

  template <typename T> KernelArguments &add(T value) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "CUDA kernel arguments must be trivially copyable");
    std::vector<std::byte> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    values_.push_back(std::move(bytes));
    return *this;
  }

  std::size_t size() const { return values_.size(); }

private:
  friend class NvidiaRuntime;
  friend class NvidiaGraphExecutable;
  std::vector<void *> rawPointers() const;
  std::vector<std::vector<std::byte>> values_;
};

struct KernelLaunch {
  NvidiaFunction function;
  Dim3 grid;
  Dim3 block;
  unsigned sharedMemoryBytes = 0;
  KernelArguments arguments;
};

// the executable form of CKL's normalized graph.
class NvidiaPlan {
public:
  using NodeId = std::size_t;
  // each node contains a KernelLaunch and a list of predecessor node IDs.
  struct Node {
    KernelLaunch launch;
    std::vector<NodeId> dependencies;
  };

  /// Append a node in topological order. Dependencies must reference earlier nodes.
  NodeId addKernel(KernelLaunch launch, std::vector<NodeId> dependencies = {});
  /// Clone the topology and arguments. Roots of each later iteration depend on all leaves of the
  /// prior iteration, preserving repeated-launch semantics inside one larger graph boundary.
  NvidiaPlan repeat(std::size_t iterations) const;
  const std::vector<Node> &nodes() const { return nodes_; }
  bool empty() const { return nodes_.empty(); }

private:
  std::vector<Node> nodes_;
};

// Represents a compiled and instantiated CUDA graph that can be launched on a stream.
class NvidiaGraphExecutable {
public:
  NvidiaGraphExecutable() = default;
  void launch(const NvidiaStream &stream) const;
  void updateKernel(NvidiaPlan::NodeId node, const KernelLaunch &launch);
  std::size_t nodeCount() const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  friend class NvidiaRuntime;
  explicit NvidiaGraphExecutable(std::shared_ptr<detail::GraphState> state)
      : state_(std::move(state)) {}
  std::shared_ptr<detail::GraphState> state_;
};

// Represents an NVIDIA runtime context and provides methods to manage streams, buffers, modules,
// and graph execution.
class NvidiaRuntime {
public:
  explicit NvidiaRuntime(int deviceOrdinal = 0);

  NvidiaStream createStream() const;
  NvidiaBuffer allocate(std::size_t bytes) const;
  NvidiaModule loadCubin(const void *data, std::size_t bytes) const;
  NvidiaModule loadCubinFile(const std::filesystem::path &path) const;

  /// Submit every node in topological order to one stream. This is intentionally serialized.
  void launchOrdinary(const NvidiaPlan &plan, const NvidiaStream &stream) const;
  NvidiaGraphExecutable instantiate(const NvidiaPlan &plan) const;
  /// Reuse an executable when the caller's key covers topology, binary, ABI, and target identity.
  std::shared_ptr<NvidiaGraphExecutable> getOrCreateGraph(const std::string &key,
                                                          const NvidiaPlan &plan);
  std::size_t cachedGraphCount() const { return graphCache_.size(); }
  void clearGraphCache() { graphCache_.clear(); }
  void synchronize() const;
  std::string deviceName() const;

private:
  std::shared_ptr<detail::ContextState> context_;
  // cache of instantiated graphs keyed by a string that uniquely identifies the graph's topology
  // the key must be unique for each graph topology, binary, ABI, and target identity.
  std::unordered_map<std::string, std::shared_ptr<NvidiaGraphExecutable>> graphCache_;
};

} // namespace mlir::ckl::runtime

#endif

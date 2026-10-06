/*
HIP Resource wrapper
*/

#ifndef CKL_RUNTIME_HIPRUNTIME_H
#define CKL_RUNTIME_HIPRUNTIME_H

#include "ckl/Runtime/ExecutionPlan.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::ckl::runtime {

namespace hip_detail {
struct ContextState;
struct ModuleState;
struct StreamState;
struct BufferState;
struct GraphState;
} // namespace hip_detail

class HipError : public std::runtime_error {
public:
  HipError(std::string operation, int result, std::string description);
  int result() const { return result_; }

private:
  int result_;
};

class HipStream {
public:
  HipStream() = default;
  void synchronize() const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  friend class HipRuntime;
  friend class HipGraphExecutable;
  explicit HipStream(std::shared_ptr<hip_detail::StreamState> state) : state_(std::move(state)) {}
  std::shared_ptr<hip_detail::StreamState> state_;
};

class HipBuffer {
public:
  HipBuffer() = default;
  std::uint64_t address() const;
  std::size_t size() const;
  void copyFromHost(const void *source, std::size_t bytes, std::size_t offset = 0) const;
  void copyToHost(void *destination, std::size_t bytes, std::size_t offset = 0) const;
  void fillZero() const;
  HipBuffer slice(std::size_t offset, std::size_t bytes) const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  friend class HipRuntime;
  explicit HipBuffer(std::shared_ptr<hip_detail::BufferState> state, std::size_t offset = 0,
                     std::size_t bytes = 0)
      : state_(std::move(state)), offset_(offset), bytes_(bytes) {}
  std::shared_ptr<hip_detail::BufferState> state_;
  std::size_t offset_ = 0;
  std::size_t bytes_ = 0;
};

class HipFunction {
public:
  HipFunction() = default;
  explicit operator bool() const { return handle_ != nullptr; }

private:
  friend class HipModule;
  friend class HipRuntime;
  friend class HipGraphExecutable;
  void *handle_ = nullptr;
  std::shared_ptr<hip_detail::ModuleState> module_;
};

class HipModule {
public:
  HipModule() = default;
  HipFunction function(const std::string &name) const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  friend class HipRuntime;
  explicit HipModule(std::shared_ptr<hip_detail::ModuleState> state) : state_(std::move(state)) {}
  std::shared_ptr<hip_detail::ModuleState> state_;
};

struct HipKernelLaunch {
  HipFunction function;
  Dim3 grid;
  Dim3 block;
  unsigned sharedMemoryBytes = 0;
  KernelArguments arguments;
};

class HipPlan {
public:
  using NodeId = std::size_t;
  struct Node {
    HipKernelLaunch launch;
    std::vector<NodeId> dependencies;
  };

  NodeId addKernel(HipKernelLaunch launch, std::vector<NodeId> dependencies = {});
  HipPlan repeat(std::size_t iterations) const;
  const std::vector<Node> &nodes() const { return nodes_; }
  bool empty() const { return nodes_.empty(); }

private:
  std::vector<Node> nodes_;
};

class HipGraphExecutable {
public:
  HipGraphExecutable() = default;
  void launch(const HipStream &stream) const;
  void updateKernel(HipPlan::NodeId node, const HipKernelLaunch &launch);
  std::size_t nodeCount() const;
  explicit operator bool() const { return state_ != nullptr; }

private:
  friend class HipRuntime;
  explicit HipGraphExecutable(std::shared_ptr<hip_detail::GraphState> state)
      : state_(std::move(state)) {}
  std::shared_ptr<hip_detail::GraphState> state_;
};

/*
Supports both CKL-owned and framework-owned objects:
- Owned allocation: HipBuffer buffer = runtime.allocate(bytes);
- Borrowed allocation: HipBuffer buffer = runtime.importBuffer(torchPointer, bytes);
Borrowed allocation are never freed by CKL.
*/
class HipRuntime {
public:
  explicit HipRuntime(int deviceOrdinal = 0);

  HipStream createStream() const;
  /// Borrow a stream owned by an embedding framework. Zero selects the default stream.
  HipStream importStream(std::uintptr_t nativeHandle) const;
  HipBuffer allocate(std::size_t bytes) const;
  /// Borrow externally allocated device memory. CKL never frees the imported allocation.
  HipBuffer importBuffer(std::uint64_t address, std::size_t bytes) const;
  HipModule loadHsaco(const void *data, std::size_t bytes) const;
  HipModule loadHsacoFile(const std::filesystem::path &path) const;

  HipPlan resolve(const ExecutionPlan &plan, const ArtifactRegistry &artifacts) const;
  void launchOrdinary(const ExecutionPlan &plan, const ArtifactRegistry &artifacts,
                      const HipStream &stream) const;
  HipGraphExecutable instantiate(const ExecutionPlan &plan,
                                 const ArtifactRegistry &artifacts) const;
  std::shared_ptr<HipGraphExecutable> getOrCreateGraph(const std::string &key,
                                                       const ExecutionPlan &plan,
                                                       const ArtifactRegistry &artifacts);

  void launchOrdinary(const HipPlan &plan, const HipStream &stream) const;
  HipGraphExecutable instantiate(const HipPlan &plan) const;
  /// Reuse an executable when the key covers topology, artifact, ABI, and target identity.
  std::shared_ptr<HipGraphExecutable> getOrCreateGraph(const std::string &key, const HipPlan &plan);
  std::size_t cachedGraphCount() const { return graphCache_.size(); }
  void clearGraphCache() { graphCache_.clear(); }
  void synchronize() const;
  std::string deviceName() const;

private:
  std::shared_ptr<hip_detail::ContextState> context_;
  std::unordered_map<std::string, std::shared_ptr<HipGraphExecutable>> graphCache_;
};

} // namespace mlir::ckl::runtime

#endif

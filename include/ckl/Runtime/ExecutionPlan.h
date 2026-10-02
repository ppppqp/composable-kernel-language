#ifndef CKL_RUNTIME_EXECUTIONPLAN_H
#define CKL_RUNTIME_EXECUTIONPLAN_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::ckl::runtime {

struct Dim3 {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;
};

/// An immutable compiler-produced object. The producer owns the meaning of its ABI; CKL only
/// requires a backend executor that understands `format` and the invocation's ABI identity.
class KernelArtifact {
public:
  KernelArtifact(std::string identity, std::string format, std::string target,
                 std::vector<std::byte> data);

  static KernelArtifact readFile(std::string identity, std::string format, std::string target,
                                 const std::filesystem::path &path);

  const std::string &identity() const { return identity_; } // e.g. "my_kernel.cubin"
  const std::string &format() const { return format_; }     // e.g. "cuda.cubin" or "rocm.hsaco"
  const std::string &target() const { return target_; }     // e.g. "sm_80" or "gfx908"
  const void *data() const { return data_->data(); } // pointer to the raw bytes of the artifact
  std::size_t size() const { return data_->size(); } // size of the raw bytes of the artifact

private:
  std::string identity_;
  std::string format_;
  std::string target_;
  std::shared_ptr<const std::vector<std::byte>> data_;
};

/// Artifact identities are stable plan inputs, not paths chosen by an executor. This lets a DSL
/// adapter hand CKL an in-memory CUBIN/HSACO without surrendering its compiler or disk cache.
class ArtifactRegistry {
public:
  void add(KernelArtifact artifact);
  const KernelArtifact &lookup(const std::string &identity) const;
  bool contains(const std::string &identity) const;
  std::size_t size() const { return artifacts_.size(); }

private:
  std::unordered_map<std::string, KernelArtifact> artifacts_;
};

/// Owns byte-exact values for the physical kernel ABI. Logical-to-physical expansion belongs to
/// the producer adapter; the backend executor only passes these slots to the device driver.
class KernelArguments {
public:
  KernelArguments() = default;

  template <typename T> KernelArguments &add(T value) {
    // moves the kernel arguments to its own stable storage.
    // At driver launch time, CKL converts these values into the expected void **kernelParams
    static_assert(std::is_trivially_copyable_v<T>, "kernel arguments must be trivially copyable");
    std::vector<std::byte> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    values_.push_back(std::move(bytes));
    return *this;
  }

  KernelArguments &addBytes(const void *data, std::size_t bytes);
  std::size_t size() const { return values_.size(); }

private:
  friend class NvidiaRuntime;
  friend class NvidiaGraphExecutable;
  std::vector<void *> rawPointers() const;
  std::vector<std::vector<std::byte>> values_;
};

/// A backend-neutral, fully physical kernel invocation. It deliberately contains no tensor IR,
/// allocation policy, or producer operation: those concerns have been resolved before execution.
struct KernelInvocation {
  std::string artifact;
  std::string entryPoint;
  std::string abi;
  int deviceOrdinal = 0;
  Dim3 grid;
  Dim3 block;
  unsigned sharedMemoryBytes = 0;
  KernelArguments arguments;
};

/// Portable handoff between CKL planning and a thin device executor. Nodes must be appended in
/// topological order so an executor can materialize a native command graph without re-analysis.

/*
auto a = plan.addKernel(invocationA);
auto b = plan.addKernel(invocationB);
plan.addKernel(invocationC, {a, b}); // C depends on A and B
*/
class ExecutionPlan {
public:
  using NodeId = std::size_t;
  struct Node {
    KernelInvocation invocation;
    std::vector<NodeId> dependencies;
  };

  NodeId addKernel(KernelInvocation invocation, std::vector<NodeId> dependencies = {});

  // repeat the plan N times, preserving topological order
  // basically a loop unrolling of the plan, but the nodes are still in topological order
  ExecutionPlan repeat(std::size_t iterations) const;
  const std::vector<Node> &nodes() const { return nodes_; }
  bool empty() const { return nodes_.empty(); }

private:
  std::vector<Node> nodes_;
};

} // namespace mlir::ckl::runtime

#endif

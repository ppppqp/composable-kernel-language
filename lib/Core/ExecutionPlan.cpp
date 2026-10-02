#include "ckl/Runtime/ExecutionPlan.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

using namespace mlir::ckl::runtime;

KernelArtifact::KernelArtifact(std::string identity, std::string format, std::string target,
                               std::vector<std::byte> data)
    : identity_(std::move(identity)), format_(std::move(format)), target_(std::move(target)),
      data_(std::make_shared<const std::vector<std::byte>>(std::move(data))) {
  if (identity_.empty() || format_.empty())
    throw std::invalid_argument("kernel artifact requires identity and format");
  if (data_->empty())
    throw std::invalid_argument("kernel artifact data must not be empty");
}

KernelArtifact KernelArtifact::readFile(std::string identity, std::string format,
                                        std::string target, const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open kernel artifact: " + path.string());
  std::streamsize size = input.tellg();
  if (size <= 0)
    throw std::runtime_error("kernel artifact is empty: " + path.string());
  input.seekg(0);
  std::vector<std::byte> data(static_cast<std::size_t>(size));
  if (!input.read(reinterpret_cast<char *>(data.data()), size))
    throw std::runtime_error("failed to read kernel artifact: " + path.string());
  return KernelArtifact(std::move(identity), std::move(format), std::move(target),
                        std::move(data));
}

void ArtifactRegistry::add(KernelArtifact artifact) {
  std::string identity = artifact.identity();
  if (!artifacts_.emplace(identity, std::move(artifact)).second)
    throw std::invalid_argument("duplicate kernel artifact identity: " + identity);
}

const KernelArtifact &ArtifactRegistry::lookup(const std::string &identity) const {
  auto found = artifacts_.find(identity);
  if (found == artifacts_.end())
    throw std::out_of_range("unknown kernel artifact: " + identity);
  return found->second;
}

bool ArtifactRegistry::contains(const std::string &identity) const {
  return artifacts_.find(identity) != artifacts_.end();
}

KernelArguments &KernelArguments::addBytes(const void *data, std::size_t bytes) {
  if (!data || !bytes)
    throw std::invalid_argument("physical kernel argument must not be empty");
  std::vector<std::byte> value(bytes);
  std::memcpy(value.data(), data, bytes);
  values_.push_back(std::move(value));
  return *this;
}

std::vector<void *> KernelArguments::rawPointers() const {
  std::vector<void *> pointers;
  pointers.reserve(values_.size());
  for (const std::vector<std::byte> &value : values_)
    pointers.push_back(const_cast<std::byte *>(value.data()));
  return pointers;
}

ExecutionPlan::NodeId ExecutionPlan::addKernel(KernelInvocation invocation,
                                               std::vector<NodeId> dependencies) {
  if (invocation.artifact.empty() || invocation.entryPoint.empty() || invocation.abi.empty())
    throw std::invalid_argument(
        "kernel invocation requires artifact, entry point, and ABI identities");
  if (invocation.deviceOrdinal < 0)
    throw std::invalid_argument("kernel invocation requires a non-negative device ordinal");
  if (!invocation.grid.x || !invocation.grid.y || !invocation.grid.z || !invocation.block.x ||
      !invocation.block.y || !invocation.block.z)
    throw std::invalid_argument("kernel invocation requires positive launch dimensions");
  NodeId id = nodes_.size();
  for (NodeId dependency : dependencies)
    if (dependency >= id)
      throw std::invalid_argument("kernel dependencies must reference earlier plan nodes");
  nodes_.push_back({std::move(invocation), std::move(dependencies)});
  return id;
}

ExecutionPlan ExecutionPlan::repeat(std::size_t iterations) const {
  if (!iterations)
    throw std::invalid_argument("execution plan repeat count must be nonzero");
  if (empty())
    throw std::invalid_argument("cannot repeat an empty execution plan");

  std::vector<bool> hasSuccessor(nodes_.size(), false);
  for (const Node &node : nodes_)
    for (NodeId dependency : node.dependencies)
      hasSuccessor[dependency] = true;
  std::vector<NodeId> leaves;
  for (NodeId node = 0; node < nodes_.size(); ++node)
    if (!hasSuccessor[node])
      leaves.push_back(node);

  ExecutionPlan result;
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
      result.addKernel(nodes_[nodeId].invocation, std::move(dependencies));
    }
  }
  return result;
}

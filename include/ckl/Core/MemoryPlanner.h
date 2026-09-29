/*
Device-independent temporary-memory planning for orchestration graphs.
*/

#ifndef CKL_CORE_MEMORYPLANNER_H
#define CKL_CORE_MEMORYPLANNER_H

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace mlir::ckl::planning {

using NodeId = std::size_t;
using ResourceId = std::size_t;

struct GraphEdge {
  NodeId from;
  NodeId to;
};

enum class ResourceKind {
  Temporary,  // eligible for reuse
  Persistent, // allocated by the plan, but never freed
  External,   // owned by the caller, not allocated or freed by the plan
};

/// A resource's users include every graph node that may access it. The planner derives lifetime
/// ordering from graph reachability rather than trusting source-order lifetime annotations.
struct Resource {
  std::string name;
  std::size_t bytes = 0;
  std::size_t alignment = 1;
  std::string addressSpace = "global";
  int device = 0;
  ResourceKind kind = ResourceKind::Temporary;
  std::vector<NodeId> users; // derived from operations
};

struct Heap {
  int device = 0;
  std::string addressSpace;
  std::size_t bytes = 0;
  std::size_t alignment = 1;
};

struct Assignment {
  ResourceId resource = 0;
  std::size_t heap = 0;
  std::size_t offset = 0;
  std::size_t bytes = 0;
  std::size_t slot = 0;
};

struct ReuseDecision {
  ResourceId resource = 0;
  std::size_t slot = 0;
  std::vector<ResourceId> sharesWith;
  std::string reason;
};

struct MemoryPlan {
  std::size_t baselineBytes = 0;        // storage without recycling
  std::size_t plannedBytes = 0;         // packed storage for reuse
  std::vector<Heap> heaps;              // one heap per unique device and address space
  std::vector<Assignment> assignments;  // resource-to-heap mapping
  std::vector<ReuseDecision> decisions; // reuse provenance and explanations

  const Assignment *find(ResourceId resource) const;
};

struct MemoryPlanResult {
  std::optional<MemoryPlan> value;
  std::string error;

  explicit operator bool() const { return value.has_value(); }
};

/// Plans storage without changing graph topology. A temporary can join a storage slot only when
/// its complete use set is strictly ordered before or after every existing resident's use set.
class MemoryPlanner {
public:
  MemoryPlan plan(std::size_t nodeCount, const std::vector<GraphEdge> &edges,
                  const std::vector<Resource> &resources) const;
  /// Non-throwing entry point for compiler pipelines built with exception handling disabled.
  MemoryPlanResult tryPlan(std::size_t nodeCount, const std::vector<GraphEdge> &edges,
                           const std::vector<Resource> &resources) const noexcept;
};

} // namespace mlir::ckl::planning

#endif

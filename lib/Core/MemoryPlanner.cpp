#include "ckl/Core/MemoryPlanner.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

using namespace mlir::ckl::planning;

namespace {

std::size_t alignTo(std::size_t value, std::size_t alignment) {
  if (!alignment || (alignment & (alignment - 1)) != 0)
    throw std::invalid_argument("resource alignment must be a nonzero power of two");
  if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    throw std::overflow_error("memory plan size overflow");
  return (value + alignment - 1) & ~(alignment - 1);
}

class Reachability {
public:
  Reachability(std::size_t nodeCount, const std::vector<GraphEdge> &edges) : nodeCount_(nodeCount) {
    if (nodeCount && nodeCount > std::numeric_limits<std::size_t>::max() / nodeCount)
      throw std::overflow_error("memory-planning graph is too large");
    reachable_.resize(nodeCount * nodeCount, false);
    for (const GraphEdge &edge : edges) {
      if (edge.from >= nodeCount || edge.to >= nodeCount)
        throw std::out_of_range("memory-planning edge references an unknown graph node");
      if (edge.from == edge.to)
        throw std::invalid_argument("memory-planning graph contains a self edge");
      reachable_[index(edge.from, edge.to)] = true;
    }
    // floyd-warshall algorithm to compute transitive closure of the graph
    for (NodeId via = 0; via < nodeCount; ++via)
      for (NodeId from = 0; from < nodeCount; ++from)
        if (reachable_[index(from, via)])
          // if from is reachable to via, then for every node to that is reachable from via, from is
          // also reachable to to
          for (NodeId to = 0; to < nodeCount; ++to)
            reachable_[index(from, to)] = reachable_[index(from, to)] || reachable_[index(via, to)];
    for (NodeId node = 0; node < nodeCount; ++node)
      if (reachable_[index(node, node)])
        throw std::invalid_argument("memory-planning graph must be acyclic");
  }

  bool before(const Resource &first, const Resource &second) const {
    // Resource A is before Resource B if every user of A is reachable to every user of B.
    // This means their lifetimes are strictly ordered and they can share a storage slot.
    if (first.users.empty() || second.users.empty())
      return false;
    for (NodeId lhs : first.users)
      for (NodeId rhs : second.users)
        if (lhs == rhs || !reachable_[index(lhs, rhs)])
          return false;
    return true;
  }

private:
  std::size_t index(NodeId from, NodeId to) const { return from * nodeCount_ + to; }
  std::size_t nodeCount_;
  std::vector<bool>
      reachable_; // reachable_[from * nodeCount_ + to] is true if from is reachable to to
};

struct Slot {
  int device;
  std::string addressSpace;
  std::size_t bytes;
  std::size_t alignment;
  bool recyclable;
  std::vector<ResourceId> residents;
  std::size_t heap = 0;
  std::size_t offset = 0;
};

bool compatible(const Resource &resource, const Slot &slot, const std::vector<Resource> &resources,
                const Reachability &reachability) {
  // a temporary can only enter an existing slot if
  // 1. the slot is recyclable (i.e., it was created for a temporary)
  // 2. the resource and slot are on the same device and address space
  // 3. the resource's lifetime is strictly ordered before or after every existing resident's
  if (!slot.recyclable || resource.device != slot.device ||
      resource.addressSpace != slot.addressSpace)
    return false;
  return std::all_of(slot.residents.begin(), slot.residents.end(), [&](ResourceId resident) {
    return reachability.before(resource, resources[resident]) ||
           reachability.before(resources[resident], resource);
  });
}

} // namespace

const Assignment *MemoryPlan::find(ResourceId resource) const {
  auto found = std::find_if(assignments.begin(), assignments.end(),
                            [=](const Assignment &item) { return item.resource == resource; });
  return found == assignments.end() ? nullptr : &*found;
}

MemoryPlan MemoryPlanner::plan(std::size_t nodeCount, const std::vector<GraphEdge> &edges,
                               const std::vector<Resource> &resources) const {
  Reachability reachability(nodeCount, edges);
  MemoryPlan result;
  std::vector<Slot> slots;
  std::vector<std::size_t> resourceSlots(resources.size(), std::numeric_limits<std::size_t>::max());

  // greedy slot selection
  for (ResourceId id = 0; id < resources.size(); ++id) {
    const Resource &resource = resources[id];
    if (!resource.bytes)
      // validate size
      throw std::invalid_argument("memory-planning resource size must be nonzero");
    // validate alignment
    alignTo(resource.bytes, resource.alignment);

    // validate users
    for (NodeId user : resource.users)
      if (user >= nodeCount)
        throw std::out_of_range("memory-planning resource references an unknown graph node");
    if (resource.kind == ResourceKind::Temporary && resource.users.empty())
      throw std::invalid_argument("temporary memory-planning resource must have at least one user");

    // external resources are not allocated or freed by the plan
    if (resource.kind == ResourceKind::External)
      continue;

    std::size_t baselineAllocation = alignTo(resource.bytes, resource.alignment);
    if (result.baselineBytes > std::numeric_limits<std::size_t>::max() - baselineAllocation)
      throw std::overflow_error("memory plan baseline size overflow");
    result.baselineBytes += baselineAllocation;
    std::size_t selected = std::numeric_limits<std::size_t>::max();
    std::size_t leastGrowth = std::numeric_limits<std::size_t>::max();
    if (resource.kind == ResourceKind::Temporary) {
      for (std::size_t index = 0; index < slots.size(); ++index) {
        const Slot &slot = slots[index];
        if (!compatible(resource, slot, resources, reachability))
          continue;
        // if compatible, select the one with least size growth to minimize allocation
        std::size_t grownBytes = std::max(slot.bytes, resource.bytes);
        std::size_t growth = grownBytes - slot.bytes;
        if (growth < leastGrowth || (growth == leastGrowth && index < selected)) {
          selected = index;
          leastGrowth = growth;
        }
      }
    }

    if (selected == std::numeric_limits<std::size_t>::max()) {
      // no compatible slot was found, so create a new one
      selected = slots.size();
      slots.push_back({resource.device,
                       resource.addressSpace,
                       resource.bytes,
                       resource.alignment,
                       resource.kind == ResourceKind::Temporary,
                       {}});
    }
    Slot &slot = slots[selected];
    std::vector<ResourceId> previous = slot.residents;
    slot.bytes = std::max(slot.bytes, resource.bytes);
    slot.alignment = std::max(slot.alignment, resource.alignment);
    slot.residents.push_back(id);
    resourceSlots[id] = selected;
    bool reused = !previous.empty();
    result.decisions.push_back({id, selected, std::move(previous),
                                reused ? "all resource uses are ordered by existing graph edges"
                                       : (resource.kind == ResourceKind::Persistent
                                              ? "persistent resource gets dedicated storage"
                                              : "no compatible ordered lifetime")});
  }

  // after reuse decisions are complete, slots are packed into heaps
  for (std::size_t slotId = 0; slotId < slots.size(); ++slotId) {
    Slot &slot = slots[slotId];
    // a separate heap is created for every unique device and address space combination
    auto heapIt = std::find_if(result.heaps.begin(), result.heaps.end(), [&](const Heap &heap) {
      return heap.device == slot.device && heap.addressSpace == slot.addressSpace;
    });
    if (heapIt == result.heaps.end()) {
      result.heaps.push_back({slot.device, slot.addressSpace, 0, slot.alignment});
      heapIt = std::prev(result.heaps.end());
    }
    slot.heap = static_cast<std::size_t>(std::distance(result.heaps.begin(), heapIt));
    heapIt->alignment = std::max(heapIt->alignment, slot.alignment);
    slot.offset = alignTo(heapIt->bytes, slot.alignment);
    if (slot.offset > std::numeric_limits<std::size_t>::max() - slot.bytes)
      throw std::overflow_error("memory plan heap size overflow");
    heapIt->bytes = slot.offset + slot.bytes;
  }

  for (Heap &heap : result.heaps) {
    heap.bytes = alignTo(heap.bytes, heap.alignment);
    if (result.plannedBytes > std::numeric_limits<std::size_t>::max() - heap.bytes)
      throw std::overflow_error("memory plan total size overflow");
    result.plannedBytes += heap.bytes;
  }
  for (ResourceId id = 0; id < resources.size(); ++id) {
    if (resources[id].kind == ResourceKind::External)
      continue;
    const Slot &slot = slots[resourceSlots[id]];
    result.assignments.push_back(
        {id, slot.heap, slot.offset, resources[id].bytes, resourceSlots[id]});
  }
  return result;
}

MemoryPlanResult MemoryPlanner::tryPlan(std::size_t nodeCount,
                                        const std::vector<GraphEdge> &edges,
                                        const std::vector<Resource> &resources) const noexcept {
  try {
    return {plan(nodeCount, edges, resources), {}};
  } catch (const std::exception &error) {
    return {std::nullopt, error.what()};
  } catch (...) {
    return {std::nullopt, "unknown memory-planning failure"};
  }
}

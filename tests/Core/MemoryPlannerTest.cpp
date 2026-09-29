#include "ckl/Core/MemoryPlanner.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace mlir::ckl::planning;

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() try {
  // 0 -> 1 -> 4 and 2 -> 3 -> 4 are parallel branches. 5 -> 6 follows both.
  std::vector<GraphEdge> edges = {{0, 1}, {1, 4}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};
  std::vector<Resource> resources = {
      {"early-a", 1024, 256, "global", 0, ResourceKind::Temporary, {0, 1}},
      {"parallel-b", 512, 256, "global", 0, ResourceKind::Temporary, {2, 3}},
      {"late", 768, 128, "global", 0, ResourceKind::Temporary, {5, 6}},
      {"persistent", 256, 64, "global", 0, ResourceKind::Persistent, {0, 6}},
      {"external", 4096, 256, "global", 0, ResourceKind::External, {0, 6}},
  };

  MemoryPlan plan = MemoryPlanner().plan(7, edges, resources);
  const Assignment *early = plan.find(0);
  const Assignment *parallel = plan.find(1);
  const Assignment *late = plan.find(2);
  const Assignment *persistent = plan.find(3);
  require(early && parallel && late && persistent, "allocated resources need assignments");
  require(!plan.find(4), "external resources must not be allocated");
  require(early->slot != parallel->slot, "parallel lifetimes must not share storage");
  require(late->slot == early->slot, "ordered compatible lifetimes should share best-fit storage");
  require(persistent->slot != early->slot && persistent->slot != parallel->slot,
          "persistent storage must not be recycled");
  require(plan.baselineBytes == 2560, "unexpected unplanned byte count");
  require(plan.plannedBytes == 1792, "unexpected planned byte count");
  require(plan.decisions[2].sharesWith.size() == 1 && plan.decisions[2].sharesWith[0] == 0,
          "reuse provenance must identify the prior resident");

  MemoryPlan persistentFirst = MemoryPlanner().plan(
      2, {{0, 1}}, {{"persistent-first", 64, 64, "global", 0, ResourceKind::Persistent, {0}},
                    {"later-temporary", 64, 64, "global", 0, ResourceKind::Temporary, {1}}});
  require(persistentFirst.find(0)->slot != persistentFirst.find(1)->slot,
          "a later temporary must not recycle persistent storage");

  bool rejectedCycle = false;
  try {
    MemoryPlanner().plan(2, {{0, 1}, {1, 0}}, {});
  } catch (const std::invalid_argument &) {
    rejectedCycle = true;
  }
  require(rejectedCycle, "cyclic graphs must be rejected");

  std::cout << "baseline_bytes=" << plan.baselineBytes << ",planned_bytes=" << plan.plannedBytes
            << ",saved_bytes=" << plan.baselineBytes - plan.plannedBytes << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ckl-memory-planner-test: " << error.what() << '\n';
  return 1;
}

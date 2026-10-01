#include "ckl/Analysis/GraphAlgorithms.h"

#include "llvm/ADT/STLExtras.h"

#include <vector>

using namespace mlir;
using namespace mlir::ckl;

DispatchDependencyMap mlir::ckl::getReducedDispatchDependencies(GraphOp graph) {
  SmallVector<GraphNodeOp> dispatches;
  std::size_t nodeCount = 0;
  for (GraphNodeOp node : graph.getBody().front().getOps<GraphNodeOp>()) {
    nodeCount = std::max(nodeCount, static_cast<std::size_t>(node.getId() + 1));
    if (node.getKind() == "dispatch")
      dispatches.push_back(node);
  }

  std::vector<bool> reachable(nodeCount * nodeCount, false);
  auto index = [=](std::size_t from, std::size_t to) { return from * nodeCount + to; };
  for (GraphEdgeOp edge : graph.getBody().front().getOps<GraphEdgeOp>())
    reachable[index(edge.getFrom(), edge.getTo())] = true;
  for (std::size_t via = 0; via < nodeCount; ++via)
    for (std::size_t from = 0; from < nodeCount; ++from)
      if (reachable[index(from, via)])
        for (std::size_t to = 0; to < nodeCount; ++to)
          reachable[index(from, to)] =
              reachable[index(from, to)] || reachable[index(via, to)];

  DispatchDependencyMap result;
  for (GraphNodeOp target : dispatches)
    for (GraphNodeOp candidate : dispatches) {
      if (candidate.getId() >= target.getId() ||
          !reachable[index(candidate.getId(), target.getId())])
        continue;
      bool transitive = llvm::any_of(dispatches, [&](GraphNodeOp intermediate) {
        return candidate.getId() < intermediate.getId() &&
               intermediate.getId() < target.getId() &&
               reachable[index(candidate.getId(), intermediate.getId())] &&
               reachable[index(intermediate.getId(), target.getId())];
      });
      if (!transitive)
        result[target.getId()].push_back(candidate.getId());
    }
  return result;
}

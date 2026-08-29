#pragma once

#include "ckl/Core/Layout/IndexMap.h"

#include <string>

namespace ckl::core {

// Space separation inspired by composable kernel (AMD) design.
struct Distribution {
  IndexSpace executorSpace; // space of independent executors, e.g. threads or workgroups
  IndexSpace localSpace;    // private space, e.g. registers or local memory
  IndexSpace tileSpace;     // space of the global tile, e.g. a tensor or matrix
  IndexMap ownership; // maps executor × local to tile, e.g. which thread owns which tile element
  IndexMap
      localStorage; // maps local to private storage, e.g. which register holds which local value
};

struct DistributionCheck {
  bool valid;
  bool covering;
  bool unique;
  std::string message;
};

DistributionCheck verifyDistribution(const Distribution &distribution, bool requireCovering = true,
                                     bool requireUnique = true);

enum class ConversionKind {
  Identity,             // ownership and local storage agree (noop)
  LocalPermutation,     // ownership agrees but local storage differs (register rename/permutation)
  SubgroupExchange,     // ownership changes within each subgroup (warp shuffle)
  SharedMemoryExchange, // ownership changes across subgroup boundaries (smem exchange)
  Unsupported, // ownership changes in a way that cannot be implemented with a single exchange
};

struct ConversionPlan {
  ConversionKind kind;
  std::string reason;
};

// layout conversion
ConversionPlan classifyConversion(const Distribution &source, const Distribution &target,
                                  std::int64_t subgroupSize);
const char *toString(ConversionKind kind);

} // namespace ckl::core

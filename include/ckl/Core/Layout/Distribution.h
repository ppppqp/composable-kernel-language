#pragma once

#include "ckl/Core/Layout/IndexMap.h"

#include <string>

namespace ckl::core {

struct Distribution {
  IndexSpace executorSpace;
  IndexSpace localSpace;
  IndexSpace tileSpace;
  IndexMap ownership;
  IndexMap localStorage;
};

struct DistributionCheck {
  bool valid;
  bool covering;
  bool unique;
  std::string message;
};

DistributionCheck verifyDistribution(const Distribution &distribution,
                                     bool requireCovering = true,
                                     bool requireUnique = true);

enum class ConversionKind {
  Identity,
  LocalPermutation,
  SubgroupExchange,
  SharedMemoryExchange,
  Unsupported,
};

struct ConversionPlan {
  ConversionKind kind;
  std::string reason;
};

ConversionPlan classifyConversion(const Distribution &source, const Distribution &target,
                                  std::int64_t subgroupSize);
const char *toString(ConversionKind kind);

} // namespace ckl::core


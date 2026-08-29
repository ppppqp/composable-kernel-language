#include "ckl/Core/Layout/Distribution.h"

#include <map>
#include <sstream>
#include <stdexcept>

namespace ckl::core {
namespace {

std::vector<std::int64_t> concatenate(const std::vector<std::int64_t> &lhs,
                                      const std::vector<std::int64_t> &rhs) {
  std::vector<std::int64_t> result = lhs;
  result.insert(result.end(), rhs.begin(), rhs.end());
  return result;
}

std::string key(const std::vector<std::int64_t> &point) {
  std::ostringstream os;
  for (std::int64_t value : point)
    os << value << ',';
  return os.str();
}

bool sameTileShape(const Distribution &lhs, const Distribution &rhs) {
  return lhs.tileSpace.sameShape(rhs.tileSpace);
}

} // namespace

DistributionCheck verifyDistribution(const Distribution &distribution, bool requireCovering,
                                     bool requireUnique) {
  if (distribution.ownership.domain().rank() !=
      distribution.executorSpace.rank() + distribution.localSpace.rank())
    return {false, false, false, "ownership domain must be executor × local"};
  if (!distribution.ownership.codomain().sameShape(distribution.tileSpace))
    return {false, false, false, "ownership codomain must match the tile"};
  if (!distribution.localStorage.domain().sameShape(distribution.localSpace))
    return {false, false, false, "local-storage domain must match local space"};

  std::map<std::string, std::size_t> owners;
  for (const auto &executor : enumerate(distribution.executorSpace)) {
    for (const auto &local : enumerate(distribution.localSpace)) {
      auto tile = distribution.ownership.apply(concatenate(executor, local));
      ++owners[key(tile)];
    }
  }

  bool covering = owners.size() == static_cast<std::size_t>(distribution.tileSpace.volume());
  bool unique = true;
  for (const auto &[unused, count] : owners) {
    (void)unused;
    unique &= count == 1;
  }
  bool valid = (!requireCovering || covering) && (!requireUnique || unique);
  return {valid, covering, unique,
          valid ? "distribution satisfies requested invariants"
                : !covering ? "distribution does not cover its tile"
                            : "distribution has replicated owners"};
}

ConversionPlan classifyConversion(const Distribution &source, const Distribution &target,
                                  std::int64_t subgroupSize) {
  if (!sameTileShape(source, target))
    return {ConversionKind::Unsupported, "tile shapes differ"};
  if (!source.executorSpace.sameShape(target.executorSpace) ||
      !source.localSpace.sameShape(target.localSpace))
    return {ConversionKind::SharedMemoryExchange,
            "executor or per-agent cardinality differs"};

  bool sameOwnership = true;
  bool sameLocalStorage = true;
  bool staysInSubgroup = true;
  std::map<std::string, std::int64_t> sourceOwners;
  std::int64_t executorLinear = 0;
  for (const auto &executor : enumerate(source.executorSpace)) {
    for (const auto &local : enumerate(source.localSpace)) {
      auto input = concatenate(executor, local);
      sourceOwners[key(source.ownership.apply(input))] = executorLinear;
      sameOwnership &= source.ownership.apply(input) == target.ownership.apply(input);
      sameLocalStorage &= source.localStorage.apply(local) == target.localStorage.apply(local);
    }
    ++executorLinear;
  }

  if (sameOwnership)
    return {sameLocalStorage ? ConversionKind::Identity : ConversionKind::LocalPermutation,
            sameLocalStorage ? "ownership and local slots agree"
                             : "ownership agrees but local slots differ"};

  executorLinear = 0;
  for (const auto &executor : enumerate(target.executorSpace)) {
    for (const auto &local : enumerate(target.localSpace)) {
      auto targetTile = target.ownership.apply(concatenate(executor, local));
      auto it = sourceOwners.find(key(targetTile));
      if (it == sourceOwners.end())
        return {ConversionKind::Unsupported, "source does not cover a required target point"};
      staysInSubgroup &= it->second / subgroupSize == executorLinear / subgroupSize;
    }
    ++executorLinear;
  }
  return {staysInSubgroup ? ConversionKind::SubgroupExchange
                          : ConversionKind::SharedMemoryExchange,
          staysInSubgroup ? "ownership changes within each subgroup"
                          : "ownership crosses subgroup boundaries"};
}

const char *toString(ConversionKind kind) {
  switch (kind) {
  case ConversionKind::Identity:
    return "identity";
  case ConversionKind::LocalPermutation:
    return "local-permutation";
  case ConversionKind::SubgroupExchange:
    return "subgroup-exchange";
  case ConversionKind::SharedMemoryExchange:
    return "shared-memory-exchange";
  case ConversionKind::Unsupported:
    return "unsupported";
  }
  return "unknown";
}

} // namespace ckl::core


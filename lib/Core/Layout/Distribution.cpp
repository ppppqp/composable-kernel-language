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
  std::vector<Axis> productAxes = distribution.executorSpace.axes();
  productAxes.insert(productAxes.end(), distribution.localSpace.axes().begin(),
                     distribution.localSpace.axes().end());
  if (!distribution.ownership.domain().sameShape(IndexSpace(productAxes)))
    return {false, false, false, 0, "ownership domain must be executor × local"};
  if (!distribution.ownership.codomain().sameShape(distribution.tileSpace))
    return {false, false, false, 0, "ownership codomain must match the tile"};

  if (!distribution.localStorage.domain().sameShape(distribution.localSpace))
    return {false, false, false, 0, "local-storage domain must match local space"};

  std::map<std::string, std::size_t> owners;
  for (const auto &executor : enumerate(distribution.executorSpace)) {
    for (const auto &local : enumerate(distribution.localSpace)) {
      auto tile = distribution.ownership.tryApply(concatenate(executor, local));
      if (tile)
        ++owners[key(*tile)];
    }
  }

  bool covering = owners.size() == static_cast<std::size_t>(distribution.tileSpace.volume());
  bool unique = true;
  std::size_t maximumReplication = 0;
  for (const auto &[unused, count] : owners) {
    (void)unused;
    unique &= count == 1;
    maximumReplication = std::max(maximumReplication, count);
  }
  bool valid = (!requireCovering || covering) &&
               (!requireUnique || unique || distribution.allowReplication);
  return {valid, covering, unique, maximumReplication,
          valid       ? "distribution satisfies requested invariants"
          : !covering ? "distribution does not cover its tile"
                      : "distribution has replicated owners"};
}

ConversionPlan classifyConversion(const Distribution &source, const Distribution &target,
                                  std::int64_t subgroupSize) {
  if (!sameTileShape(source, target))
    return {ConversionKind::Unsupported, "tile shapes differ", {}};
  const bool crossesScope = source.scope != target.scope;
  const bool requiresGlobal =
      crossesScope && (source.scope == ExecutionScope::Cluster ||
                       source.scope == ExecutionScope::Grid ||
                       target.scope == ExecutionScope::Cluster ||
                       target.scope == ExecutionScope::Grid);
  if (!source.executorSpace.sameShape(target.executorSpace) ||
      !source.localSpace.sameShape(target.localSpace))
    return {requiresGlobal ? ConversionKind::GlobalMemoryExchange
                             : ConversionKind::SharedMemoryExchange,
            requiresGlobal ? "execution scope and cardinality differ"
                             : "executor or per-agent cardinality differs", {}};

  bool sameOwnership = true;
  bool sameLocalStorage = true;
  bool staysInSubgroup = true;
  struct Owner {
    std::int64_t linear;
    std::vector<std::int64_t> executor;
    std::vector<std::int64_t> local;
  };
  std::map<std::string, Owner> sourceOwners;
  std::int64_t executorLinear = 0;
  for (const auto &executor : enumerate(source.executorSpace)) {
    for (const auto &local : enumerate(source.localSpace)) {
      auto input = concatenate(executor, local);
      auto sourceTile = source.ownership.tryApply(input);
      auto targetTile = target.ownership.tryApply(input);
      if (sourceTile)
        sourceOwners[key(*sourceTile)] = Owner{executorLinear, executor, local};
      sameOwnership &= sourceTile == targetTile;
      sameLocalStorage &= source.localStorage.apply(local) == target.localStorage.apply(local);
    }
    ++executorLinear;
  }

  if (sameOwnership) {
    std::vector<ConversionPlan::Move> moves;
    if (!sameLocalStorage || crossesScope) {
      for (const auto &executor : enumerate(source.executorSpace)) {
        for (const auto &local : enumerate(source.localSpace)) {
          auto tile = source.ownership.tryApply(concatenate(executor, local));
          if (tile)
            moves.push_back({*tile, executor, source.localStorage.apply(local), executor,
                             target.localStorage.apply(local)});
        }
      }
    }
    if (crossesScope)
      return {requiresGlobal ? ConversionKind::GlobalMemoryExchange
                             : ConversionKind::SharedMemoryExchange,
              requiresGlobal ? "ownership crosses workgroup execution scopes"
                             : "ownership crosses subgroup execution scope",
              std::move(moves)};
    return {sameLocalStorage ? ConversionKind::Identity : ConversionKind::LocalPermutation,
            sameLocalStorage ? "ownership and local slots agree"
                             : "ownership agrees but local slots differ", std::move(moves)};
  }

  std::vector<ConversionPlan::Move> moves;
  executorLinear = 0;
  for (const auto &executor : enumerate(target.executorSpace)) {
    for (const auto &local : enumerate(target.localSpace)) {
      auto targetTile = target.ownership.tryApply(concatenate(executor, local));
      if (!targetTile)
        continue;
      auto it = sourceOwners.find(key(*targetTile));
      if (it == sourceOwners.end())
        return {ConversionKind::Unsupported, "source does not cover a required target point", {}};
      staysInSubgroup &= it->second.linear / subgroupSize == executorLinear / subgroupSize;
      moves.push_back({*targetTile, it->second.executor,
                       source.localStorage.apply(it->second.local), executor,
                       target.localStorage.apply(local)});
    }
    ++executorLinear;
  }
  return {requiresGlobal ? ConversionKind::GlobalMemoryExchange
                           : crossesScope ? ConversionKind::SharedMemoryExchange
                           : staysInSubgroup ? ConversionKind::SubgroupExchange
                                             : ConversionKind::SharedMemoryExchange,
          requiresGlobal ? "ownership crosses workgroup execution scopes"
          : crossesScope ? "ownership crosses subgroup execution scope"
          : staysInSubgroup ? "ownership changes within each subgroup"
                            : "ownership crosses subgroup boundaries", std::move(moves)};
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
  case ConversionKind::GlobalMemoryExchange:
    return "global-memory-exchange";
  case ConversionKind::Unsupported:
    return "unsupported";
  }
  return "unknown";
}

} // namespace ckl::core

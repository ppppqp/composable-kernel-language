#include "ckl/Core/Planning/ExchangeSchedule.h"

#include <map>
#include <sstream>

namespace ckl::core {

ExchangeSchedule scheduleConversion(const ConversionPlan &plan) {
  ExchangeSchedule schedule{plan.kind, {}, 0, 0, false, {}};
  switch (plan.kind) {
  case ConversionKind::Identity:
    schedule.explanation = "no data movement required";
    return schedule;
  case ConversionKind::Unsupported:
    schedule.explanation = "conversion has no legal schedule";
    return schedule;
  case ConversionKind::LocalPermutation:
    for (const auto &move : plan.moves)
      schedule.steps.push_back({ExchangeStepKind::RegisterMove, move, std::nullopt});
    schedule.explanation = "per-executor register permutation";
    return schedule;
  case ConversionKind::SubgroupExchange:
    for (const auto &move : plan.moves)
      schedule.steps.push_back({ExchangeStepKind::SubgroupShuffle, move, std::nullopt});
    schedule.explanation = "subgroup shuffle network";
    return schedule;
  case ConversionKind::SharedMemoryExchange:
    break;
  case ConversionKind::GlobalMemoryExchange:
    // global stores -> kernel boundary -> global loads
    for (const auto &move : plan.moves)
      schedule.steps.push_back({ExchangeStepKind::GlobalStore, move, std::nullopt});
    schedule.steps.push_back({ExchangeStepKind::KernelBoundary, std::nullopt, std::nullopt});
    for (const auto &move : plan.moves)
      schedule.steps.push_back({ExchangeStepKind::GlobalLoad, move, std::nullopt});
    schedule.globalElements = static_cast<std::int64_t>(plan.moves.size());
    schedule.explanation = "global materialization across execution scopes";
    return schedule;
  }

  if (plan.moves.empty()) {
    schedule.explanation = "shared exchange requires remapping synthesis";
    return schedule;
  }

  std::map<std::vector<std::int64_t>, std::int64_t> offsets;
  for (const auto &move : plan.moves) {
    auto [it, inserted] = offsets.emplace(move.tile, static_cast<std::int64_t>(offsets.size()));
    (void)inserted;
    schedule.steps.push_back({ExchangeStepKind::SharedStore, move, it->second});
  }
  schedule.steps.push_back({ExchangeStepKind::Barrier, std::nullopt, std::nullopt});
  for (const auto &move : plan.moves)
    schedule.steps.push_back({ExchangeStepKind::SharedLoad, move, offsets.at(move.tile)});
  schedule.sharedElements = static_cast<std::int64_t>(offsets.size());
  schedule.requiresBarrier = true;
  schedule.explanation = "shared-memory transpose with one workgroup barrier";
  return schedule;
}

} // namespace ckl::core

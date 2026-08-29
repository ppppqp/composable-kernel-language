#pragma once

#include "ckl/Core/Layout/Distribution.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ckl::core {

enum class ExchangeStepKind {
  RegisterMove,
  SubgroupShuffle,
  SharedStore,
  Barrier,
  SharedLoad,
};

struct ExchangeStep {
  ExchangeStepKind kind;
  std::optional<ConversionPlan::Move> move;
  std::optional<std::int64_t> sharedOffset;
};

/*
Materializes a conversion plan into a sequence of exchange steps, which can be executed by a
kernel. The schedule may require a workgroup barrier if the conversion involves shared memory.
The schedule is not necessarily optimal, but it is guaranteed to be valid if the conversion plan is
valid.
*/
struct ExchangeSchedule {
  ConversionKind kind;
  std::vector<ExchangeStep> steps;
  std::int64_t sharedElements = 0;
  bool requiresBarrier = false;
  std::string explanation;
};

ExchangeSchedule scheduleConversion(const ConversionPlan &plan);

} // namespace ckl::core

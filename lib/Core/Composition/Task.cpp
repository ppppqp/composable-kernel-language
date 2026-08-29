#include "ckl/Core/Composition/Task.h"

#include <algorithm>

namespace ckl::core {
namespace {

const PortRealization *findPort(const std::vector<PortRealization> &ports,
                                const std::string &name) {
  auto it = std::find_if(ports.begin(), ports.end(),
                         [&](const PortRealization &port) { return port.name == name; });
  return it == ports.end() ? nullptr : &*it;
}

std::int64_t conversionCost(ConversionKind kind) {
  switch (kind) {
  case ConversionKind::Identity: return 0;
  case ConversionKind::LocalPermutation: return 10;
  case ConversionKind::SubgroupExchange: return 100;
  case ConversionKind::SharedMemoryExchange: return 1000;
  case ConversionKind::Unsupported: return 1'000'000;
  }
  return 1'000'000;
}

} // namespace

CompositionDecision selectComposition(const std::vector<TaskAlternative> &producers,
                                      const std::vector<TaskAlternative> &consumers,
                                      const std::string &producerPort,
                                      const std::string &consumerPort,
                                      std::int64_t subgroupSize,
                                      std::int64_t registerLimit,
                                      std::int64_t sharedMemoryLimit) {
  CompositionDecision decision;
  for (std::size_t p = 0; p < producers.size(); ++p) {
    for (std::size_t c = 0; c < consumers.size(); ++c) {
      const PortRealization *out = findPort(producers[p].outputs, producerPort);
      const PortRealization *in = findPort(consumers[c].inputs, consumerPort);
      if (!out || !in)
        continue;
      ConversionPlan plan = classifyConversion(out->distribution, in->distribution, subgroupSize);
      const bool resourcesFit =
          producers[p].registersPerThread + consumers[c].registersPerThread <= registerLimit &&
          producers[p].sharedMemoryBytes + consumers[c].sharedMemoryBytes <= sharedMemoryLimit;
      std::int64_t score = conversionCost(plan.kind);
      if (out->placement == Placement::Global || in->placement == Placement::Global)
        score += 100'000;
      if (!resourcesFit || plan.kind == ConversionKind::Unsupported)
        score = 1'000'000;
      std::string explanation = !resourcesFit
                                    ? "resource limit exceeded"
                                    : plan.kind == ConversionKind::Unsupported
                                          ? "no legal boundary conversion"
                                          : "legal candidate";
      CompositionCandidate candidate{p, c, std::move(plan), score, std::move(explanation)};
      decision.considered.push_back(candidate);
      if (score < 1'000'000 && (!decision.selected || score < decision.selected->score))
        decision.selected = candidate;
    }
  }
  return decision;
}

} // namespace ckl::core

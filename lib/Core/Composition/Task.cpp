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
  case ConversionKind::GlobalMemoryExchange: return 100'000;
  case ConversionKind::Unsupported: return 1'000'000;
  }
  return 1'000'000;
}

bool supports(const TaskAlternative &alternative,
              const std::vector<std::string> &availableCapabilities) {
  return std::all_of(alternative.requiredCapabilities.begin(),
                     alternative.requiredCapabilities.end(), [&](const std::string &required) {
    return std::find(availableCapabilities.begin(), availableCapabilities.end(), required) !=
           availableCapabilities.end();
  });
}

bool validLifetimes(const TaskAlternative &alternative) {
  return std::all_of(alternative.resources.begin(), alternative.resources.end(),
                     [](const ResourceLifetime &resource) {
    return resource.bytes >= 0 && resource.begin >= 0 && resource.end > resource.begin;
  });
}

} // namespace

CompositionDecision selectComposition(const std::vector<TaskAlternative> &producers,
                                      const std::vector<TaskAlternative> &consumers,
                                      const std::string &producerPort,
                                      const std::string &consumerPort,
                                      std::int64_t subgroupSize,
                                      std::int64_t registerLimit,
                                      std::int64_t sharedMemoryLimit,
                                      const std::vector<std::string> &availableCapabilities) {
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
      const bool capabilitiesFit = supports(producers[p], availableCapabilities) &&
                                   supports(consumers[c], availableCapabilities);
      const bool lifetimesValid = validLifetimes(producers[p]) && validLifetimes(consumers[c]);
      std::int64_t score = conversionCost(plan.kind);
      if (out->placement == Placement::Global || in->placement == Placement::Global)
        score += 100'000;
      if (!resourcesFit || !capabilitiesFit || !lifetimesValid ||
          plan.kind == ConversionKind::Unsupported)
        score = 1'000'000;
      std::string explanation = !resourcesFit
                                    ? "resource limit exceeded"
                                    : !capabilitiesFit
                                          ? "target capability unavailable"
                                          : !lifetimesValid
                                                ? "invalid resource lifetime"
                                    : plan.kind == ConversionKind::Unsupported
                                          ? "no legal boundary conversion"
                                          : "legal candidate";
      std::vector<std::string> provenance{
          "producer=" + producers[p].task + ":" + producers[p].name,
          "consumer=" + consumers[c].task + ":" + consumers[c].name,
          "conversion=" + std::string(toString(plan.kind)),
          "resources=" + std::to_string(producers[p].registersPerThread +
                                          consumers[c].registersPerThread) +
              " registers/thread"};
      CompositionCandidate candidate{p, c, std::move(plan), score, std::move(explanation),
                                     std::move(provenance)};
      decision.considered.push_back(candidate);
      if (score < 1'000'000 && (!decision.selected || score < decision.selected->score))
        decision.selected = candidate;
    }
  }
  return decision;
}

} // namespace ckl::core

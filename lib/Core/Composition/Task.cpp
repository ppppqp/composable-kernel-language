#include "ckl/Core/Composition/Task.h"

#include <algorithm>
#include <limits>

namespace ckl::core {
namespace {

// finds a name input or output port within an alternative
const PortRealization *findPort(const std::vector<PortRealization> &ports,
                                const std::string &name) {
  auto it = std::find_if(ports.begin(), ports.end(),
                         [&](const PortRealization &port) { return port.name == name; });
  return it == ports.end() ? nullptr : &*it;
}

// Cost model for conversion kinds. Can be calibrated by target extensions, but lower is always
// better.
std::int64_t conversionCost(ConversionKind kind) {
  switch (kind) {
  case ConversionKind::Identity:
    return 0;
  case ConversionKind::LocalPermutation:
    return 10;
  case ConversionKind::SubgroupExchange:
    return 100;
  case ConversionKind::SharedMemoryExchange:
    return 1000;
  case ConversionKind::GlobalMemoryExchange:
    return 100'000;
  case ConversionKind::Unsupported:
    return 1'000'000;
  }
  return 1'000'000;
}

// Returns true if the alternative's required capabilities are all present in the available set.
bool supports(const TaskAlternative &alternative,
              const std::vector<std::string> &availableCapabilities) {
  return std::all_of(alternative.requiredCapabilities.begin(),
                     alternative.requiredCapabilities.end(), [&](const std::string &required) {
                       return std::find(availableCapabilities.begin(), availableCapabilities.end(),
                                        required) != availableCapabilities.end();
                     });
}

// Returns true if all resource lifetimes in the alternative are valid.
// bytes >= 0, begin >= 0, end > begin
bool validLifetimes(const TaskAlternative &alternative) {
  return std::all_of(alternative.resources.begin(), alternative.resources.end(),
                     [](const ResourceLifetime &resource) {
                       return resource.bytes >= 0 && resource.begin >= 0 &&
                              resource.end > resource.begin;
                     });
}

// Returns true if the alternative is individually legal given the resource limits and available
// capabilities.
bool individuallyLegal(const TaskAlternative &alternative, std::int64_t registerLimit,
                       std::int64_t sharedMemoryLimit,
                       const std::vector<std::string> &availableCapabilities) {
  return alternative.registersPerThread >= 0 && alternative.registersPerThread <= registerLimit &&
         alternative.sharedMemoryBytes >= 0 && alternative.sharedMemoryBytes <= sharedMemoryLimit &&
         alternative.estimatedExecutionCost >= 0 && validLifetimes(alternative) &&
         supports(alternative, availableCapabilities);
}

// conversion-kind cost + global placement penalty if either side is global
// Even if the distribution happen to match, a port explicitly placed in global memory
// receives the additional penalty. This prevents an identity mapping from hiding a mandatory
// global-memory boundary
std::int64_t boundaryCost(const PortRealization &output, const PortRealization &input,
                          const ConversionPlan &conversion) {
  std::int64_t cost = conversionCost(conversion.kind);
  if (output.placement == Placement::Global || input.placement == Placement::Global)
    cost += 100'000;
  return cost;
}

// returns the stable implementation ID for a task alternative, falling back to task:name if the
// implementationId is empty
std::string implementationIdentity(const TaskAlternative &alternative) {
  return alternative.implementationId.empty() ? alternative.task + ":" + alternative.name
                                              : alternative.implementationId;
}

} // namespace

// Selects the best producer-consumer alternative pair given the resource limits, available
// capabilities, and subgroup size. Returns a CompositionDecision with the selected candidate and
// all considered candidates. If no legal candidate exists, selected will be empty.
// Pariwise optimization only, so no global optimality guarantee.
// For example, it chooses the best B alternative for A -> B, but does not consider B -> C during
// the selection.
CompositionDecision selectComposition(const std::vector<TaskAlternative> &producers,
                                      const std::vector<TaskAlternative> &consumers,
                                      const std::string &producerPort,
                                      const std::string &consumerPort, std::int64_t subgroupSize,
                                      std::int64_t registerLimit, std::int64_t sharedMemoryLimit,
                                      const std::vector<std::string> &availableCapabilities) {
  CompositionDecision decision;
  for (std::size_t p = 0; p < producers.size(); ++p) {
    for (std::size_t c = 0; c < consumers.size(); ++c) {
      // exhaustively consider all producer-consumer pairs.
      const PortRealization *out = findPort(producers[p].outputs, producerPort);
      const PortRealization *in = findPort(consumers[c].inputs, consumerPort);
      if (!out || !in)
        continue;
      // classify the conversion based on the source and target distributions
      ConversionPlan plan = classifyConversion(out->distribution, in->distribution, subgroupSize);
      const bool resourcesFit =
          producers[p].registersPerThread + consumers[c].registersPerThread <= registerLimit &&
          producers[p].sharedMemoryBytes + consumers[c].sharedMemoryBytes <= sharedMemoryLimit;
      const bool capabilitiesFit = supports(producers[p], availableCapabilities) &&
                                   supports(consumers[c], availableCapabilities);
      const bool lifetimesValid = validLifetimes(producers[p]) && validLifetimes(consumers[c]);
      const bool executionCostsValid =
          producers[p].estimatedExecutionCost >= 0 && consumers[c].estimatedExecutionCost >= 0;

      // compute the score for this candidate
      std::int64_t score = producers[p].estimatedExecutionCost +
                           consumers[c].estimatedExecutionCost + conversionCost(plan.kind);
      // Apply a penalty if either port is placed in global memory
      if (out->placement == Placement::Global || in->placement == Placement::Global)
        score += 100'000;
      if (!resourcesFit || !capabilitiesFit || !lifetimesValid || !executionCostsValid ||
          plan.kind == ConversionKind::Unsupported)
        score = 1'000'000;
      std::string explanation = !resourcesFit          ? "resource limit exceeded"
                                : !capabilitiesFit     ? "target capability unavailable"
                                : !lifetimesValid      ? "invalid resource lifetime"
                                : !executionCostsValid ? "invalid execution cost"
                                : plan.kind == ConversionKind::Unsupported
                                    ? "no legal boundary conversion"
                                    : "legal candidate";
      std::vector<std::string> provenance{
          "producer=" + implementationIdentity(producers[p]),
          "consumer=" + implementationIdentity(consumers[c]),
          "conversion=" + std::string(toString(plan.kind)),
          "execution-cost=" + std::to_string(producers[p].estimatedExecutionCost +
                                             consumers[c].estimatedExecutionCost),
          "resources=" +
              std::to_string(producers[p].registersPerThread + consumers[c].registersPerThread) +
              " registers/thread"};
      CompositionCandidate candidate{
          p, c, std::move(plan), score, std::move(explanation), std::move(provenance)};
      decision.considered.push_back(candidate);
      const bool legal = resourcesFit && capabilitiesFit && lifetimesValid && executionCostsValid &&
                         candidate.conversion.kind != ConversionKind::Unsupported;
      if (legal && (!decision.selected || score < decision.selected->score)) {
        // select the best legal candidate
        decision.selected = candidate;
      }
    }
  }
  return decision;
}

/*
Graph search for the best linear pipeline through a sequence of stages, each with multiple
alternatives.
Assume K alternatives per stage and N stages, the complexity is O(N*K^2)
TODO: does not handle
- branches (non-linear pipelines)
- fusion partitions (e.g. A -> B -> C and A -> D -> C)
- combined fused resource pressure
- overlapping lifetimes
- effects that prohibits fusion
- user pins/partial constraints
*/
PipelineDecision selectLinearPipeline(const std::vector<PipelineStage> &stages,
                                      std::int64_t subgroupSize, std::int64_t registerLimit,
                                      std::int64_t sharedMemoryLimit,
                                      const std::vector<std::string> &availableCapabilities) {
  PipelineDecision decision;
  if (stages.empty()) {
    decision.selected = PipelineSelection{};
    return decision;
  }

  constexpr std::int64_t unreachable = std::numeric_limits<std::int64_t>::max();

  // The minimum total cost of reaching this alternative at this stage from the beginning of the
  // pipeline.
  std::vector<std::vector<std::int64_t>> costs(stages.size());
  std::vector<std::vector<std::size_t>> predecessors(stages.size());
  std::vector<std::vector<ConversionPlan>> incomingPlans(stages.size());
  for (std::size_t stage = 0; stage < stages.size(); ++stage) {
    const auto count = stages[stage].alternatives.size();
    costs[stage].assign(count, unreachable);
    predecessors[stage].assign(count, 0);
    incomingPlans[stage].resize(count);
  }

  for (std::size_t candidate = 0; candidate < stages.front().alternatives.size(); ++candidate) {
    const auto &alternative = stages.front().alternatives[candidate];
    if (individuallyLegal(alternative, registerLimit, sharedMemoryLimit, availableCapabilities))
      // initialize the first stage with the cost of each legal alternative.
      costs.front()[candidate] = alternative.estimatedExecutionCost;
  }

  for (std::size_t stage = 1; stage < stages.size(); ++stage) {
    for (std::size_t current = 0; current < stages[stage].alternatives.size(); ++current) {
      // for each stage, consider each alternative and find the best predecessor from the previous
      // stage
      const auto &currentAlternative = stages[stage].alternatives[current];
      if (!individuallyLegal(currentAlternative, registerLimit, sharedMemoryLimit,
                             availableCapabilities))
        continue;
      const PortRealization *input = findPort(currentAlternative.inputs, stages[stage].inputPort);
      if (!input)
        continue;
      for (std::size_t previous = 0; previous < stages[stage - 1].alternatives.size(); ++previous) {
        if (costs[stage - 1][previous] == unreachable)
          continue;
        const auto &previousAlternative = stages[stage - 1].alternatives[previous];
        const PortRealization *output =
            findPort(previousAlternative.outputs, stages[stage - 1].outputPort);
        if (!output)
          continue;
        ConversionPlan conversion =
            classifyConversion(output->distribution, input->distribution, subgroupSize);
        if (conversion.kind == ConversionKind::Unsupported)
          continue;
        const std::int64_t edgeCost = boundaryCost(*output, *input, conversion);
        if (costs[stage - 1][previous] >
            unreachable - currentAlternative.estimatedExecutionCost - edgeCost)
          continue;

        // new cost = best cost reaching previous alternative + current implementation cost +
        // boundary cost
        const std::int64_t score =
            costs[stage - 1][previous] + currentAlternative.estimatedExecutionCost + edgeCost;
        if (score < costs[stage][current]) {
          costs[stage][current] = score;
          predecessors[stage][current] = previous;
          incomingPlans[stage][current] = std::move(conversion);
        }
      }
    }
  }

  auto best = std::min_element(costs.back().begin(), costs.back().end());
  if (best == costs.back().end() || *best == unreachable) {
    decision.diagnostics.push_back("no legal realization path through linear task pipeline");
    return decision;
  }

  PipelineSelection selection;
  selection.score = *best;
  selection.alternatives.resize(stages.size());
  selection.conversions.resize(stages.size() - 1);
  std::size_t candidate = std::distance(costs.back().begin(), best);
  for (std::size_t stage = stages.size(); stage-- > 0;) {
    // backtrack through the predecessors to find the selected alternative for each stage
    selection.alternatives[stage] = candidate;
    const auto &alternative = stages[stage].alternatives[candidate];
    selection.provenance.push_back("invocation=" + stages[stage].invocation +
                                   ",implementation=" + implementationIdentity(alternative));
    if (stage != 0) {
      selection.conversions[stage - 1] = incomingPlans[stage][candidate];
      candidate = predecessors[stage][candidate];
    }
  }
  std::reverse(selection.provenance.begin(), selection.provenance.end());
  decision.selected = std::move(selection);
  return decision;
}

} // namespace ckl::core

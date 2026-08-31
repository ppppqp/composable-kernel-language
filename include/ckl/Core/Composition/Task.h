#pragma once

#include "ckl/Core/Layout/Distribution.h"

#include <optional>
#include <string>
#include <vector>

namespace ckl::core {

enum class Placement { Private, Shared, Global };
enum class EffectKind { Read, Write, Consume, ChannelPut, ChannelGet, ChannelRelease };
enum class AlternativeOrigin { Unspecified, User, Compiler, Extension, Library };

// For interference detection between overlapping resources
struct ResourceLifetime {
  std::string name;
  std::int64_t bytes;
  std::int64_t begin;
  std::int64_t end;
};

struct TaskEffect {
  EffectKind kind;
  std::string resource;
  std::int64_t stage = 0;
};

struct PortRealization {
  std::string name;
  Distribution distribution;
  Placement placement;
  std::int64_t vectorWidth = 1;
};

// A task alternative is a specific implementation of a task, e.g. a kernel variant or a
// function specialization. It has a name, a set of input and output ports, and resource
// requirements (registers and shared memory). The input and output ports are realized with
// specific distributions and placements, which may differ from the task's logical ports.
struct TaskAlternative {
  std::string task;
  std::string name;
  std::vector<PortRealization> inputs;
  std::vector<PortRealization> outputs;
  std::int64_t registersPerThread = 0;
  std::int64_t sharedMemoryBytes = 0;
  std::vector<std::string> requiredCapabilities; // e.g. "mma", "async-copy", "wgmma"
  std::vector<ResourceLifetime> resources;
  std::vector<TaskEffect> effects;
  // Target-independent relative estimate used for bounded planning. Target
  // extensions may calibrate this scale, but lower is always better.
  std::int64_t estimatedExecutionCost = 0;
  // Stable identity for provenance and cache keys. When empty, planners use
  // task:name as the deterministic fallback.
  std::string implementationId;
  AlternativeOrigin origin = AlternativeOrigin::Unspecified;
  // Optional callable symbol implementing this fixed realization. Derived
  // planning alternatives may remain bodyless until selection is committed.
  std::string implementationSymbol;
};

struct CompositionCandidate {
  std::size_t producerAlternative;
  std::size_t consumerAlternative;
  ConversionPlan conversion;
  std::int64_t score;
  std::string explanation;
  std::vector<std::string> provenance;
};

struct CompositionDecision {
  std::optional<CompositionCandidate> selected;
  std::vector<CompositionCandidate> considered;
};

CompositionDecision selectComposition(const std::vector<TaskAlternative> &producers,
                                      const std::vector<TaskAlternative> &consumers,
                                      const std::string &producerPort,
                                      const std::string &consumerPort, std::int64_t subgroupSize,
                                      std::int64_t registerLimit, std::int64_t sharedMemoryLimit,
                                      const std::vector<std::string> &availableCapabilities = {});

struct PipelineStage {
  std::string invocation;
  std::vector<TaskAlternative> alternatives;
  std::string inputPort;
  std::string outputPort;
};

struct PipelineSelection {
  std::vector<std::size_t> alternatives;
  std::vector<ConversionPlan> conversions;
  std::int64_t score = 0;
  std::vector<std::string> provenance;
};

struct PipelineDecision {
  std::optional<PipelineSelection> selected;
  std::vector<std::string> diagnostics;
};

// Globally selects a realization for each stage in a linear task pipeline.
// Each boundary connects stages[i].outputPort to stages[i + 1].inputPort.
PipelineDecision selectLinearPipeline(
    const std::vector<PipelineStage> &stages, std::int64_t subgroupSize,
    std::int64_t registerLimit, std::int64_t sharedMemoryLimit,
    const std::vector<std::string> &availableCapabilities = {});

struct TaskGraphNode {
  std::string invocation;
  std::vector<TaskAlternative> alternatives;
};

struct TaskGraphEdge {
  std::size_t producer;
  std::size_t consumer;
  std::string producerPort;
  std::string consumerPort;
};

struct TaskGraphSelection {
  std::vector<std::size_t> alternatives;
  std::vector<ConversionPlan> conversions;
  std::int64_t score = 0;
  std::size_t combinationsExplored = 0;
  std::vector<std::string> provenance;
};

struct TaskGraphDecision {
  std::optional<TaskGraphSelection> selected;
  std::size_t combinationsExplored = 0;
  bool searchLimitReached = false;
  std::vector<std::string> diagnostics;
};

// Deterministically explores the bounded Cartesian product of node
// alternatives. Execution cost is charged once per node and conversion cost
// once per edge, which makes fan-out decisions globally visible.
TaskGraphDecision selectTaskGraph(
    const std::vector<TaskGraphNode> &nodes, const std::vector<TaskGraphEdge> &edges,
    std::int64_t subgroupSize, std::int64_t registerLimit,
    std::int64_t sharedMemoryLimit, std::size_t maximumCombinations,
    const std::vector<std::string> &availableCapabilities = {});

} // namespace ckl::core

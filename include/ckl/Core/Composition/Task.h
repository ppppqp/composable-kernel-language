#pragma once

#include "ckl/Core/Layout/Distribution.h"

#include <optional>
#include <string>
#include <vector>

namespace ckl::core {

enum class Placement { Private, Shared, Global };
enum class EffectKind { Read, Write, Consume, ChannelPut, ChannelGet, ChannelRelease };

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

} // namespace ckl::core

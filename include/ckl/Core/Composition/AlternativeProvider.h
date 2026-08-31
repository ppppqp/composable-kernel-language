#pragma once

#include "ckl/Core/Composition/Task.h"

#include <string>
#include <vector>

namespace ckl::core {

// Target-neutral description passed to a bounded candidate provider. Target
// extensions keep instruction-specific knobs internally and return only
// generic task contracts, distributions, resources, costs, and capabilities.
struct TaskAlternativeRequest {
  std::string task;
  std::vector<std::string> inputPorts;
  std::vector<std::string> outputPorts;
  std::string target;
  std::string architecture;
  std::vector<std::string> availableCapabilities;
};

class TaskAlternativeProvider {
public:
  virtual ~TaskAlternativeProvider() = default;
  virtual std::string providerId() const = 0;
  virtual std::vector<TaskAlternative> enumerate(const TaskAlternativeRequest &request) const = 0;
};

// Target-neutral collection of task alternatives with diagnostics for invalid candidates.
struct AlternativeCollection {
  std::vector<TaskAlternative> alternatives;
  std::vector<std::string> diagnostics;
};

// Calls providers in the supplied order and validates their output. Invalid
// candidates are diagnosed and omitted; valid candidates retain stable order.
AlternativeCollection
collectTaskAlternatives(const TaskAlternativeRequest &request,
                        const std::vector<const TaskAlternativeProvider *> &providers);

// Providers are owned by their extension libraries and registered during
// driver initialization. Duplicate identifiers are rejected.
bool registerTaskAlternativeProvider(const TaskAlternativeProvider &provider);
const TaskAlternativeProvider *findTaskAlternativeProvider(const std::string &providerId);

} // namespace ckl::core

#include "ckl/Core/Composition/AlternativeProvider.h"

#include <exception>
#include <unordered_set>

namespace ckl::core {
namespace {

bool validatePorts(const std::vector<PortRealization> &ports,
                   const std::vector<std::string> &expected, const std::string &kind,
                   const std::string &candidate, std::vector<std::string> &diagnostics) {
  if (ports.size() != expected.size()) {
    diagnostics.push_back(candidate + " has " + std::to_string(ports.size()) + " " + kind +
                          " ports; expected " + std::to_string(expected.size()));
    return false;
  }
  for (std::size_t index = 0; index < ports.size(); ++index) {
    if (ports[index].name != expected[index]) {
      diagnostics.push_back(candidate + " names positional " + kind + " port " +
                            std::to_string(index) + " '" + ports[index].name +
                            "'; expected '" + expected[index] + "'");
      return false;
    }
    if (ports[index].vectorWidth <= 0) {
      diagnostics.push_back(candidate + " has a non-positive vector width on " + kind +
                            " port '" + ports[index].name + "'");
      return false;
    }
    try {
      DistributionCheck check = verifyDistribution(ports[index].distribution);
      if (!check.valid) {
        diagnostics.push_back(candidate + " has an invalid distribution on " + kind +
                              " port '" + ports[index].name + "': " + check.message);
        return false;
      }
    } catch (const std::exception &error) {
      diagnostics.push_back(candidate + " distribution validation failed: " + error.what());
      return false;
    }
  }
  return true;
}

} // namespace

AlternativeCollection collectTaskAlternatives(
    const TaskAlternativeRequest &request,
    const std::vector<const TaskAlternativeProvider *> &providers) {
  AlternativeCollection result;
  std::unordered_set<std::string> names;
  std::unordered_set<std::string> implementationIds;
  for (const TaskAlternativeProvider *provider : providers) {
    if (!provider) {
      result.diagnostics.push_back("ignored null task-alternative provider");
      continue;
    }
    const std::string id = provider->providerId();
    if (id.empty()) {
      result.diagnostics.push_back("ignored task-alternative provider with an empty identifier");
      continue;
    }
    std::vector<TaskAlternative> candidates;
    try {
      candidates = provider->enumerate(request);
    } catch (const std::exception &error) {
      result.diagnostics.push_back("provider '" + id + "' failed: " + error.what());
      continue;
    }
    for (TaskAlternative &candidate : candidates) {
      const std::string label = id + ":" + (candidate.name.empty() ? "<unnamed>" : candidate.name);
      if (candidate.name.empty()) {
        result.diagnostics.push_back(label + " has an empty alternative name");
        continue;
      }
      if (!candidate.task.empty() && candidate.task != request.task) {
        result.diagnostics.push_back(label + " belongs to task '" + candidate.task +
                                     "', not requested task '" + request.task + "'");
        continue;
      }
      candidate.task = request.task;
      if (candidate.origin == AlternativeOrigin::Unspecified)
        candidate.origin = AlternativeOrigin::Extension;
      if (candidate.implementationId.empty())
        candidate.implementationId = id + ":" + request.task + ":" + candidate.name;
      if (!validatePorts(candidate.inputs, request.inputPorts, "input", label,
                         result.diagnostics) ||
          !validatePorts(candidate.outputs, request.outputPorts, "output", label,
                         result.diagnostics))
        continue;
      if (!names.insert(candidate.name).second) {
        result.diagnostics.push_back(label + " duplicates alternative name '" + candidate.name +
                                     "'");
        continue;
      }
      if (!implementationIds.insert(candidate.implementationId).second) {
        result.diagnostics.push_back(label + " duplicates implementation identifier '" +
                                     candidate.implementationId + "'");
        continue;
      }
      result.alternatives.push_back(std::move(candidate));
    }
  }
  return result;
}

} // namespace ckl::core

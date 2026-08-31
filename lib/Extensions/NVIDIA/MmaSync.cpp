#include "ckl/Extensions/NVIDIA/MmaSync.h"

namespace ckl::extensions::nvidia {
namespace {

// Cross-check these maps against upstream MLIR's
// MmaSyncBuilder::{m16n8k16f16Lhs,m16n8k16f16Rhs,m16n8k16f16Res} in
// mlir/lib/Dialect/NVGPU/TransformOps/NVGPUTransformOps.cpp.

core::IndexExpr lane() { return core::IndexExpr::input(0); }
core::IndexExpr group() {
  return core::IndexExpr::floorDiv(lane(), 4);
}
core::IndexExpr threadInGroup() {
  return core::IndexExpr::modulo(lane(), 4);
}
core::IndexExpr scaled(core::IndexExpr value, std::int64_t factor) {
  return core::IndexExpr::multiply(std::move(value), core::IndexExpr::constant(factor));
}
core::IndexExpr sum(core::IndexExpr lhs, core::IndexExpr rhs) {
  return core::IndexExpr::add(std::move(lhs), std::move(rhs));
}

core::Distribution makeDistribution(core::IndexSpace local, core::IndexSpace tile,
                                    std::vector<core::IndexExpr> results) {
  core::IndexSpace executor({{"lane", 32}});
  std::vector<core::Axis> domainAxes = executor.axes();
  domainAxes.insert(domainAxes.end(), local.axes().begin(), local.axes().end());
  core::IndexSpace domain(std::move(domainAxes));
  return {executor, local, tile, core::IndexMap(domain, tile, std::move(results)),
          core::IndexMap::strided(local, [&] {
            std::vector<std::int64_t> strides(local.rank(), 1);
            for (std::size_t i = local.rank(); i-- > 1;)
              strides[i - 1] = strides[i] * local.axes()[i].extent;
            return strides;
          }()),
          false, core::ExecutionScope::Subgroup};
}

} // namespace

core::Distribution makeMmaSyncM16N8K16F16LhsDistribution() {
  // local inputs: kHalf, rowHalf, pair.
  core::IndexSpace local({{"kHalf", 2}, {"rowHalf", 2}, {"pair", 2}});
  core::IndexExpr row = sum(group(), scaled(core::IndexExpr::input(2), 8));
  core::IndexExpr column =
      sum(sum(scaled(threadInGroup(), 2), core::IndexExpr::input(3)),
          scaled(core::IndexExpr::input(1), 8));
  return makeDistribution(std::move(local), core::IndexSpace({{"m", 16}, {"k", 16}}),
                          {std::move(row), std::move(column)});
}

core::Distribution makeMmaSyncM16N8K16F16RhsDistribution() {
  // local inputs: kHalf, pair.
  core::IndexSpace local({{"kHalf", 2}, {"pair", 2}});
  core::IndexExpr row =
      sum(sum(scaled(threadInGroup(), 2), core::IndexExpr::input(2)),
          scaled(core::IndexExpr::input(1), 8));
  return makeDistribution(std::move(local), core::IndexSpace({{"k", 16}, {"n", 8}}),
                          {std::move(row), group()});
}

core::Distribution makeMmaSyncM16N8K16F32AccumulatorDistribution() {
  // local inputs: rowHalf, pair.
  core::IndexSpace local({{"rowHalf", 2}, {"pair", 2}});
  core::IndexExpr row = sum(group(), scaled(core::IndexExpr::input(1), 8));
  core::IndexExpr column = sum(scaled(threadInGroup(), 2), core::IndexExpr::input(2));
  return makeDistribution(std::move(local), core::IndexSpace({{"m", 16}, {"n", 8}}),
                          {std::move(row), std::move(column)});
}

std::string MmaSyncF16F32Provider::providerId() const {
  return "nvidia.mma-sync.f16-f32";
}

std::vector<core::TaskAlternative>
MmaSyncF16F32Provider::enumerate(const core::TaskAlternativeRequest &request) const {
  if (request.target != "nvidia" ||
      request.inputPorts != std::vector<std::string>{"lhs", "rhs", "acc"} ||
      request.outputPorts != std::vector<std::string>{"result"})
    return {};
  core::TaskAlternative alternative;
  alternative.name = "mma-sync-m16n8k16-f16-f32";
  alternative.inputs = {
      {"lhs", makeMmaSyncM16N8K16F16LhsDistribution(), core::Placement::Private, 2},
      {"rhs", makeMmaSyncM16N8K16F16RhsDistribution(), core::Placement::Private, 2},
      {"acc", makeMmaSyncM16N8K16F32AccumulatorDistribution(), core::Placement::Private, 1}};
  alternative.outputs = {{"result", makeMmaSyncM16N8K16F32AccumulatorDistribution(),
                          core::Placement::Private, 1}};
  alternative.registersPerThread = 10;
  alternative.requiredCapabilities = {"nvidia.mma.sync.m16n8k16.f16"};
  alternative.estimatedExecutionCost = 1;
  alternative.implementationId = "nvidia.mma.sync.m16n8k16.row.col.f32.f16.f16.f32";
  alternative.origin = core::AlternativeOrigin::Extension;
  return {std::move(alternative)};
}

void registerAlternativeProviders() {
  static MmaSyncF16F32Provider provider;
  (void)core::registerTaskAlternativeProvider(provider);
}

} // namespace ckl::extensions::nvidia

#include "ckl/Extensions/AMD/MfmaLayouts.h"

#include <stdexcept>

namespace ckl::extensions::amd {

core::Distribution makeMfmaAccumulatorDistribution(MfmaAccumulatorParameters p) {
  if (p.m0PerLane <= 0 || p.mLane <= 0 || p.m1PerLane <= 0 || p.nLane <= 0 ||
      p.mLane * p.nLane != 64)
    throw std::invalid_argument("AMD MFMA accumulator parameters must describe one 64-lane wave");

  core::IndexSpace executor({{"lane", 64}});
  core::IndexSpace local({{"m0", p.m0PerLane}, {"m1", p.m1PerLane}});
  core::IndexSpace tile({{"m", p.m0PerLane * p.mLane * p.m1PerLane}, {"n", p.nLane}});
  core::IndexSpace executorLocal(
      {{"lane", 64}, {"m0", p.m0PerLane}, {"m1", p.m1PerLane}});

  core::IndexExpr lane = core::IndexExpr::input(0);
  core::IndexExpr m0 = core::IndexExpr::input(1);
  core::IndexExpr m1 = core::IndexExpr::input(2);
  core::IndexExpr laneM = core::IndexExpr::floorDiv(lane, p.nLane);
  core::IndexExpr laneN = core::IndexExpr::modulo(lane, p.nLane);
  core::IndexExpr m = core::IndexExpr::add(
      core::IndexExpr::multiply(m0, core::IndexExpr::constant(p.mLane * p.m1PerLane)),
      core::IndexExpr::add(
          core::IndexExpr::multiply(laneM, core::IndexExpr::constant(p.m1PerLane)), m1));
  core::IndexMap ownership(executorLocal, tile, {m, laneN});
  core::IndexMap localStorage = core::IndexMap::strided(local, {p.m1PerLane, 1});
  return {executor, local, tile, ownership, localStorage, false, core::ExecutionScope::Subgroup};
}

} // namespace ckl::extensions::amd

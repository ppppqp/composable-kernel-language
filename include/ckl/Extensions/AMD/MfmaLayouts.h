#pragma once

#include "ckl/Core/Layout/Distribution.h"

namespace ckl::extensions::amd {

// Parameters used by AMD MFMA C-tile distribution encodings. The logical M factors are
// [m0PerLane, mLane, m1PerLane], N is [nLane], and the 64-lane wave owns the lane factors.
struct MfmaAccumulatorParameters {
  std::int64_t m0PerLane;
  std::int64_t mLane;
  std::int64_t m1PerLane;
  std::int64_t nLane;
};

core::Distribution makeMfmaAccumulatorDistribution(MfmaAccumulatorParameters parameters);

} // namespace ckl::extensions::amd

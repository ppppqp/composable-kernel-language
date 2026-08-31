#pragma once

#include "ckl/Core/Composition/AlternativeProvider.h"

namespace ckl::extensions::nvidia {

// Register-fragment distributions for
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.
core::Distribution makeMmaSyncM16N8K16F16LhsDistribution();
core::Distribution makeMmaSyncM16N8K16F16RhsDistribution();
core::Distribution makeMmaSyncM16N8K16F32AccumulatorDistribution();

class MmaSyncF16F32Provider final : public core::TaskAlternativeProvider {
public:
  std::string providerId() const override;
  std::vector<core::TaskAlternative>
  enumerate(const core::TaskAlternativeRequest &request) const override;
};

void registerAlternativeProviders();

} // namespace ckl::extensions::nvidia

#include "ckl/Core/Layout/StorageLayout.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>

namespace ckl::core {

StorageCheck verifyStorageLayout(const StorageLayout &layout) {
  if (!layout.address.domain().sameShape(layout.logicalSpace))
    return {false, false, 0, 0, "address-map domain does not match logical space"};
  if (layout.address.codomain().rank() != 1)
    return {false, false, 0, 0, "address map must produce one coordinate"};
  if (layout.alignment <= 0 || (layout.alignment & (layout.alignment - 1)) != 0)
    return {false, false, 0, 0, "alignment must be a positive power of two"};

  std::map<std::int64_t, std::size_t> uses;
  std::int64_t minimum = 0;
  std::int64_t maximum = 0;

  // The first flag is used to detect the first point, so that we can
  // initialize the minimum and maximum addresses correctly.
  bool first = true;
  for (const auto &point : enumerate(layout.logicalSpace)) {
    auto address = layout.address.tryApply(point);
    if (!address)
      continue;
    const std::int64_t value = address->front();
    ++uses[value];
    minimum = first ? value : std::min(minimum, value);
    maximum = first ? value : std::max(maximum, value);
    first = false;
  }
  bool injective = true;
  for (const auto &[address, count] : uses) {
    (void)address;
    injective &= count == 1;
  }
  const bool valid = injective || layout.aliasPolicy != AliasPolicy::Unique;

  // The inBounds flag checks whether the maximum address is within the allocated elements, if
  // specified.
  const bool inBounds = !layout.allocationElements || first || maximum < *layout.allocationElements;
  return {valid && inBounds, injective, minimum, maximum,
          !valid      ? "unique storage layout aliases"
          : !inBounds ? "storage layout exceeds its allocation"
                      : "storage layout satisfies alias and allocation policy"};
}

// The bank conflict analysis is a simple pointwise simulation of the layout's address map.
// It counts how many points map to each bank, and reports the maximum count.
// TODO: make it AddressUnit-aware
// TODO: make it more precise by considering the actual address values, not just the bank indices.
// since several lanes might read the same address by broadcasting, which is not a conflict.
BankConflictReport
analyzeBankConflicts(const StorageLayout &layout,
                     const std::vector<std::vector<std::int64_t>> &simultaneousPoints,
                     std::size_t bankCount, std::size_t elementsPerBankUnit) {
  if (bankCount == 0 || elementsPerBankUnit == 0)
    throw std::invalid_argument("bank geometry must be nonzero");
  BankConflictReport report{0, std::vector<std::size_t>(bankCount)};
  for (const auto &point : simultaneousPoints) {
    auto address = layout.address.tryApply(point);
    if (!address)
      continue;
    // only need the first coordinate of the address, since the codomain is 1-D
    const auto bank = static_cast<std::size_t>(address->front() / elementsPerBankUnit) % bankCount;
    report.maximumConflict = std::max(report.maximumConflict, ++report.bankUseCounts[bank]);
  }
  return report;
}

std::string serialize(const StorageLayout &layout) {
  std::ostringstream os;
  os << "storage<space=" << layout.logicalSpace.str() << ", address=" << layout.address.str()
     << ", alignment=" << layout.alignment << ", alias=" << static_cast<int>(layout.aliasPolicy)
     << '>';
  return os.str();
}

} // namespace ckl::core

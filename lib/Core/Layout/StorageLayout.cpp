#include "ckl/Core/Layout/StorageLayout.h"

#include <algorithm>
#include <map>
#include <sstream>

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
  return {valid, injective, minimum, maximum,
          valid ? "storage layout satisfies alias policy" : "unique storage layout aliases"};
}

std::string serialize(const StorageLayout &layout) {
  std::ostringstream os;
  os << "storage<space=" << layout.logicalSpace.str() << ", address=" << layout.address.str()
     << ", alignment=" << layout.alignment << ", alias="
     << static_cast<int>(layout.aliasPolicy) << '>';
  return os.str();
}

} // namespace ckl::core


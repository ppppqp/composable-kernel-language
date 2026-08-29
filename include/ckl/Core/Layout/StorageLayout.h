#pragma once

#include "ckl/Core/Layout/IndexMap.h"

#include <optional>
#include <string>
#include <vector>

namespace ckl::core {

enum class AddressSpace { Global, Shared, Private, Target };
enum class AddressUnit { Element, Byte };
enum class AliasPolicy { Unique, ReadOnlyAliasing, Reduction };

struct StorageLayout {
  IndexSpace logicalSpace;
  AddressSpace addressSpace;
  AddressUnit addressUnit;
  IndexMap address;
  std::int64_t alignment;
  AliasPolicy aliasPolicy;
  std::optional<std::string> aliasClass;

  // optional number of elements allocated for this storage. If not specified, the allocation is
  // assumed to be unbounded. If specified, the storage layout is valid only if all addresses are
  // less than this number.
  std::optional<std::int64_t> allocationElements = std::nullopt;
};

struct StorageCheck {
  bool valid;
  bool injective;
  std::int64_t minimumAddress;
  std::int64_t maximumAddress;
  std::string message;
};

StorageCheck verifyStorageLayout(const StorageLayout &layout);

struct BankConflictReport {
  std::size_t maximumConflict;
  std::vector<std::size_t> bankUseCounts;
};

BankConflictReport
analyzeBankConflicts(const StorageLayout &layout,
                     const std::vector<std::vector<std::int64_t>> &simultaneousPoints,
                     std::size_t bankCount, std::size_t elementsPerBankUnit = 1);
std::string serialize(const StorageLayout &layout);

} // namespace ckl::core

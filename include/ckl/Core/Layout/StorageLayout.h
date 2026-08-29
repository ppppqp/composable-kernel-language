#pragma once

#include "ckl/Core/Layout/IndexMap.h"

#include <optional>
#include <string>

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
};

struct StorageCheck {
  bool valid;
  bool injective;
  std::int64_t minimumAddress;
  std::int64_t maximumAddress;
  std::string message;
};

StorageCheck verifyStorageLayout(const StorageLayout &layout);
std::string serialize(const StorageLayout &layout);

} // namespace ckl::core


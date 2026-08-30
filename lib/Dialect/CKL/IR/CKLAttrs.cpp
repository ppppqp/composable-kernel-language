#include "ckl/Dialect/CKL/IR/CKLAttrs.h"

#include "ckl/Core/Layout/IndexMap.h"
#include "ckl/Core/Layout/Distribution.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include <sstream>

using namespace mlir;
using namespace mlir::ckl;

#define GET_ATTRDEF_CLASSES
#include "ckl/Dialect/CKL/IR/CKLAttrs.cpp.inc"

void CKLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ckl/Dialect/CKL/IR/CKLAttrs.cpp.inc"
      >();
}

namespace {

template <typename AttrType>
FailureOr<AttrType> parseQualifiedAttribute(AsmParser &parser, StringRef description) {
  Attribute attribute;
  if (failed(parser.parseAttribute(attribute)))
    return failure();
  auto typed = mlir::dyn_cast<AttrType>(attribute);
  if (!typed) {
    parser.emitError(parser.getCurrentLocation()) << "expected " << description;
    return failure();
  }
  return typed;
}

LogicalResult parseNamedField(AsmParser &parser, StringRef name) {
  return failure(parser.parseKeyword(name) || parser.parseEqual());
}

const char *scopeName(::ckl::core::ExecutionScope scope) {
  switch (scope) {
  case ::ckl::core::ExecutionScope::Subgroup: return "subgroup";
  case ::ckl::core::ExecutionScope::Workgroup: return "workgroup";
  case ::ckl::core::ExecutionScope::Cluster: return "cluster";
  case ::ckl::core::ExecutionScope::Grid: return "grid";
  }
  return "unknown";
}

FailureOr<::ckl::core::ExecutionScope> parseScope(AsmParser &parser) {
  StringRef value;
  if (failed(parser.parseKeyword(&value)))
    return failure();
  if (value == "subgroup") return ::ckl::core::ExecutionScope::Subgroup;
  if (value == "workgroup") return ::ckl::core::ExecutionScope::Workgroup;
  if (value == "cluster") return ::ckl::core::ExecutionScope::Cluster;
  if (value == "grid") return ::ckl::core::ExecutionScope::Grid;
  parser.emitError(parser.getCurrentLocation()) << "unknown CKL execution scope '" << value << "'";
  return failure();
}

} // namespace

Attribute SpaceAttr::parse(AsmParser &parser, Type) {
  SMLoc location = parser.getCurrentLocation();
  if (failed(parser.parseLess()) || failed(parser.parseLSquare()))
    return {};
  std::vector<::ckl::core::Axis> axes;
  if (failed(parser.parseOptionalRSquare())) {
    while (true) {
      StringRef name;
      if (failed(parser.parseKeyword(&name)) || failed(parser.parseEqual()))
        return {};
      std::int64_t extent;
      OptionalParseResult parsedExtent = parser.parseOptionalInteger(extent);
      if (parsedExtent.has_value()) {
        if (failed(*parsedExtent))
          return {};
        axes.emplace_back(name.str(), extent);
      } else {
        std::string symbol;
        if (failed(parser.parseKeyword("symbol")) || failed(parser.parseLess()) ||
            failed(parser.parseString(&symbol)) || failed(parser.parseGreater()))
          return {};
        axes.push_back(::ckl::core::Axis::symbolic(name.str(), std::move(symbol)));
      }
      if (succeeded(parser.parseOptionalComma()))
        continue;
      if (failed(parser.parseRSquare()))
        return {};
      break;
    }
  }
  if (failed(parser.parseGreater()))
    return {};
  try {
    std::string value = ::ckl::core::IndexSpace(std::move(axes)).serialize();
    return parser.getChecked<SpaceAttr>(location, parser.getContext(), value);
  } catch (const std::exception &error) {
    parser.emitError(location) << "invalid CKL index space: " << error.what();
    return {};
  }
}

void SpaceAttr::print(AsmPrinter &printer) const {
  auto space = ::ckl::core::IndexSpace::deserialize(getValue().str());
  printer << "<[";
  llvm::interleaveComma(space.axes(), printer, [&](const ::ckl::core::Axis &axis) {
    printer << axis.name << " = ";
    if (axis.isStatic())
      printer << axis.extent;
    else
      printer << "symbol<\"" << *axis.extentSymbol << "\">";
  });
  printer << "]>";
}

Attribute IndexMapAttr::parse(AsmParser &parser, Type) {
  SMLoc location = parser.getCurrentLocation();
  if (failed(parser.parseLess()) || failed(parseNamedField(parser, "domain")))
    return {};
  auto domain = parseQualifiedAttribute<SpaceAttr>(parser, "#ckl.space attribute");
  if (failed(domain) || failed(parser.parseComma()) ||
      failed(parseNamedField(parser, "codomain")))
    return {};
  auto codomain = parseQualifiedAttribute<SpaceAttr>(parser, "#ckl.space attribute");
  if (failed(codomain) || failed(parser.parseComma()) ||
      failed(parseNamedField(parser, "results")) || failed(parser.parseLSquare()))
    return {};
  std::vector<std::string> results;
  if (failed(parser.parseOptionalRSquare())) {
    while (true) {
      std::string result;
      if (failed(parser.parseString(&result)))
        return {};
      results.push_back(std::move(result));
      if (succeeded(parser.parseOptionalComma()))
        continue;
      if (failed(parser.parseRSquare()))
        return {};
      break;
    }
  }
  std::string predicate;
  if (failed(parser.parseComma()) || failed(parseNamedField(parser, "predicate")) ||
      failed(parser.parseString(&predicate)) || failed(parser.parseGreater()))
    return {};
  std::ostringstream value;
  value << "map(" << (*domain).getValue().str() << ';' << (*codomain).getValue().str() << ";[";
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (i) value << ',';
    value << results[i];
  }
  value << "];" << predicate << ')';
  return parser.getChecked<IndexMapAttr>(location, parser.getContext(), value.str());
}

void IndexMapAttr::print(AsmPrinter &printer) const {
  auto map = ::ckl::core::IndexMap::deserialize(getValue().str());
  printer << "<domain = " << SpaceAttr::get(getContext(), map.domain().serialize())
          << ", codomain = " << SpaceAttr::get(getContext(), map.codomain().serialize())
          << ", results = [";
  llvm::interleaveComma(map.results(), printer, [&](const ::ckl::core::IndexExpr &result) {
    printer << '"' << result.serialize() << '"';
  });
  printer << "], predicate = \"" << map.predicate().serialize() << "\">";
}

Attribute DistributionAttr::parse(AsmParser &parser, Type) {
  SMLoc location = parser.getCurrentLocation();
  if (failed(parser.parseLess()) || failed(parseNamedField(parser, "executors")))
    return {};
  auto executors = parseQualifiedAttribute<SpaceAttr>(parser, "#ckl.space attribute");
  if (failed(executors) || failed(parser.parseComma()) ||
      failed(parseNamedField(parser, "local")))
    return {};
  auto local = parseQualifiedAttribute<SpaceAttr>(parser, "#ckl.space attribute");
  if (failed(local) || failed(parser.parseComma()) || failed(parseNamedField(parser, "tile")))
    return {};
  auto tile = parseQualifiedAttribute<SpaceAttr>(parser, "#ckl.space attribute");
  if (failed(tile) || failed(parser.parseComma()) ||
      failed(parseNamedField(parser, "ownership")))
    return {};
  auto ownership = parseQualifiedAttribute<IndexMapAttr>(parser, "#ckl.index_map attribute");
  if (failed(ownership) || failed(parser.parseComma()) ||
      failed(parseNamedField(parser, "local_storage")))
    return {};
  auto storage = parseQualifiedAttribute<IndexMapAttr>(parser, "#ckl.index_map attribute");
  if (failed(storage) || failed(parser.parseComma()) || failed(parseNamedField(parser, "scope")))
    return {};
  auto scope = parseScope(parser);
  if (failed(scope) || failed(parser.parseComma()) ||
      failed(parseNamedField(parser, "replicated")))
    return {};
  StringRef replicated;
  if (failed(parser.parseKeyword(&replicated)) || (replicated != "true" && replicated != "false")) {
    parser.emitError(parser.getCurrentLocation()) << "expected true or false";
    return {};
  }
  if (failed(parser.parseGreater()))
    return {};
  try {
    ::ckl::core::Distribution distribution{
        ::ckl::core::IndexSpace::deserialize((*executors).getValue().str()),
        ::ckl::core::IndexSpace::deserialize((*local).getValue().str()),
        ::ckl::core::IndexSpace::deserialize((*tile).getValue().str()),
        ::ckl::core::IndexMap::deserialize((*ownership).getValue().str()),
        ::ckl::core::IndexMap::deserialize((*storage).getValue().str()), replicated == "true",
        *scope};
    return parser.getChecked<DistributionAttr>(location, parser.getContext(),
                                               ::ckl::core::serialize(distribution));
  } catch (const std::exception &error) {
    parser.emitError(location) << "invalid CKL distribution: " << error.what();
    return {};
  }
}

void DistributionAttr::print(AsmPrinter &printer) const {
  auto distribution = ::ckl::core::deserializeDistribution(getValue().str());
  printer << "<executors = "
          << SpaceAttr::get(getContext(), distribution.executorSpace.serialize())
          << ", local = " << SpaceAttr::get(getContext(), distribution.localSpace.serialize())
          << ", tile = " << SpaceAttr::get(getContext(), distribution.tileSpace.serialize())
          << ", ownership = "
          << IndexMapAttr::get(getContext(), distribution.ownership.serialize())
          << ", local_storage = "
          << IndexMapAttr::get(getContext(), distribution.localStorage.serialize())
          << ", scope = " << scopeName(distribution.scope)
          << ", replicated = " << (distribution.allowReplication ? "true" : "false") << '>';
}

LogicalResult SpaceAttr::verify(function_ref<InFlightDiagnostic()> emitError, StringRef value) {
  try {
    ::ckl::core::IndexSpace space = ::ckl::core::IndexSpace::deserialize(value.str());
    if (space.serialize() != value)
      return emitError() << "space must use canonical CKLCore spelling '" << space.serialize()
                         << "'";
  } catch (const std::exception &error) {
    return emitError() << "invalid CKL index space: " << error.what();
  }
  return success();
}

LogicalResult IndexMapAttr::verify(function_ref<InFlightDiagnostic()> emitError, StringRef value) {
  try {
    ::ckl::core::IndexMap map = ::ckl::core::IndexMap::deserialize(value.str());
    if (map.serialize() != value)
      return emitError() << "index map must use canonical CKLCore spelling '" << map.serialize()
                         << "'";
  } catch (const std::exception &error) {
    return emitError() << "invalid CKL index map: " << error.what();
  }
  return success();
}

LogicalResult DistributionAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                       StringRef value) {
  try {
    ::ckl::core::Distribution distribution = ::ckl::core::deserializeDistribution(value.str());
    if (::ckl::core::serialize(distribution) != value)
      return emitError() << "distribution must use canonical CKLCore spelling";
    auto check = ::ckl::core::verifyDistribution(distribution);
    if (!check.valid)
      return emitError() << "invalid CKL distribution: " << check.message;
  } catch (const std::exception &error) {
    return emitError() << "invalid CKL distribution: " << error.what();
  }
  return success();
}

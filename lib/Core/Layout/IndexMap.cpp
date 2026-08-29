#include "ckl/Core/Layout/IndexMap.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ckl::core {

bool Axis::operator==(const Axis &other) const {
  return name == other.name && extent == other.extent;
}

IndexSpace::IndexSpace(std::vector<Axis> axes) : axes_(std::move(axes)) {
  for (const Axis &axis : axes_) {
    if (axis.extent <= 0)
      throw std::invalid_argument("index-space extents must be positive");
  }
}

std::int64_t IndexSpace::volume() const {
  std::int64_t result = 1;
  for (const Axis &axis : axes_)
    result *= axis.extent;
  return result;
}

bool IndexSpace::contains(const std::vector<std::int64_t> &point) const {
  if (point.size() != axes_.size())
    return false;
  for (std::size_t i = 0; i < point.size(); ++i) {
    if (point[i] < 0 || point[i] >= axes_[i].extent)
      return false;
  }
  return true;
}

bool IndexSpace::sameShape(const IndexSpace &other) const {
  if (rank() != other.rank())
    return false;
  for (std::size_t i = 0; i < rank(); ++i) {
    if (axes_[i].extent != other.axes_[i].extent)
      return false;
  }
  return true;
}

std::string IndexSpace::str() const {
  std::ostringstream os;
  os << '(';
  for (std::size_t i = 0; i < axes_.size(); ++i) {
    if (i)
      os << ", ";
    os << axes_[i].name << ':' << axes_[i].extent;
  }
  return os.str() + ')';
}

struct IndexExpr::Node {
  Kind kind;
  std::int64_t value;
  std::shared_ptr<const Node> lhs;
  std::shared_ptr<const Node> rhs;
};

IndexExpr::IndexExpr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}

static IndexExpr makeLeaf(IndexExpr::Kind kind, std::int64_t value) {
  return kind == IndexExpr::Kind::Input ? IndexExpr::input(static_cast<std::size_t>(value))
                                        : IndexExpr::constant(value);
}

IndexExpr IndexExpr::input(std::size_t index) {
  return IndexExpr(
      std::make_shared<Node>(Node{Kind::Input, static_cast<std::int64_t>(index), {}, {}}));
}

IndexExpr IndexExpr::constant(std::int64_t value) {
  return IndexExpr(std::make_shared<Node>(Node{Kind::Constant, value, {}, {}}));
}

IndexExpr IndexExpr::add(IndexExpr lhs, IndexExpr rhs) {
  return IndexExpr(std::make_shared<Node>(Node{Kind::Add, 0, lhs.node_, rhs.node_}));
}

IndexExpr IndexExpr::multiply(IndexExpr lhs, IndexExpr rhs) {
  return IndexExpr(std::make_shared<Node>(Node{Kind::Multiply, 0, lhs.node_, rhs.node_}));
}

IndexExpr IndexExpr::floorDiv(IndexExpr value, std::int64_t divisor) {
  if (divisor <= 0)
    throw std::invalid_argument("floor-divisor must be positive");
  return IndexExpr(std::make_shared<Node>(Node{Kind::FloorDiv, divisor, value.node_, {}}));
}

IndexExpr IndexExpr::modulo(IndexExpr value, std::int64_t modulus) {
  if (modulus <= 0)
    throw std::invalid_argument("modulus must be positive");
  return IndexExpr(std::make_shared<Node>(Node{Kind::Modulo, modulus, value.node_, {}}));
}

IndexExpr IndexExpr::bitXor(IndexExpr lhs, IndexExpr rhs) {
  return IndexExpr(std::make_shared<Node>(Node{Kind::Xor, 0, lhs.node_, rhs.node_}));
}

std::int64_t IndexExpr::evaluate(const std::vector<std::int64_t> &inputs) const {
  std::function<std::int64_t(const std::shared_ptr<const Node> &)> eval =
      [&](const std::shared_ptr<const Node> &node) -> std::int64_t {
    switch (node->kind) {
    case Kind::Input:
      if (node->value < 0 || static_cast<std::size_t>(node->value) >= inputs.size())
        throw std::out_of_range("index expression input is out of range");
      return inputs[static_cast<std::size_t>(node->value)];
    case Kind::Constant:
      return node->value;
    case Kind::Add:
      return eval(node->lhs) + eval(node->rhs);
    case Kind::Multiply:
      return eval(node->lhs) * eval(node->rhs);
    case Kind::FloorDiv:
      return eval(node->lhs) / node->value;
    case Kind::Modulo:
      return eval(node->lhs) % node->value;
    case Kind::Xor:
      return eval(node->lhs) ^ eval(node->rhs);
    }
    throw std::logic_error("unknown index expression kind");
  };
  return eval(node_);
}

IndexExpr IndexExpr::substitute(const std::vector<IndexExpr> &replacements) const {
  std::function<IndexExpr(const std::shared_ptr<const Node> &)> rewrite =
      [&](const std::shared_ptr<const Node> &node) -> IndexExpr {
    switch (node->kind) {
    case Kind::Input:
      if (node->value < 0 || static_cast<std::size_t>(node->value) >= replacements.size())
        throw std::out_of_range("substitution input is out of range");
      return replacements[static_cast<std::size_t>(node->value)];
    case Kind::Constant:
      return makeLeaf(node->kind, node->value);
    case Kind::Add:
      return add(rewrite(node->lhs), rewrite(node->rhs));
    case Kind::Multiply:
      return multiply(rewrite(node->lhs), rewrite(node->rhs));
    case Kind::FloorDiv:
      return floorDiv(rewrite(node->lhs), node->value);
    case Kind::Modulo:
      return modulo(rewrite(node->lhs), node->value);
    case Kind::Xor:
      return bitXor(rewrite(node->lhs), rewrite(node->rhs));
    }
    throw std::logic_error("unknown index expression kind");
  };
  return rewrite(node_);
}

std::string IndexExpr::str() const {
  std::function<std::string(const std::shared_ptr<const Node> &)> print =
      [&](const std::shared_ptr<const Node> &node) -> std::string {
    if (node->kind == Kind::Input)
      return "d" + std::to_string(node->value);
    if (node->kind == Kind::Constant)
      return std::to_string(node->value);
    if (node->kind == Kind::FloorDiv)
      return "floordiv(" + print(node->lhs) + ", " + std::to_string(node->value) + ')';
    if (node->kind == Kind::Modulo)
      return "mod(" + print(node->lhs) + ", " + std::to_string(node->value) + ')';
    const char *op = node->kind == Kind::Add        ? " + "
                     : node->kind == Kind::Multiply ? " * "
                                                    : " xor ";
    return '(' + print(node->lhs) + op + print(node->rhs) + ')';
  };
  return print(node_);
}

IndexMap::IndexMap(IndexSpace domain, IndexSpace codomain, std::vector<IndexExpr> results)
    : domain_(std::move(domain)), codomain_(std::move(codomain)), results_(std::move(results)) {
  if (results_.size() != codomain_.rank())
    throw std::invalid_argument("index map must produce one expression per codomain axis");
}

IndexMap IndexMap::identity(IndexSpace space) {
  std::vector<IndexExpr> results;
  for (std::size_t i = 0; i < space.rank(); ++i)
    results.push_back(IndexExpr::input(i));
  return IndexMap(space, space, std::move(results));
}

IndexMap IndexMap::permutation(IndexSpace domain, std::vector<std::size_t> order,
                               std::vector<std::string> resultNames) {
  if (order.size() != domain.rank())
    throw std::invalid_argument("permutation rank mismatch");
  std::vector<bool> seen(order.size());
  std::vector<Axis> axes;
  std::vector<IndexExpr> results;
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (order[i] >= order.size() || seen[order[i]])
      throw std::invalid_argument("invalid permutation");
    seen[order[i]] = true;
    axes.push_back({resultNames.empty() ? domain.axes()[order[i]].name : resultNames[i],
                    domain.axes()[order[i]].extent});
    results.push_back(IndexExpr::input(order[i]));
  }
  return IndexMap(std::move(domain), IndexSpace(std::move(axes)), std::move(results));
}

static IndexExpr linearize(const IndexSpace &space) {
  IndexExpr result = IndexExpr::constant(0);
  std::int64_t stride = 1;
  for (std::size_t i = space.rank(); i-- > 0;) {
    result = IndexExpr::add(result,
                            IndexExpr::multiply(IndexExpr::input(i), IndexExpr::constant(stride)));
    stride *= space.axes()[i].extent;
  }
  return result;
}

IndexMap IndexMap::reshape(IndexSpace domain, IndexSpace codomain) {
  if (domain.volume() != codomain.volume())
    throw std::invalid_argument("reshape spaces must have equal volume");
  IndexExpr linear = linearize(domain);
  std::vector<IndexExpr> results;
  std::int64_t divisor = codomain.volume();
  for (const Axis &axis : codomain.axes()) {
    divisor /= axis.extent;
    results.push_back(IndexExpr::modulo(IndexExpr::floorDiv(linear, divisor), axis.extent));
  }
  return IndexMap(std::move(domain), std::move(codomain), std::move(results));
}

IndexMap IndexMap::strided(IndexSpace domain, std::vector<std::int64_t> strides,
                           std::int64_t offset) {
  if (strides.size() != domain.rank())
    throw std::invalid_argument("stride rank mismatch");
  IndexExpr result = IndexExpr::constant(offset);
  std::int64_t minAddress = offset;
  std::int64_t maxAddress = offset;
  for (std::size_t i = 0; i < strides.size(); ++i) {
    result = IndexExpr::add(
        result, IndexExpr::multiply(IndexExpr::input(i), IndexExpr::constant(strides[i])));
    const std::int64_t span = (domain.axes()[i].extent - 1) * strides[i];
    minAddress += std::min<std::int64_t>(0, span);
    maxAddress += std::max<std::int64_t>(0, span);
  }
  if (minAddress < 0)
    throw std::invalid_argument("strided layout produces negative addresses");
  return IndexMap(std::move(domain), IndexSpace({{"address", maxAddress + 1}}), {result});
}

std::vector<std::int64_t> IndexMap::apply(const std::vector<std::int64_t> &point) const {
  if (!domain_.contains(point))
    throw std::out_of_range("point is outside index-map domain");
  std::vector<std::int64_t> result;
  for (const IndexExpr &expr : results_)
    result.push_back(expr.evaluate(point));
  if (!codomain_.contains(result))
    throw std::out_of_range("index map produced a point outside its codomain");
  return result;
}

std::string IndexMap::str() const {
  std::ostringstream os;
  os << domain_.str() << " -> " << codomain_.str() << " {";
  for (std::size_t i = 0; i < results_.size(); ++i) {
    if (i)
      os << ", ";
    os << results_[i].str();
  }
  return os.str() + '}';
}

// NOTE: This assumes a canonical row-major correspondence between factorizations of the same
// volume. A more general implementation must preserve nested factor structure and prove that a
// refinement is compatible.
IndexMap compose(const IndexMap &outer, const IndexMap &inner) {
  if (inner.codomain().volume() != outer.domain().volume())
    throw std::invalid_argument("composition intermediate spaces have different volumes");
  std::vector<IndexExpr> replacements = inner.results();
  if (!inner.codomain().sameShape(outer.domain())) {
    IndexMap refinement = IndexMap::reshape(inner.codomain(), outer.domain());
    replacements.clear();
    for (const IndexExpr &expr : refinement.results())
      replacements.push_back(expr.substitute(inner.results()));
  }
  std::vector<IndexExpr> results;
  for (const IndexExpr &expr : outer.results())
    // for each axis in the outer map, substitute the inner map's expressions for its inputs
    results.push_back(expr.substitute(replacements));
  return IndexMap(inner.domain(), outer.codomain(), std::move(results));
}

std::vector<std::vector<std::int64_t>> enumerate(const IndexSpace &space) {
  std::vector<std::vector<std::int64_t>> points;
  std::vector<std::int64_t> point(space.rank());
  std::function<void(std::size_t)> visit = [&](std::size_t axis) {
    if (axis == space.rank()) {
      points.push_back(point);
      return;
    }
    for (std::int64_t i = 0; i < space.axes()[axis].extent; ++i) {
      point[axis] = i;
      visit(axis + 1);
    }
  };
  visit(0);
  return points;
}

// NOTE: currently the implementation enumerates every point and compare the results
// Not scalable. Move to a more efficient symbolic proof in the future (Presburger reasoning)
EquivalenceResult proveEquivalent(const IndexMap &lhs, const IndexMap &rhs,
                                  std::int64_t exhaustiveLimit) {
  if (!lhs.domain().sameShape(rhs.domain()) || !lhs.codomain().sameShape(rhs.codomain()))
    return {EquivalenceResult::Status::NotEquivalent, std::nullopt,
            "source or result shapes differ"};
  if (lhs.domain().volume() > exhaustiveLimit)
    return {EquivalenceResult::Status::Unknown, std::nullopt,
            "domain exceeds exhaustive proof limit"};
  for (const auto &point : enumerate(lhs.domain())) {
    if (lhs.apply(point) != rhs.apply(point))
      return {EquivalenceResult::Status::NotEquivalent, point, "maps differ at witness point"};
  }
  return {EquivalenceResult::Status::Equivalent, std::nullopt, "exhaustive finite-domain proof"};
}

} // namespace ckl::core

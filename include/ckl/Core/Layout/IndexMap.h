#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace ckl::core {

struct Axis {
  std::string name;
  std::int64_t extent;

  bool operator==(const Axis &other) const;
};

class IndexSpace {
public:
  explicit IndexSpace(std::vector<Axis> axes = {});

  const std::vector<Axis> &axes() const { return axes_; }
  std::size_t rank() const { return axes_.size(); }            // number of axes
  std::int64_t volume() const;                                 // product of all extents
  bool contains(const std::vector<std::int64_t> &point) const; // bounds checking
  bool sameShape(const IndexSpace &other) const; // compares extents, ignoring axis names
  std::string str() const;

private:
  std::vector<Axis> axes_;
};

/*
Immutable expression tree
For example, a row-major offset 4 * i + j can be expressed as:
  IndexExpr::add(
      IndexExpr::multiply(IndexExpr::input(0), IndexExpr::constant(4)),
      IndexExpr::input(1))

*/

class IndexExpr {
public:
  enum class Kind { Input, Constant, Add, Multiply, FloorDiv, Modulo, Xor };

  static IndexExpr input(std::size_t index);
  static IndexExpr constant(std::int64_t value);
  static IndexExpr add(IndexExpr lhs, IndexExpr rhs);
  static IndexExpr multiply(IndexExpr lhs, IndexExpr rhs);
  static IndexExpr floorDiv(IndexExpr value, std::int64_t divisor);
  static IndexExpr modulo(IndexExpr value, std::int64_t modulus);
  static IndexExpr bitXor(IndexExpr lhs, IndexExpr rhs);

  // Recursively evaluates the tree for one input coordinate
  std::int64_t evaluate(const std::vector<std::int64_t> &inputs) const;

  /*
  Substitution replaces each input node with the corresponding expression in `replacements`.
  Suppose f(x) = x + 1, g(y) = 2 * y, and h(z) = f(g(z)). Then h.substitute({g}) produces the
  expression tree for 2 * z + 1.
  */
  IndexExpr substitute(const std::vector<IndexExpr> &replacements) const;
  std::string str() const;

private:
  struct Node;
  explicit IndexExpr(std::shared_ptr<const Node> node);
  std::shared_ptr<const Node> node_;
};

/*

*/
class IndexMap {
public:
  IndexMap(IndexSpace domain, IndexSpace codomain, std::vector<IndexExpr> results);

  static IndexMap identity(IndexSpace space);
  static IndexMap permutation(IndexSpace domain, std::vector<std::size_t> order,
                              std::vector<std::string> resultNames = {});

  // linearizes the domain and then reshapes it to the codomain. The two spaces must have equal
  // volume.
  static IndexMap reshape(IndexSpace domain, IndexSpace codomain);

  // maps a logical coordinate to a 1-D address
  // address = offset + sum_i(stride[i] * coordinate[i])
  static IndexMap strided(IndexSpace domain, std::vector<std::int64_t> strides,
                          std::int64_t offset = 0);

  const IndexSpace &domain() const { return domain_; }
  const IndexSpace &codomain() const { return codomain_; }
  const std::vector<IndexExpr> &results() const { return results_; }

  // Evaluates the map at a single point. Throws std::out_of_range if the point is outside the
  // domain
  std::vector<std::int64_t> apply(const std::vector<std::int64_t> &point) const;
  std::string str() const;

private:
  IndexSpace domain_;
  IndexSpace codomain_;

  // A map must have exactly one expression per codomain axis, in order.
  std::vector<IndexExpr> results_;
};

// Returns `outer(inner(x))`. Intermediate spaces need only have the same volume;
// a canonical row-major refinement is inserted when their factorization differs.
// e.g. inner: A -> (2, 3, 6) outter: (6, 6) -> B produces a map A -> B via the refinement (2, 3, 6)
// -> (6, 6).
IndexMap compose(const IndexMap &outer, const IndexMap &inner);

struct EquivalenceResult {
  enum class Status { Equivalent, NotEquivalent, Unknown } status;
  std::optional<std::vector<std::int64_t>> witness;
  std::string reason;
};

EquivalenceResult proveEquivalent(const IndexMap &lhs, const IndexMap &rhs,
                                  std::int64_t exhaustiveLimit = 1'000'000);

std::vector<std::vector<std::int64_t>> enumerate(const IndexSpace &space);

} // namespace ckl::core

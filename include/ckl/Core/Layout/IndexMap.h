#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ckl::core {

// Represents a single axis of an index space, with a name and an extent. The extent can be either
// a positive integer or a symbolic name, which can be instantiated later.
// It can not be enumerated or have its volume computed until all symbolic extents are instantiated.
struct Axis {
  Axis() = default;
  Axis(std::string name, std::int64_t extent);

  std::string name;
  std::int64_t extent = -1;
  std::optional<std::string> extentSymbol;

  static Axis symbolic(std::string name, std::string extentSymbol);
  bool isStatic() const { return !extentSymbol.has_value(); }

  bool operator==(const Axis &other) const;
};

class IndexSpace {
public:
  struct Structure {
    std::string name;
    std::optional<Axis> axis;
    std::vector<std::shared_ptr<const Structure>> children;
  };

  explicit IndexSpace(std::vector<Axis> axes = {});

  /*
  Preserves factor grouping.
  For example:
  IndexSpace mn = IndexSpace::product("tile", {IndexSpace({{"m", 4}}), IndexSpace::product(
                                            "n-factors", {IndexSpace({{"n0", 2}}),
                                                          IndexSpace({{"n1", 4}})})});
  represents: tile(m:4, n-factors(n0:2, n1:4)) with a flat coordinate space of (m, n0, n1) and a
  profile of "tile([*],n-factors([*],[*]))".
  It is represented as a tree of Structure nodes, where each leaf node corresponds to an axis.
  */
  static IndexSpace product(std::string name, std::vector<IndexSpace> children);

  const std::vector<Axis> &axes() const { return axes_; }
  std::size_t rank() const { return axes_.size(); } // number of axes
  std::int64_t volume() const;                      // product of all extents
  bool isStatic() const;
  IndexSpace instantiate(const std::unordered_map<std::string, std::int64_t> &bindings) const;
  bool contains(const std::vector<std::int64_t> &point) const; // bounds checking
  bool sameShape(const IndexSpace &other) const; // compares extents, ignoring axis names
  const std::string &profile() const { return profile_; }
  const std::shared_ptr<const Structure> &structure() const { return structure_; }
  std::size_t hash() const;
  std::string str() const;
  std::string serialize() const;

private:
  IndexSpace(std::vector<Axis> axes, std::string profile,
             std::shared_ptr<const Structure> structure);
  std::vector<Axis> axes_;
  std::string profile_;
  std::shared_ptr<const Structure> structure_;
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
  IndexExpr normalize() const;
  std::size_t hash() const;
  std::string serialize() const;
  std::string str() const;

private:
  struct Node;
  explicit IndexExpr(std::shared_ptr<const Node> node);
  std::shared_ptr<const Node> node_;
};

/*
e.g. firstFive = IndexPredicate::compare(IndexExpr::input(0), IndexPredicate::Comparison::Less,
                                 IndexExpr::constant(5));
*/
class IndexPredicate {
public:
  enum class Comparison { Less, LessEqual, Equal, NotEqual, GreaterEqual, Greater };

  static IndexPredicate always();
  static IndexPredicate compare(IndexExpr lhs, Comparison comparison, IndexExpr rhs);
  static IndexPredicate logicalAnd(IndexPredicate lhs, IndexPredicate rhs);

  bool evaluate(const std::vector<std::int64_t> &inputs) const;
  IndexPredicate substitute(const std::vector<IndexExpr> &replacements) const;
  std::string serialize() const;
  std::string str() const;
  std::size_t hash() const;

private:
  struct Node;
  explicit IndexPredicate(std::shared_ptr<const Node> node);
  std::shared_ptr<const Node> node_;
};

class IndexMap {
public:
  IndexMap(IndexSpace domain, IndexSpace codomain, std::vector<IndexExpr> results,
           IndexPredicate predicate = IndexPredicate::always());

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
  const IndexPredicate &predicate() const { return predicate_; }

  // Evaluates the map at a single point. Throws std::out_of_range if the point is outside the
  // domain
  std::vector<std::int64_t> apply(const std::vector<std::int64_t> &point) const;
  std::optional<std::vector<std::int64_t>> tryApply(const std::vector<std::int64_t> &point) const;
  IndexMap normalize() const;
  std::size_t hash() const;
  std::string str() const;
  std::string serialize() const;
  static IndexMap deserialize(const std::string &text);

  /*
  Prints bounded map as a coordinate table
  */
  std::string diagram(std::int64_t pointLimit = 256) const;

private:
  IndexSpace domain_;
  IndexSpace codomain_;

  // A map must have exactly one expression per codomain axis, in order.
  std::vector<IndexExpr> results_;
  IndexPredicate predicate_;
};

// Returns `outer(inner(x))`. Intermediate spaces need only have the same volume;
// a canonical row-major refinement is inserted when their factorization differs.
// e.g. inner: A -> (2, 3, 6) outter: (6, 6) -> B produces a map A -> B via the refinement (2, 3, 6)
// -> (6, 6).
// Predicate is transformed as well. E.g. if inner is valid under Pinner(x) and outer is valid under
// Pouter(y), then the composition is valid under Pinner(x) && Pouter(inner(x)).
IndexMap compose(const IndexMap &outer, const IndexMap &inner);

/*
For reshape, we can not silently insert a row-major refinement between equal-volume spaces, because
the factorization of the two spaces may differ. For example, the spaces (6, 2) and (4, 3) have equal
volume but no common refinement.
FactorRefinement provides an explicit canonical prime-factor refinement of two equal-volume spaces.
For example, the spaces (6, 2) and (4, 3) have a common refinement (2, 2, 3) with maps
(6, 2) -> (2, 2, 3) and (4, 3) -> (2, 2, 3).
*/
struct FactorRefinement {
  IndexSpace commonSpace;
  IndexMap fromLeft;
  IndexMap fromRight;
  std::vector<std::int64_t> primeFactors;
};

// Produces an explicit common prime-factor space and maps from both equal-volume spaces.
std::optional<FactorRefinement> computeMutualRefinement(const IndexSpace &left,
                                                        const IndexSpace &right);

struct EquivalenceResult {
  enum class Status { Equivalent, NotEquivalent, Unknown } status;
  std::optional<std::vector<std::int64_t>> witness;
  std::string reason;
};

EquivalenceResult proveEquivalent(const IndexMap &lhs, const IndexMap &rhs,
                                  std::int64_t exhaustiveLimit = 1'000'000);

std::vector<std::vector<std::int64_t>> enumerate(const IndexSpace &space);

} // namespace ckl::core

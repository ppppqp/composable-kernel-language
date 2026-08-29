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
  std::size_t rank() const { return axes_.size(); }
  std::int64_t volume() const;
  bool contains(const std::vector<std::int64_t> &point) const;
  bool sameShape(const IndexSpace &other) const;
  std::string str() const;

private:
  std::vector<Axis> axes_;
};

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

  std::int64_t evaluate(const std::vector<std::int64_t> &inputs) const;
  IndexExpr substitute(const std::vector<IndexExpr> &replacements) const;
  std::string str() const;

private:
  struct Node;
  explicit IndexExpr(std::shared_ptr<const Node> node);
  std::shared_ptr<const Node> node_;
};

class IndexMap {
public:
  IndexMap(IndexSpace domain, IndexSpace codomain, std::vector<IndexExpr> results);

  static IndexMap identity(IndexSpace space);
  static IndexMap permutation(IndexSpace domain, std::vector<std::size_t> order,
                              std::vector<std::string> resultNames = {});
  static IndexMap reshape(IndexSpace domain, IndexSpace codomain);
  static IndexMap strided(IndexSpace domain, std::vector<std::int64_t> strides,
                          std::int64_t offset = 0);

  const IndexSpace &domain() const { return domain_; }
  const IndexSpace &codomain() const { return codomain_; }
  const std::vector<IndexExpr> &results() const { return results_; }

  std::vector<std::int64_t> apply(const std::vector<std::int64_t> &point) const;
  std::string str() const;

private:
  IndexSpace domain_;
  IndexSpace codomain_;
  std::vector<IndexExpr> results_;
};

// Returns `outer(inner(x))`. Intermediate spaces need only have the same volume;
// a canonical row-major refinement is inserted when their factorization differs.
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


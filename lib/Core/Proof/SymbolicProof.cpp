#include "ckl/Core/Proof/SymbolicProof.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"

#include <algorithm>
#include <stdexcept>

namespace ckl::core {
using mlir::presburger::IntegerPolyhedron;
using mlir::presburger::PresburgerSpace;

namespace {

void validate(const SymbolicDomain &domain, const AffineForm &form) {
  if (form.coefficients.size() != domain.dimensions + domain.symbols + 1)
    throw std::invalid_argument("affine form has the wrong number of coefficients");
}

// constructs an IntegerPolyhedron from a SymbolicDomain, which is a set of integer points
// satisfying the domain's inequalities and equalities
IntegerPolyhedron buildSet(const SymbolicDomain &domain) {
  IntegerPolyhedron set(PresburgerSpace::getSetSpace(domain.dimensions, domain.symbols));
  for (const auto &inequality : domain.inequalities) {
    validate(domain, inequality);
    set.addInequality(inequality.coefficients);
  }
  for (const auto &equality : domain.equalities) {
    validate(domain, equality);
    set.addEquality(equality.coefficients);
  }
  return set;
}

} // namespace

/*
Translates an IndexSpace into a Presburger domain.
(m: 4, n: $N) is equivalent to the Presburger domain:
  d0 >= 0
  d0 <= 3
  d1 >= 0
  d1 <= N - 1
  N >= 1
*/
IndexSpaceDomain makeIndexSpaceDomain(const IndexSpace &space) {
  std::vector<std::string> symbols;
  // collect all symbols in all axes
  for (const Axis &axis : space.axes()) {
    if (!axis.isStatic() &&
        std::find(symbols.begin(), symbols.end(), *axis.extentSymbol) == symbols.end())
      symbols.push_back(*axis.extentSymbol);
  }
  SymbolicDomain domain{space.rank(), symbols.size(), {}, {}};
  const std::size_t width = domain.dimensions + domain.symbols + 1;
  for (std::size_t dimension = 0; dimension < space.rank(); ++dimension) {
    // the lower/upper is organized as dimension coeff, symbol coeff, constant term
    // d0...dn-1, s0...sm-1, constant
    std::vector<std::int64_t> lower(width);

    // for each dimension, we add a lower bound of 0 and an upper bound of extent - 1
    lower[dimension] = 1;
    // inequalities is a list of affine expressions constrained to be >= 0, so we add the lower
    // bound as -d_i <= 0
    domain.inequalities.push_back({std::move(lower)});

    std::vector<std::int64_t> upper(width);
    upper[dimension] = -1;
    const Axis &axis = space.axes()[dimension];
    if (axis.isStatic()) {
      // static, no need to add symbol
      // -d + extent - 1 >= 0
      upper.back() = axis.extent - 1;
    } else {
      // add symbol
      // -d + N - 1 >= 0
      auto symbol = std::find(symbols.begin(), symbols.end(), *axis.extentSymbol);
      upper[domain.dimensions + static_cast<std::size_t>(symbol - symbols.begin())] = 1;
      upper.back() = -1;
    }
    domain.inequalities.push_back({std::move(upper)});
  }
  for (std::size_t symbol = 0; symbol < symbols.size(); ++symbol) {
    //  N - 1 >= 0
    std::vector<std::int64_t> positive(width);
    positive[domain.dimensions + symbol] = 1;
    positive.back() = -1;
    domain.inequalities.push_back({std::move(positive)});
  }
  return {std::move(domain), std::move(symbols)};
}

/*
difference = lhs - rhs
If difference is always zero, then lhs == rhs. Otherwise, there exists an integer point in
counter-domain.
*/
SymbolicProofResult proveAffineEqual(const SymbolicDomain &domain, const AffineForm &lhs,
                                     const AffineForm &rhs) {
  validate(domain, lhs);
  validate(domain, rhs);
  std::vector<std::int64_t> difference(lhs.coefficients.size());
  for (std::size_t i = 0; i < difference.size(); ++i)
    difference[i] = lhs.coefficients[i] - rhs.coefficients[i];

  IntegerPolyhedron positive = buildSet(domain);
  auto positiveDifference = difference;
  --positiveDifference.back(); // lhs - rhs >= 1
  // bound the difference
  positive.addInequality(positiveDifference);

  IntegerPolyhedron negative = buildSet(domain);
  auto negativeDifference = difference;
  for (auto &coefficient : negativeDifference)
    coefficient = -coefficient;
  --negativeDifference.back(); // rhs - lhs >= 1
  negative.addInequality(negativeDifference);

  const bool equal = positive.isIntegerEmpty() && negative.isIntegerEmpty();
  return {equal, equal ? "Presburger proof: difference is always zero"
                       : "Presburger counter-domain is non-empty"};
}

} // namespace ckl::core

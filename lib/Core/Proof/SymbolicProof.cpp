#include "ckl/Core/Proof/SymbolicProof.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"

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

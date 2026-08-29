#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ckl::core {

// Coefficients are ordered as dimensions, symbols, then the constant term.
// For example, for one dimension d and one symbol n, the affine expression 2 * d + 3 * n + 4 is
// represented as {2, 3, 4}.
struct AffineForm {
  std::vector<std::int64_t> coefficients;
};

struct SymbolicDomain {
  std::size_t dimensions;
  std::size_t symbols;
  // Each row denotes an affine expression constrained to be >= 0.
  std::vector<AffineForm> inequalities;
  // Each row denotes an affine expression constrained to be == 0.
  std::vector<AffineForm> equalities;
};

struct SymbolicProofResult {
  bool proven;
  std::string reason;
};

SymbolicProofResult proveAffineEqual(const SymbolicDomain &domain, const AffineForm &lhs,
                                     const AffineForm &rhs);

} // namespace ckl::core

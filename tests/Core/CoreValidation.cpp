#include "ckl/Core/Layout/Distribution.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ckl::core;

namespace {

int failures = 0;

void check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

IndexSpace space(std::initializer_list<std::int64_t> extents, const std::string &prefix = "d") {
  std::vector<Axis> axes;
  std::size_t index = 0;
  for (std::int64_t extent : extents)
    axes.push_back({prefix + std::to_string(index++), extent});
  return IndexSpace(std::move(axes));
}

Distribution contiguousDistribution(std::int64_t executors, std::int64_t localValues,
                                    bool reverseLocalStorage = false) {
  IndexSpace executor = space({executors}, "p");
  IndexSpace local = space({localValues}, "y");
  IndexSpace tile = space({executors * localValues}, "x");
  IndexSpace product({{"p0", executors}, {"y0", localValues}});
  IndexExpr owner =
      IndexExpr::add(IndexExpr::multiply(IndexExpr::input(0), IndexExpr::constant(localValues)),
                     IndexExpr::input(1));
  IndexMap ownership(product, tile, {owner});
  IndexExpr slot =
      reverseLocalStorage
          ? IndexExpr::add(IndexExpr::constant(localValues - 1),
                           IndexExpr::multiply(IndexExpr::input(0), IndexExpr::constant(-1)))
          : IndexExpr::input(0);
  IndexMap localStorage(local, space({localValues}, "r"), {slot});
  return {executor, local, tile, ownership, localStorage};
}

void testIdentityAndAssociativity() {
  IndexSpace a = space({2, 3});
  IndexSpace b = space({3, 2});
  IndexSpace c = space({6});
  IndexMap ab = IndexMap::reshape(a, b);
  IndexMap bc = IndexMap::reshape(b, c);
  IndexMap id = IndexMap::identity(a);

  check(proveEquivalent(compose(ab, id), ab).status == EquivalenceResult::Status::Equivalent,
        "right identity law");
  check(proveEquivalent(compose(IndexMap::identity(b), ab), ab).status ==
            EquivalenceResult::Status::Equivalent,
        "left identity law");

  IndexMap ca = IndexMap::reshape(c, a);
  IndexMap lhs = compose(ca, compose(bc, ab));
  IndexMap rhs = compose(compose(ca, bc), ab);
  check(proveEquivalent(lhs, rhs).status == EquivalenceResult::Status::Equivalent,
        "composition associativity");
}

void testFactorRefinement() {
  IndexSpace sixSix = space({6, 6}, "a");
  IndexSpace twelveThree = space({12, 3}, "b");
  IndexSpace refined = space({2, 3, 6}, "r");
  IndexMap first = IndexMap::reshape(sixSix, refined);
  IndexMap second = IndexMap::reshape(twelveThree, space({36}, "z"));
  IndexMap composed = compose(second, first);
  IndexMap direct = IndexMap::reshape(sixSix, space({36}, "z"));
  check(proveEquivalent(composed, direct).status == EquivalenceResult::Status::Equivalent,
        "composition inserts mutual-volume factor refinement");
}

void testFactorizationFamily() {
  for (std::int64_t volume = 2; volume <= 48; ++volume) {
    IndexSpace flat = space({volume}, "flat");
    for (std::int64_t factor = 1; factor <= volume; ++factor) {
      if (volume % factor != 0)
        continue;
      IndexSpace factored = space({factor, volume / factor}, "factor");
      IndexMap split = IndexMap::reshape(flat, factored);
      IndexMap merge = IndexMap::reshape(factored, flat);
      check(proveEquivalent(compose(merge, split), IndexMap::identity(flat)).status ==
                EquivalenceResult::Status::Equivalent,
            "split/merge refinement round-trip for volume " + std::to_string(volume));
    }
  }
}

void testStorageViewComposition() {
  IndexSpace matrix = space({4, 8}, "x");
  IndexMap transpose = IndexMap::permutation(matrix, {1, 0});
  IndexMap rowMajor = IndexMap::strided(transpose.codomain(), {4, 1});
  IndexMap access = compose(rowMajor, transpose);
  check(access.apply({2, 5}) == std::vector<std::int64_t>{22},
        "view composes independently with storage layout");
}

void testEquivalenceWitness() {
  IndexSpace matrix = space({2, 3});
  IndexMap rowMajor = IndexMap::strided(matrix, {3, 1});
  IndexMap columnMajor = IndexMap::strided(matrix, {1, 2});
  EquivalenceResult result = proveEquivalent(rowMajor, columnMajor);
  check(result.status == EquivalenceResult::Status::NotEquivalent && result.witness.has_value(),
        "inequality produces a witness coordinate");
}

void testDistributionAndConversions() {
  Distribution base = contiguousDistribution(4, 2);
  DistributionCheck validity = verifyDistribution(base);
  check(validity.valid && validity.covering && validity.unique,
        "CK-style P x Y to X distribution is a unique cover");

  Distribution replicated = base;
  IndexSpace product({{"p0", 4}, {"y0", 2}});
  replicated.ownership =
      IndexMap(product, base.tileSpace,
               {IndexExpr::add(IndexExpr::multiply(IndexExpr::modulo(IndexExpr::input(0), 2),
                                                   IndexExpr::constant(2)),
                               IndexExpr::input(1))});
  DistributionCheck replicationCheck = verifyDistribution(replicated);
  check(!replicationCheck.valid && !replicationCheck.covering && !replicationCheck.unique,
        "invalid replicated/incomplete ownership is rejected by unique-cover "
        "verification");

  Distribution reversedSlots = contiguousDistribution(4, 2, true);
  check(classifyConversion(base, reversedSlots, 4).kind == ConversionKind::LocalPermutation,
        "local register order is distinct from ownership");

  Distribution swappedOwners = base;
  IndexExpr swapped = IndexExpr::add(
      IndexExpr::multiply(
          IndexExpr::modulo(IndexExpr::add(IndexExpr::input(0), IndexExpr::constant(1)), 4),
          IndexExpr::constant(2)),
      IndexExpr::input(1));
  swappedOwners.ownership = IndexMap(product, base.tileSpace, {swapped});
  check(classifyConversion(base, swappedOwners, 4).kind == ConversionKind::SubgroupExchange,
        "lane ownership change is a subgroup exchange");

  check(classifyConversion(base, swappedOwners, 2).kind == ConversionKind::SharedMemoryExchange,
        "ownership crossing subgroup boundaries requires shared memory");
}

void testUnknownProofLimit() {
  IndexSpace large = space({1024, 1024});
  IndexMap id = IndexMap::identity(large);
  check(proveEquivalent(id, id, 1024).status == EquivalenceResult::Status::Unknown,
        "proof limits fail conservatively");
}

} // namespace

int main() {
  try {
    testIdentityAndAssociativity();
    testFactorRefinement();
    testFactorizationFamily();
    testStorageViewComposition();
    testEquivalenceWitness();
    testDistributionAndConversions();
    testUnknownProofLimit();
  } catch (const std::exception &error) {
    std::cerr << "UNCAUGHT: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  if (failures != 0) {
    std::cerr << failures << " core validation checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "CKL core validation passed\n";
  return EXIT_SUCCESS;
}

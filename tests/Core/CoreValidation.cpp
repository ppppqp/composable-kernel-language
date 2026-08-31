#include "ckl/Core/Layout/Distribution.h"
#include "ckl/Extensions/AMD/MfmaLayouts.h"
#include "ckl/Core/Layout/StorageLayout.h"
#include "ckl/Core/Composition/Task.h"
#include "ckl/Core/Planning/ExchangeSchedule.h"
#ifdef CKL_ENABLE_MLIR
#include "ckl/Core/Proof/SymbolicProof.h"
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ckl::core;
using ckl::extensions::amd::makeMfmaAccumulatorDistribution;

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

Distribution rotateExecutorOwnership(Distribution distribution) {
  const std::int64_t executors = distribution.executorSpace.axes().front().extent;
  const std::int64_t localValues = distribution.localSpace.axes().front().extent;
  IndexSpace product({{"p0", executors}, {"y0", localValues}});
  IndexExpr owner = IndexExpr::add(
      IndexExpr::multiply(
          IndexExpr::modulo(IndexExpr::add(IndexExpr::input(0), IndexExpr::constant(1)),
                            executors),
          IndexExpr::constant(localValues)),
      IndexExpr::input(1));
  distribution.ownership = IndexMap(product, distribution.tileSpace, {owner});
  return distribution;
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
  check(serialize(deserializeDistribution(serialize(base))) == serialize(base),
        "distributions have a deterministic semantic serialization round-trip");
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

void testNestingNormalizationAndHashing() {
  IndexSpace mn = IndexSpace::product(
      "tile", {IndexSpace({{"m", 4}}), IndexSpace::product(
                                            "n-factors", {IndexSpace({{"n0", 2}}),
                                                          IndexSpace({{"n1", 4}})})});
  check(mn.sameShape(space({4, 2, 4})) && mn.profile() == "tile([*],n-factors([*],[*]))",
        "nested products preserve a profile while exposing a flat coordinate space");
  check(mn.structure()->children.size() == 2 &&
            mn.structure()->children[1]->name == "n-factors" &&
            mn.structure()->children[1]->children.size() == 2,
        "nested products retain a structured tree, not only diagnostic text");

  IndexExpr a = IndexExpr::add(IndexExpr::input(0), IndexExpr::input(1));
  IndexExpr b = IndexExpr::add(IndexExpr::input(1), IndexExpr::input(0));
  check(a.normalize().str() == b.normalize().str() && a.hash() == b.hash(),
        "commutative expression normalization provides stable structural hashes");
}

void testExplicitMutualRefinement() {
  auto refinement = computeMutualRefinement(space({6, 2}), space({4, 3}));
  check(refinement.has_value() && refinement->primeFactors == std::vector<std::int64_t>({2, 2, 3}) &&
            refinement->commonSpace.volume() == 12,
        "equal-volume spaces produce an explicit canonical prime-factor refinement");
  check(proveEquivalent(compose(IndexMap::reshape(refinement->commonSpace, space({4, 3})),
                                refinement->fromLeft),
                        IndexMap::reshape(space({6, 2}), space({4, 3}))).status ==
            EquivalenceResult::Status::Equivalent,
        "refinement maps preserve row-major coordinate realization");
  check(!computeMutualRefinement(space({6}), space({8})),
        "different-volume spaces have no mutual refinement");
}

void testPartialMaps() {
  IndexSpace domain = space({8});
  IndexPredicate firstFive = IndexPredicate::compare(
      IndexExpr::input(0), IndexPredicate::Comparison::Less, IndexExpr::constant(5));
  IndexMap partial(domain, domain, {IndexExpr::input(0)}, firstFive);
  check(partial.tryApply({4}).has_value() && !partial.tryApply({5}).has_value(),
        "partial maps distinguish valid and masked points");
  IndexMap address = IndexMap::strided(domain, {2});
  IndexMap composed = compose(address, partial);
  check(composed.tryApply({3}) == std::optional<std::vector<std::int64_t>>({{6}}) &&
            !composed.tryApply({7}),
        "predicates propagate through composition");
}

void testReplication() {
  Distribution replicated = contiguousDistribution(4, 2);
  replicated.tileSpace = space({4}, "x");
  IndexSpace product({{"p0", 4}, {"y0", 2}});
  replicated.ownership = IndexMap(
      product, replicated.tileSpace,
      {IndexExpr::add(IndexExpr::multiply(IndexExpr::modulo(IndexExpr::input(0), 2),
                                          IndexExpr::constant(2)),
                      IndexExpr::input(1))});
  replicated.allowReplication = true;
  DistributionCheck checkResult = verifyDistribution(replicated);
  check(checkResult.valid && checkResult.covering && !checkResult.unique &&
            checkResult.maximumReplication == 2,
        "declared replication is accepted and its maximum multiplicity is measured");
}

void testXorSwizzle() {
  IndexSpace domain = space({8});
  IndexExpr swizzled = IndexExpr::bitXor(IndexExpr::input(0), IndexExpr::constant(3));
  IndexMap xorMap(domain, domain, {swizzled});
  IndexMap twice = compose(xorMap, xorMap);
  check(proveEquivalent(twice, IndexMap::identity(domain)).status ==
            EquivalenceResult::Status::Equivalent,
        "bounded XOR swizzle is validated pointwise as an involution");
}

void testStorageLayouts() {
  IndexSpace logical = space({4, 4});
  StorageLayout packed{logical, AddressSpace::Shared, AddressUnit::Element,
                       IndexMap::strided(logical, {4, 1}), 16, AliasPolicy::Unique, "tile", 16};
  StorageCheck packedCheck = verifyStorageLayout(packed);
  check(packedCheck.valid && packedCheck.injective && packedCheck.maximumAddress == 15,
        "packed shared layout is injective and bounded");

  IndexMap broadcast(logical, space({1}, "address"), {IndexExpr::constant(0)});
  StorageLayout illegalAlias{logical, AddressSpace::Shared, AddressUnit::Element, broadcast, 4,
                             AliasPolicy::Unique, std::nullopt};
  check(!verifyStorageLayout(illegalAlias).valid,
        "writable unique storage rejects aliasing");
  illegalAlias.aliasPolicy = AliasPolicy::ReadOnlyAliasing;
  check(verifyStorageLayout(illegalAlias).valid && !serialize(illegalAlias).empty(),
        "declared read-only aliasing is legal and serializable");

  packed.allocationElements = 15;
  check(!verifyStorageLayout(packed).valid,
        "storage verification rejects addresses beyond the allocation");

  StorageLayout banked{logical, AddressSpace::Shared, AddressUnit::Element,
                       IndexMap::strided(logical, {4, 1}), 4, AliasPolicy::Unique, std::nullopt, 16};
  BankConflictReport conflictFree = analyzeBankConflicts(
      banked, {{0, 0}, {0, 1}, {0, 2}, {0, 3}}, 4);
  BankConflictReport conflicting = analyzeBankConflicts(
      banked, {{0, 0}, {1, 0}, {2, 0}, {3, 0}}, 4);
  check(conflictFree.maximumConflict == 1 && conflicting.maximumConflict == 4,
        "bank analysis distinguishes conflict-free rows from same-bank columns");
}

void testConcreteConversionPlans() {
  Distribution base = contiguousDistribution(4, 2);
  Distribution reversed = contiguousDistribution(4, 2, true);
  ConversionPlan local = classifyConversion(base, reversed, 4);
  check(local.kind == ConversionKind::LocalPermutation && local.moves.size() == 8 &&
            local.moves.front().sourceLocal != local.moves.front().targetLocal,
        "local conversion contains concrete source and target slots");

  Distribution rotated = base;
  IndexSpace product({{"p0", 4}, {"y0", 2}});
  rotated.ownership = IndexMap(
      product, base.tileSpace,
      {IndexExpr::add(IndexExpr::multiply(IndexExpr::modulo(
                                              IndexExpr::add(IndexExpr::input(0),
                                                             IndexExpr::constant(1)),
                                              4),
                                          IndexExpr::constant(2)),
                      IndexExpr::input(1))});
  ConversionPlan exchange = classifyConversion(base, rotated, 4);
  check(exchange.kind == ConversionKind::SubgroupExchange && exchange.moves.size() == 8,
        "subgroup conversion enumerates every logical value movement");

  ExchangeSchedule shuffle = scheduleConversion(exchange);
  check(shuffle.steps.size() == 8 &&
            shuffle.steps.front().kind == ExchangeStepKind::SubgroupShuffle,
        "subgroup movements lower to a concrete shuffle schedule");
  ConversionPlan sharedPlan = classifyConversion(base, rotated, 2);
  ExchangeSchedule shared = scheduleConversion(sharedPlan);
  check(shared.requiresBarrier && shared.sharedElements == 8 && shared.steps.size() == 17 &&
            shared.steps[8].kind == ExchangeStepKind::Barrier,
        "cross-subgroup movements lower to store/barrier/load shared-memory schedule");

  Distribution gridTarget = rotated;
  gridTarget.scope = ExecutionScope::Grid;
  ConversionPlan globalPlan = classifyConversion(base, gridTarget, 4);
  ExchangeSchedule global = scheduleConversion(globalPlan);
  check(globalPlan.kind == ConversionKind::GlobalMemoryExchange &&
            global.globalElements == 8 && global.steps.size() == 17 &&
            global.steps[8].kind == ExchangeStepKind::KernelBoundary,
        "cross-scope ownership lowers to global store/kernel-boundary/load materialization");

  Distribution identicalGridTarget = base;
  identicalGridTarget.scope = ExecutionScope::Grid;
  check(classifyConversion(base, identicalGridTarget, 4).kind ==
            ConversionKind::GlobalMemoryExchange,
        "scope changes require materialization even when coordinate ownership is identical");
  identicalGridTarget.scope = ExecutionScope::Workgroup;
  base.scope = ExecutionScope::Subgroup;
  check(classifyConversion(base, identicalGridTarget, 4).kind ==
            ConversionKind::SharedMemoryExchange,
        "subgroup-to-workgroup scope changes materialize in shared memory");
  base.scope = ExecutionScope::Grid;
  identicalGridTarget.scope = ExecutionScope::Grid;
  check(classifyConversion(base, identicalGridTarget, 4).kind == ConversionKind::Identity,
        "equal grid scopes do not force a spurious global round trip");
}

void testStableSerializationAndDiagram() {
  IndexSpace domain = space({8}, "lane");
  IndexPredicate active = IndexPredicate::compare(
      IndexExpr::input(0), IndexPredicate::Comparison::Less, IndexExpr::constant(5));
  IndexMap map(domain, domain,
               {IndexExpr::bitXor(IndexExpr::input(0), IndexExpr::constant(3))}, active);
  std::string serialized = map.serialize();
  IndexMap parsed = IndexMap::deserialize(serialized);
  check(parsed.serialize() == serialized &&
            proveEquivalent(map, parsed).status == EquivalenceResult::Status::Equivalent,
        "index maps have a deterministic semantic parse/print round-trip");
  std::string picture = parsed.diagram();
  check(picture.find("[0] -> [3]") != std::string::npos &&
            picture.find("[5] -> inactive") != std::string::npos,
        "map diagrams expose active mappings and predicate-masked coordinates");

  bool rejected = false;
  try {
    (void)IndexMap::deserialize(serialized + "junk");
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "index-map parser rejects trailing malformed input");
}

void testSymbolicIndexSpaces() {
  IndexSpace symbolic({{"m", 4}, Axis::symbolic("n", "N")});
  check(!symbolic.isStatic() && symbolic.str() == "(m:4, n:$N)",
        "index spaces retain named symbolic extents");
  IndexSpace concrete = symbolic.instantiate({{"N", 8}});
  check(concrete.isStatic() && concrete.volume() == 32 && concrete.axes()[1].extent == 8,
        "symbolic index spaces instantiate to ordinary enumerable spaces");
  IndexMap symbolicIdentity = IndexMap::identity(symbolic);
  check(IndexMap::deserialize(symbolicIdentity.serialize()).serialize() ==
            symbolicIdentity.serialize(),
        "symbolic extents survive index-map parse/print round-trips");

  bool rejected = false;
  try {
    (void)symbolic.instantiate({{"N", 0}});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "symbolic extent bindings must be positive");
}

void testCkMfmaAccumulatorFamily() {
  // CK WarpGemmAttributeMfmaImplF32F32F32M16N16K4 uses {1,4,4,16}.
  Distribution mfma16 = makeMfmaAccumulatorDistribution({1, 4, 4, 16});
  DistributionCheck check16 = verifyDistribution(mfma16);
  check(check16.valid && check16.covering && check16.unique &&
            mfma16.ownership.apply({37, 0, 3}) == std::vector<std::int64_t>({11, 5}),
        "CK MFMA 16x16 accumulator encoding is a unique wave-level tile cover");

  // CK's M32N32 implementations use {4,2,4,32}.
  Distribution mfma32 = makeMfmaAccumulatorDistribution({4, 2, 4, 32});
  DistributionCheck check32 = verifyDistribution(mfma32);
  check(check32.valid && check32.covering && check32.unique &&
            mfma32.ownership.apply({17, 3, 2}) == std::vector<std::int64_t>({26, 17}),
        "the same CK factor encoding composes into the 32x32 MFMA family");
}

void testTaskComposition() {
  Distribution direct = contiguousDistribution(4, 2);
  Distribution permuted = contiguousDistribution(4, 2, true);
  TaskAlternative producerDirect{"dequant", "direct", {},
                                 {{"weight", direct, Placement::Private, 2}}, 16, 0, {}, {}, {}, 0,
                                 "", AlternativeOrigin::Unspecified, ""};
  TaskAlternative producerExpensive{"dequant", "expensive", {},
                                    {{"weight", permuted, Placement::Shared, 2}}, 80, 8192,
                                    {"async-copy"}, {{"scratch", 8192, 0, 2}},
                                    {{EffectKind::Write, "scratch", 0}}, 0, "",
                                    AlternativeOrigin::Unspecified, ""};
  TaskAlternative consumer{"mma", "operand", {{"rhs", direct, Placement::Private, 2}}, {}, 32, 0,
                           {"mma"}, {}, {{EffectKind::Consume, "rhs", 0}}, 0, "",
                           AlternativeOrigin::Unspecified, ""};
  CompositionDecision decision = selectComposition(
      {producerExpensive, producerDirect}, {consumer}, "weight", "rhs", 4, 64, 4096, {"mma"});
  check(decision.selected.has_value() && decision.selected->producerAlternative == 1 &&
            decision.selected->conversion.kind == ConversionKind::Identity &&
            decision.considered.size() == 2,
        "task composition selects the direct legal boundary and records rejected alternatives");
  check(decision.selected->provenance.size() == 5 &&
            decision.considered.front().explanation == "resource limit exceeded",
        "composition decisions retain provenance and rejection reasons");

  TaskAlternative costlyIdentity = producerDirect;
  costlyIdentity.name = "costly-identity";
  costlyIdentity.estimatedExecutionCost = 100;
  TaskAlternative cheapPermutation = producerDirect;
  cheapPermutation.name = "cheap-permutation";
  cheapPermutation.outputs.front().distribution = permuted;
  cheapPermutation.estimatedExecutionCost = 0;
  CompositionDecision costAware = selectComposition(
      {costlyIdentity, cheapPermutation}, {consumer}, "weight", "rhs", 4, 64, 4096, {"mma"});
  check(costAware.selected && costAware.selected->producerAlternative == 1 &&
            costAware.selected->conversion.kind == ConversionKind::LocalPermutation &&
            costAware.selected->score == 10,
        "task composition includes internal execution cost instead of greedily preferring identity");
}

void testLinearPipelineSelection() {
  Distribution direct = contiguousDistribution(4, 2);
  Distribution permuted = contiguousDistribution(4, 2, true);
  Distribution rotated = rotateExecutorOwnership(direct);

  TaskAlternative source{"source", "only", {}, {{"out", direct, Placement::Private}},
                         8, 0, {}, {}, {}, 0, "", AlternativeOrigin::Unspecified, ""};
  source.implementationId = "source.impl.v1";
  TaskAlternative greedyMiddle{"middle", "greedy", {{"in", direct, Placement::Private}},
                               {{"out", rotated, Placement::Private}}, 8, 0, {}, {}, {}, 0, "",
                               AlternativeOrigin::Unspecified, ""};
  greedyMiddle.implementationId = "middle.greedy.v1";
  TaskAlternative globalMiddle{"middle", "global", {{"in", permuted, Placement::Private}},
                               {{"out", direct, Placement::Private}}, 8, 0, {}, {}, {}, 0, "",
                               AlternativeOrigin::Unspecified, ""};
  globalMiddle.implementationId = "middle.global.v1";
  TaskAlternative sink{"sink", "only", {{"in", direct, Placement::Private}}, {},
                       8, 0, {}, {}, {}, 0, "", AlternativeOrigin::Unspecified, ""};
  sink.implementationId = "sink.impl.v1";

  CompositionDecision edgeLocal = selectComposition(
      {source}, {greedyMiddle, globalMiddle}, "out", "in", 4, 64, 4096);
  check(edgeLocal.selected && edgeLocal.selected->consumerAlternative == 0,
        "edge-local selection greedily prefers the identity input boundary");

  PipelineDecision global = selectLinearPipeline(
      {{"source-call", {source}, "", "out"},
       {"middle-call", {greedyMiddle, globalMiddle}, "in", "out"},
       {"sink-call", {sink}, "in", ""}},
      4, 64, 4096);
  check(global.selected && global.selected->alternatives == std::vector<std::size_t>({0, 1, 0}) &&
            global.selected->score == 10 && global.selected->conversions.size() == 2 &&
            global.selected->conversions[0].kind == ConversionKind::LocalPermutation &&
            global.selected->conversions[1].kind == ConversionKind::Identity,
        "linear dynamic programming chooses the globally cheaper layout path");
  check(global.selected && global.selected->provenance.size() == 3 &&
            global.selected->provenance[1].find("middle.global.v1") != std::string::npos,
        "pipeline decisions retain stable invocation and implementation identities");
}

void testTaskGraphFanoutSelection() {
  Distribution direct = contiguousDistribution(4, 2);
  Distribution rotated = rotateExecutorOwnership(direct);
  TaskAlternative directProducer{
      "producer", "direct", {}, {{"out", direct, Placement::Private}},
      8, 0, {}, {}, {}, 150, "producer.direct.v1", AlternativeOrigin::Compiler, ""};
  TaskAlternative cheapProducer{
      "producer", "cheap", {}, {{"out", rotated, Placement::Private}},
      8, 0, {}, {}, {}, 0, "producer.cheap.v1", AlternativeOrigin::Compiler, ""};
  TaskAlternative left{
      "left", "only", {{"in", direct, Placement::Private}}, {},
      8, 0, {}, {}, {}, 0, "left.v1", AlternativeOrigin::Compiler, ""};
  TaskAlternative right{
      "right", "only", {{"in", direct, Placement::Private}}, {},
      8, 0, {}, {}, {}, 0, "right.v1", AlternativeOrigin::Compiler, ""};

  CompositionDecision greedy = selectComposition(
      {directProducer, cheapProducer}, {left}, "out", "in", 4, 64, 4096);
  check(greedy.selected && greedy.selected->producerAlternative == 1,
        "one-edge selection prefers cheap execution plus one subgroup exchange");

  TaskGraphDecision graph = selectTaskGraph(
      {{"producer-call", {directProducer, cheapProducer}},
       {"left-call", {left}}, {"right-call", {right}}},
      {{0, 1, "out", "in"}, {0, 2, "out", "in"}},
      4, 64, 4096, 16);
  check(graph.selected && graph.selected->alternatives ==
                              std::vector<std::size_t>({0, 0, 0}) &&
            graph.selected->score == 150 && graph.selected->conversions.size() == 2 &&
            graph.selected->conversions[0].kind == ConversionKind::Identity &&
            graph.selected->conversions[1].kind == ConversionKind::Identity &&
            graph.combinationsExplored == 2,
        "bounded graph selection charges a fan-out producer once and every outgoing edge");

  TaskGraphDecision bounded = selectTaskGraph(
      {{"producer-call", {directProducer, cheapProducer}},
       {"left-call", {left}}, {"right-call", {right}}},
      {{0, 1, "out", "in"}, {0, 2, "out", "in"}},
      4, 64, 4096, 1);
  check(!bounded.selected && bounded.searchLimitReached &&
            bounded.combinationsExplored == 1,
        "graph selection refuses to commit when its search bound cannot prove optimality");
}

void testCkStyleHierarchicalFixture() {
  IndexSpace executor({{"warp", 2}, {"lane", 4}});
  IndexSpace local({{"value", 2}});
  IndexSpace tile({{"row", 4}, {"column", 4}});
  IndexSpace product({{"warp", 2}, {"lane", 4}, {"value", 2}});
  IndexMap ownership(
      product, tile,
      {IndexExpr::add(IndexExpr::multiply(IndexExpr::input(0), IndexExpr::constant(2)),
                      IndexExpr::input(2)),
       IndexExpr::input(1)});
  Distribution fixture{executor, local, tile, ownership, IndexMap::identity(local)};
  DistributionCheck result = verifyDistribution(fixture);
  check(result.valid && result.covering && result.unique,
        "CK-style (warp,lane) x value to 2-D tile fixture is a unique cover");

  std::ifstream fixtureFile(std::string(CKL_SOURCE_DIR) +
                            "/tests/Core/fixtures/ck_hierarchical_distribution.csv");
  std::string line;
  std::size_t checked = 0;
  while (std::getline(fixtureFile, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::vector<std::int64_t> values;
    std::size_t begin = 0;
    while (begin < line.size()) {
      std::size_t end = line.find(',', begin);
      values.push_back(std::stoll(line.substr(begin, end - begin)));
      if (end == std::string::npos)
        break;
      begin = end + 1;
    }
    check(values.size() == 5 && ownership.apply({values[0], values[1], values[2]}) ==
                                    std::vector<std::int64_t>({values[3], values[4]}),
          "CK golden distribution coordinate " + std::to_string(checked));
    ++checked;
  }
  check(checked == 16, "all CK golden distribution coordinates were loaded and checked");
}

#ifdef CKL_ENABLE_MLIR
void testSymbolicPresburgerProof() {
  // Variables are coordinate d and symbolic extent n. Domain: d >= 0, n >= 1, d < n.
  SymbolicDomain domain{1, 1,
                        {{{1, 0, 0}}, {{0, 1, -1}}, {{-1, 1, -1}}}, {}};
  SymbolicProofResult equal = proveAffineEqual(domain, {{2, 1, 3}}, {{2, 1, 3}});
  SymbolicProofResult unequal = proveAffineEqual(domain, {{1, 0, 0}}, {{0, 0, 0}});
  check(equal.proven && !unequal.proven,
        "MLIR Presburger proves symbolic affine equality and finds a counter-domain");

  IndexSpaceDomain bounds =
      makeIndexSpaceDomain(IndexSpace({{"m", 4}, Axis::symbolic("n", "N")}));
  check(bounds.extentSymbols == std::vector<std::string>({"N"}) &&
            bounds.domain.dimensions == 2 && bounds.domain.symbols == 1 &&
            bounds.domain.inequalities.size() == 5,
        "symbolic IndexSpace extents lower to Presburger coordinate and positivity bounds");
}
#endif

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
    testNestingNormalizationAndHashing();
    testExplicitMutualRefinement();
    testPartialMaps();
    testReplication();
    testXorSwizzle();
    testStorageLayouts();
    testConcreteConversionPlans();
    testStableSerializationAndDiagram();
    testSymbolicIndexSpaces();
    testCkMfmaAccumulatorFamily();
    testTaskComposition();
    testLinearPipelineSelection();
    testTaskGraphFanoutSelection();
    testCkStyleHierarchicalFixture();
#ifdef CKL_ENABLE_MLIR
    testSymbolicPresburgerProof();
#endif
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

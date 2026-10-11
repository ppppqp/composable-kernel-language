# Composable Kernel Language: Self-Contained GPU DSL

**Status:** Revised project specification after the standalone-language pivot

**Date:** 2026-10-11

**Initial target:** A single NVIDIA GPU through CUDA and CUDA Graphs

**Secondary target:** A single AMD GPU through HIP module launches and HIP Graphs

**Future targets:** Multi-GPU execution and communication graph nodes

## 1. Project identity

CKL is a new, self-contained GPU programming language implemented with MLIR. Authors write both
kernel computation and repeated multi-kernel programs in CKL. CKL owns the frontend, typed kernel
and program IR, effect analysis, specialization, device code generation, launch ABI, storage
planning, executable plans, and thin runtimes.

The primary proposition is to compile a CKL program into correct device kernels and a competitive
execution graph without asking its author to maintain separate effect lists, ports, or dependency
metadata. Kernel bodies are the source of truth. Program order defines the conservative reference
behavior; inferred independence permits safe concurrent execution.

Integration with existing GPU DSLs is outside project scope. External semantic adapters, foreign
compiler artifact exports, cross-DSL dispatch normalization, and executable-manifest import have
been removed. Standard upstream MLIR dialects and interfaces remain internal compiler
infrastructure. Caller-owned buffers and streams remain useful application/runtime facilities.

## 2. Goals and non-goals

### Initial goals

1. Provide a small frontend for typed GPU kernels and straight-line multi-kernel programs.
2. Represent indexed memory accesses, device indices, arithmetic, and structured kernel control flow.
3. Infer interprocedural memory effects and aliases from the same bodies compiled to device code.
4. Derive dependencies from SSA, memory conflicts, resource lifetime, and explicit synchronization.
5. Generate device artifacts and physical arguments using CKL-owned backend lowering.
6. Execute a complete CKL program through ordinary CUDA launches and explicit CUDA Graphs.
7. Cache executable graphs and safely reuse compatible temporary storage.
8. Measure complete execution plans and retain decision provenance.
9. Validate numerically against sequential execution and independent native graphs.
10. Port a repeated scientific multi-field workload entirely to CKL source.

### Subsequent goals

- Complete the CKL-to-HIP source compilation path using the retained HIP executor.
- Add atomics, synchronization, shared memory, reductions, and layouts with explicit semantics.
- Add sub-buffer region precision where measurements justify it.
- Tune graph boundaries, iteration batching, and instances in flight.
- Represent bounded host/program control flow explicitly in executable IR.
- Add multi-GPU placement and communication nodes when single-device results warrant it.

### Non-goals

- Integrating, adapting, or retaining compatibility with existing kernel DSLs.
- A producer-neutral compiler service or launch-plan interchange format.
- General tensor-graph optimization, model partitioning, or distributed tensor compilation.
- A static model that predicts detailed kernel runtime or ranks complete execution plans.
- Layout/software-pipeline search as the initial project identity.
- Reordering arbitrary opaque binaries with unknown behavior.
- Multi-node execution or stable public language compatibility in the initial release.

## 3. Design principles

**CKL owns the complete compilation path.** Effects, binary generation, ABI packing, and launch
metadata must agree on the same specialized kernel body. The source API does not expose physical
argument slots or artifact registry construction.

**Bodies are authoritative.** Effect summaries are compiler-owned diagnostic/cache artifacts.
Changing a body, specialization, view, or ABI invalidates them.

**Soundness precedes precision.** Unknown regions overlap the whole resource. Caller-supplied
buffers may alias. Missing semantics produce an error or a conservative barrier, never an
unjustified edge removal.

**Analysis and execution are separate.** Analysis may conservatively represent programs that
cannot yet be executed. Executable verification rejects unsupported semantics before host emission.

**Measurements rank plans.** Static facts establish legality and search bounds. Complete plans are
ranked by target measurements with correctness checks and reproducible cache keys.

**Uncertainty is inspectable.** Reports distinguish proven conflicts, possible aliases, unavailable
regions, control-flow limitations, explicit ordering, and unknown effects.

## 4. Language and frontend

The initial direction is a Python-embedded frontend with explicit kernel and program entry points.
Syntax is provisional; [the language design](../docs/dsl.md) describes the proposed subset. No
kernel decorators or tracing frontend exist in the current Python package.

Kernel IR must express typed scalars and buffers, indexed loads/stores, thread/block/global
indices, arithmetic, comparisons, and structured conditionals/loops. Launch configuration must be
explicitly represented and verified. Out-of-bounds behavior must be defined; examples use bounds
checks rather than assuming partial blocks are safe.

Program IR represents calls to CKL kernels, temporary allocations, caller-owned resources, and
completion ordering. Authors do not supply manual per-kernel read/write declarations. The initial
executable subset uses one device, static geometry and resource sizes, base identity-layout
buffers, and direct pointer/scalar arguments. Unsupported dynamic programs must fail visibly.

The existing whole-buffer `ckl.load` and `ckl.store` operations support effect-analysis fixtures;
they are not sufficient indexed kernel IR. Extending them or introducing suitable indexed
operations is a prerequisite to real CKL device compilation.

## 5. Architecture

```text
CKL kernel + program source
              |
              v
       typed CKL/MLIR IR
              |
              v
  effect summaries + alias analysis
              |
              v
  dependency graph + provenance
              |
              v
       resource memory plan
              |
       +------+-------------------+
       |                          |
       v                          v
CKL device lowering       verified ckl_exec plan
GPU -> NVVM/ROCDL                 |
CUBIN/HSACO artifacts             v
       |                   generated host builder
       +------------+-------------+
                    v
          CUDA/HIP executor
```

Effect extraction runs before target lowering erases memory semantics. Upstream arithmetic,
control-flow, memref, GPU, NVVM, and ROCDL facilities may be reused inside the compiler. CKL's
access-region and dispatch interfaces separate internal analyses from operation implementation;
they are not an external DSL integration contract.

## 6. Effect and alias semantics

Operations expose reads, writes, allocations, frees, or purity through MLIR memory effects. Atomics
report read/write effects and ordering scope. Synchronization and observable implicit state must
be modeled explicitly. Unsupported operations are rejected in strict mode.

The internal access model records kind, base resource, region, ordering scope, and provenance.
Initial regions are whole-buffer or unknown. Later intervals/strided/affine regions are an
optimization aid; lack of precision cannot make a program incorrect.

Each compiler-visible allocation has a fresh resource identity. Casts and views retain base
identity. Caller-owned arguments may alias unless a verified source/ABI assertion proves otherwise.
Private nonescaping allocations do not become dispatch-level resources.

Inference walks reachable kernel operations and regions, resolves base resources, merges control
paths conservatively, records synchronization, and expresses effects relative to formal arguments.
Calls substitute actual arguments into callee summaries. Missing callees and unsupported recursion
remain unknown unless analysis can prove a sound summary.

Conservative mode marks unknown operations as reading and writing all reachable resources and
imposing a barrier. This is an analysis fallback, not permission to execute an unsupported kernel.
Existing summary attributes never override recomputed body semantics.

## 7. Dependency graph

Two accesses conflict when their resources may alias, regions may overlap, and at least one writes,
allocates, frees, or imposes ordering. Read/read does not create an edge. Unknown regions overlap;
proven disjoint resources do not conflict.

`ckl.dispatch` references a CKL kernel body, carries ordered actual arguments and launch metadata,
and returns an SSA completion token. Summaries are bound to each dispatch's arguments.

Edges derive from SSA flow, memory conflicts, resource lifetime, explicit completion ordering,
control flow, and unknown-effect barriers. Direction preserves the source program's observable
reference semantics. Every edge records reasons and endpoint locations. Transitive reduction may
simplify execution while analysis retains provenance.

`ckl.graph` is an analysis snapshot containing normalized nodes, edges, resource descriptors, launch
metadata, and DOT output. The current whole-buffer correctness baseline is implemented, including
a fixed-seed dependency-oracle test. More precise regions require evidence of lost concurrency.

## 8. Storage and executable IR

Static identity-layout integer/float memrefs have compiler-derived sizes and user sets. Dynamic or
unsupported resources receive explicit unavailable diagnostics. No guessed byte sizes are allowed.

Temporary storage is reused only when reachability proves every use of one resource strictly
precedes every use of another and size, alignment, address space, and device are compatible. Reuse
never adds serialization. External and persistent resources are excluded. Plans retain baseline
and planned bytes, heap placement, offsets, prior residents, and reasons.

`ckl-lower-graph-to-exec` emits verified `ckl_exec.plan` operations containing heaps, resource
bindings, artifact/entry-point identities, CKL backend ABI identities, ordered physical argument
slots, static launch geometry, and reduced dependencies. Verification rejects invalid placement,
unknown resources, invalid slots, unsupported packing, and invalid dependency references.

The current executable subset requires unconditional dispatches in one host entry block. Nested
host branches/loops and multi-block CFGs fail until predicates and trip counts have an executable
representation. Whole-buffer alias normalization does not erase argument offsets: views/subviews
and non-identity layouts fail direct-pointer lowering until CKL implements view-aware packing.

The current host path supports pointer resources and i32/i64/f32/f64 scalar bindings/constants.
Computed host scalars, dynamic geometry, and unknown ABIs fail with diagnostics. Backend lowering
must eventually generate these descriptors rather than requiring source authors to supply them.

## 9. Device compilation and runtime

CKL device lowering owns symbol outlining, target specialization, launch metadata, artifact identity,
and logical-to-physical argument packing. An analyzed kernel and its executed binary must derive
from the same body. Hand-written binaries paired with analysis-only bodies are test fixtures, not
an acceptable completed language pipeline.

`KernelArtifact`, `ArtifactRegistry`, `KernelInvocation`, and `ExecutionPlan` are internal handoff
types. An invocation names an artifact, entry point, ABI, device, geometry, physical arguments,
and predecessors. The executor imports artifacts and resolves native functions.

`ckl-hostgen` consumes verified executable IR, allocates declared heaps, retains caller-owned
resources, packs supported arguments, and emits an owning C++ plan builder. It rejects analysis IR.

The CUDA executor supports ordinary topological launches, explicit graphs, caching, parameter
updates, bounded allocation views, and topology batching. HIP supports ordinary launches, explicit
graphs, caching, and owned/borrowed buffers and streams. Borrowed handles must remain alive and
belong to the selected device/context. The driver schedules ready nodes inside a graph.

Complete plans are measured to choose ordinary launches versus graphs, legal storage reuse,
batching, and instances in flight. Every candidate is validated numerically. Cache keys include
source/body, specialization, graph, ABI, binary, target, runtime/driver, and planning options.
Multiple kernel implementations are optional; if introduced, their observable effects must agree.

## 10. Validation and acceptance

The first language gate is a CKL-authored vector update followed by a two-kernel program compiled
from source through device artifacts and executable plans. Compare every output element with a
sequential reference. Verify that inferred summaries, physical arguments, and binary identity all
come from the compiled CKL bodies.

Correctness tests cover distinct/aliased buffers, in-place updates, allocation lifetimes, calls,
control flow, synchronization, and unknown effects. Region-aware execution adds overlapping and
disjoint view tests when supported. Generated graphs are compared with an independent dependency
oracle: missing edges are correctness defects and unnecessary edges are precision limitations.

The first nontrivial workload is a repeated scientific multi-field stencil/solver with small
kernels, independent branches, temporary buffers, and a reduction/convergence step. A saturating
single kernel provides insufficient orchestration opportunity.

Each performance workload compares sequential launches, single-stream capture, manual streams,
a hand-written minimal-dependency graph, an untuned CKL graph, and the measured CKL plan. Report
device time, latency, throughput, host overhead, setup/tuning costs, amortization iterations,
overlap/critical path, peak memory, and multi-instance tail latency.

Acceptance requires:

1. End-to-end compilation and execution of CKL-authored kernels and a composed program.
2. No incorrect executions in numerical and alias/effect validation.
3. Straightforward graphs within 3--5% of the independent hand-written explicit graph.
4. Material improvement over capture on at least two workloads with real graph/storage/batching
   opportunity, before claiming general orchestration success.
5. Reduced temporary memory on a nontrivial CKL workload.
6. Inspectable explanations of conservative dependencies and optimization choices.

Before extensive optimization, native feasibility baselines must establish opportunity. Results
from synthetic or hand-written runtime fixtures do not fulfill CKL language acceptance.

## 11. Retained implementation and new roadmap

The former orchestration Milestones 0--4 provide useful infrastructure:

- Feasibility: `multi_field` explicit graphs measured 1.564x capture on RTX 5060 Ti; the linear
  negative control measured 1.000x. Recorded checksums passed.
- Effects and graphs: CKL memory semantics, interprocedural summaries, aliases, provenance,
  conservative barriers, and randomized structural dependency checks are implemented.
- CUDA runtime: generated upstream GPU CUBIN loading, ordinary launches, explicit graphs,
  caching, and updates are validated. The six-kernel fixture measured CKL graph time at 0.988x
  its independently constructed Driver API graph.
- Planning: temporary storage fell from 16,384 to 8,192 bytes per instance in the recorded fixture;
  batching and in-flight candidates were checked numerically and measured.
- Runtime handoff: physical execution plans, generated host builders, artifact registration,
  borrowed buffers/streams, and the optional HIP executor remain implemented.

Those results remain in `benchmarks/milestone0`, `benchmarks/milestone3`,
`benchmarks/milestone4`, and `benchmarks/hip-runtime-overhead`. External DSL benchmarks and adapters
have been removed. The old milestone numbering records infrastructure history; it does not imply
the standalone language is complete.

The new sequence is:

1. **Kernel semantics:** define indexed memory operations, device indices, typed buffers/scalars,
   structured kernel control flow, and launch rules.
2. **Frontend:** implement the initial source subset and a straight-line CKL program entry point.
3. **CUDA device lowering:** compile the analyzed CKL bodies to CUBINs and generate consistent
   ABI/artifact/launch metadata.
4. **End-to-end execution:** connect compilation to verified plans and host/runtime invocation;
   validate vector update and a two-kernel program.
5. **CKL workload:** replace hand-written kernels with CKL source for the branched scientific
   workload, then re-evaluate memory reuse, batching, and instances in flight.
6. **HIP source path and justified features:** add CKL-to-ROCDL generation, needed language features,
   region precision, and explicit program control flow.

## 12. Risks and decisions

The principal risk is expanding language scope faster than a complete path can be validated. Start
with indexed vector kernels and a composed program before layout search or richer scheduling.
Keep target lowering behind CKL backend stages while reusing upstream MLIR infrastructure.

Effect/binary divergence is a correctness risk. Compile the same specialized bodies used for
analysis and invalidate summaries and plans when source, ABI, or target changes.

Conservative aliases can hide concurrency. Profile false dependencies and introduce only the
region forms needed to remove them. Workload feasibility and robust warm-up/repeated measurements
remain required to distinguish optimization from timing noise.

Open language questions include exact Python syntax, buffer/type notation, explicit launch
configuration, specialization rules, bounds behavior, and the first reduction/shared-memory model.
These must be resolved by the kernel/frontend implementation rather than by adapting a foreign DSL.

Decision record:

1. CKL is a self-contained GPU DSL with its own compiler and runtimes.
2. Kernel bodies and their specialization determine both effects and executed device code.
3. Effect-derived orchestration remains a core language/compiler feature.
4. Existing-DSL integration and launch-plan interchange are removed from scope.
5. Upstream MLIR lowering and standard semantic interfaces remain internal infrastructure.
6. Whole-buffer soundness and a static single-device executable subset come first.
7. Complete-plan measurements rank optimization candidates.
8. CUDA is the first complete language backend; HIP runtime infrastructure is retained.
9. A CKL-authored source-to-device program is the next acceptance gate.
10. Multi-GPU and more elaborate language/graph features follow validated single-device results.

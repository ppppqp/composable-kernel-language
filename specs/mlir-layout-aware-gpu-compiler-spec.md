# Composable Kernel Language: Effect-Derived GPU Orchestration

**Status:** Revised project specification

**Date:** 2026-09-26

**Initial target:** A single NVIDIA GPU through CUDA and CUDA Graphs

**Future targets:** HIP Graph and multi-GPU CUDA/NCCL execution

## 1. Executive summary

CKL is an MLIR-based compiler and runtime for orchestrating repeated GPU programs. It derives
kernel memory effects, aliases, and dependencies from compiler IR; constructs a sound execution
graph; manages temporary-resource lifetimes; and measures complete execution plans on the target
GPU.

CKL is not another general GPU kernel language. Its Python frontend and kernel dialect are one
producer, but other MLIR-based GPU DSLs can reuse the orchestration layer by implementing standard
MLIR interfaces and a small set of CKL interfaces, either directly or through external models.

The project's primary proposition is:

> Derive a competitive GPU execution graph from the behavior already present in kernel IR,
> without requiring programmers to maintain a second set of ports, effects, and dependencies.

Correctness is conservative. CKL may serialize operations when alias or region information is
imprecise, but it must never remove a required dependency. Performance decisions use measurements
of complete plans rather than a static model of intra-kernel execution.

The initial use case is a repeated scientific simulation or iterative solver containing multiple
small-to-medium kernels, temporary buffers, independent branches, and reductions. CKL is compared
with sequential launches, stream capture, manually scheduled streams, and a hand-written explicit
CUDA Graph.

## 2. Motivation

GPU kernel DSLs normally preserve enough information to describe loads, stores, views,
allocations, atomics, and synchronization. That information is often lost at the launch boundary.
Applications then reconstruct ordering manually with streams, events, graph APIs, or separate
read/write declarations.

Manual orchestration has three recurring problems:

1. Effect declarations can be incomplete or become stale as a kernel changes.
2. Conservative program order hides independence between kernels or buffer regions.
3. Runtime decisions are separated from the compiler IR that could justify and explain them.

CKL makes the kernel body the source of truth. It preserves effect semantics across calls and
dialects, binds summaries to actual launch operands, and lowers the resulting dependency graph to
a target command mechanism.

This boundary is intentionally different from intra-kernel optimization. CKL assumes the supplied
kernels are already suitable implementations. It does not need to synthesize tensor-core code,
search layouts, or predict speed-of-light kernel performance to improve launch overhead, resource
lifetime, graph reuse, or legal inter-kernel concurrency.

## 3. Goals and non-goals

### 3.1 Initial goals

1. Infer sound whole-buffer memory effects from CKL and supported upstream MLIR operations.
2. Compute interprocedural summaries and bind them to dispatch operands.
3. Derive dependencies from SSA flow, effects, aliases, and explicit synchronization.
4. Expose a reusable interface contract for other MLIR GPU dialects.
5. Verify that every schedulable operation has known or conservative behavior.
6. Lower a single-GPU graph to ordinary CUDA launches and an explicit CUDA Graph.
7. Cache and update executable graphs for stable or shape-bucketed topology.
8. Infer temporary-buffer lifetimes and safely reuse compatible allocations.
9. Benchmark complete plans and retain inspectable decision provenance.
10. Validate on a repeated scientific mini-application and randomized effect tests.

### 3.2 Medium-term goals

1. Infer static or affine sub-buffer access regions where profitable.
2. Tune graph boundaries, iteration batching, and multiple in-flight instances.
3. Integrate a producer dialect outside CKL through external interface models.
4. Support conditional CUDA Graph nodes for bounded device-side control flow.
5. Add a HIP Graph backend without changing the producer interface.
6. Add multi-GPU placement, P2P copies, and NCCL operations as graph nodes.

### 3.3 Initial non-goals

- Replacing Triton, TileLang, CuTe, FlyDSL, or another kernel DSL.
- General tensor-graph optimization or model partitioning.
- Generating high-performance communication kernels.
- Predicting detailed kernel runtime statically.
- Making layout or software-pipeline search the project identity.
- Optimizing arbitrary opaque binaries with unknown effects.
- Multi-node distributed execution.
- Providing a stable public language or compatibility guarantee.

## 4. Design principles

### Kernel IR is authoritative

Effects are derived from operations in a kernel body. Materialized summaries are compiler-owned
cache and diagnostic artifacts, not annotations that override the body.

### Soundness precedes precision

An unknown region aliases the entire resource. Unknown external arguments may alias one another.
An unknown operation becomes a barrier or a compilation error. Precision is added only when it
enables a measured optimization.

### Interfaces precede dialect-specific cases

Orchestration passes query semantic interfaces rather than individual producer operation classes.
Existing operations may participate through MLIR external models.

### Graph construction and execution are separate

CKL constructs a dependency and resource graph. CUDA or another backend decides when ready work
actually executes. CKL does not replace the GPU driver's internal scheduler.

### Measurement precedes performance modeling

Static information establishes legality and search bounds. Complete plans are ranked by target
measurements using reproducible inputs and cache keys.

### Uncertainty is visible

Reports identify dependencies caused by proven overlap, unknown aliasing, unavailable regions,
foreign effects, or explicit synchronization.

## 5. Architecture

```text
CKL kernel dialect --------+
MLIR GPU/Linalg/Affine ----+--> semantic interfaces
other MLIR GPU dialects ---+          |
                                      v
                            effect and alias analysis
                                      |
                                      v
                            CKL orchestration graph
                              |       |       |
                              |       |       +--> decision report
                              |       +----------> memory planner
                              +------------------> measured plan search
                                      |
                                      v
                         CUDA streams / CUDA Graph
```

The architecture has four layers:

1. **Producer IR:** kernel bodies and launch sites in one or more dialects.
2. **Semantic adapters:** interfaces for effects, dispatches, views, and executables.
3. **Orchestration IR:** resources, dependencies, lifetimes, and candidate plans.
4. **Runtime backends:** executable construction, update, launch, measurement, and caching.

Kernel computation may remain in its producer dialect. CKL requires only enough semantics to
summarize behavior and enough target integration to obtain a launchable artifact.

## 6. Dialect interoperability contract

Interfaces may be implemented directly or registered as external models in an MLIR
`DialectRegistry`.

### 6.1 Memory effects

Operations use MLIR's memory-effect interface to expose reads, writes, allocations, frees, or no
effect on a specific SSA value or abstract resource. Atomics report read and write effects with
their ordering scope. Pure arithmetic and index computation report no effects.

Device globals, random-number state, communication channels, host-visible output, and other
implicit state require explicit abstract resources. Hidden state prevents safe reordering.

### 6.2 Access regions

CKL defines an optional access-region interface whose conceptual result is:

```text
Access {
  kind: read | write | read_write | allocate | free
  resource: SSA value or abstract resource
  region: whole | byte_interval | strided_region | affine_region | unknown
  ordering_scope: optional
}
```

Regions may use static values or symbolic values visible at dispatch. Without the interface, CKL
uses the whole resource. If a region cannot be proven, CKL uses an unknown region. Region
information affects optimization only, never correctness.

### 6.3 Dispatches

A dispatch interface identifies a schedulable GPU invocation and provides:

- referenced kernel body or symbol;
- ordered actual arguments;
- grid and block dimensions;
- dynamic shared-memory bytes;
- target device and required capabilities;
- explicit completion dependencies; and
- stable implementation identity.

CKL summarizes the body before lowering makes memory semantics opaque.

### 6.4 Aliases and views

View-like operations expose their base resource and coordinate mapping. CKL reuses upstream
interfaces for subviews, offsets, sizes, strides, and value bounds where possible.

- Each compiler-visible allocation has a fresh identity.
- A cast, view, or subview retains the identity of its base resource.
- Region mappings compose through nested views.
- External arguments may alias unless their ABI proves otherwise.
- Unresolved pointer arithmetic produces an unknown region.

`noalias` information may improve optimization but is not required for correctness.

### 6.5 Executables

A producer may use CKL's NVIDIA lowering or provide an executable interface containing a binary or
runtime module identity, exported symbol, target features, argument ABI, and static launch metadata.
This separates orchestration from kernel code generation.

### 6.6 Strict and conservative modes

Strict mode requires every reachable operation to be known pure, completely effect-modeled, or a
structured call with an analyzable callee. Anything else is rejected.

Conservative mode treats an unknown operation as reading and writing every reachable resource and
as an ordering barrier. Foreign calls and opaque binaries use this path until an adapter supplies a
trustworthy summary.

## 7. Effect and dependency analysis

### 7.1 Internal model

```text
ResourceId = allocation, external argument, or abstract state resource

Access = {
  kind, resource, region, stage, ordering_scope, provenance
}

KernelSummary = {
  accesses relative to formal arguments,
  synchronization behavior,
  unknown-effect flag
}
```

Stages preserve ordering among multiple implicit effects when a producer exposes it.

### 7.2 Intraprocedural inference

For each kernel body, CKL:

1. walks reachable operations and regions;
2. queries memory effects and access regions;
3. resolves accessed values to base resources;
4. composes transformations through views;
5. records synchronization and abstract-resource effects;
6. merges control-flow paths conservatively; and
7. expresses visible effects relative to formal arguments.

Private allocations that do not escape are not launch-level resources.

### 7.3 Interprocedural inference

Calls substitute actual operands into the callee summary. Summaries are computed in call-graph
order, with recursive components widened to a conservative fixed point. An unavailable callee is
unknown.

Any transformation that changes a body, view, ABI, or specialization invalidates its summary.
Serialized summaries include hashes of those inputs.

### 7.4 Conflict relation

Two accesses conflict when their resources may alias, their regions may overlap, and at least one
access writes, allocates, frees, or imposes ordering. Thus:

- read/read does not create an edge;
- write/read, read/write, and write/write create edges on possible overlap;
- proven-disjoint regions do not conflict; and
- unknown regions are assumed to overlap.

Edge direction preserves observable sequential semantics. CKL removes unnecessary ordering but
does not redefine races already present in the source.

### 7.5 Graph construction

Edges arise from SSA flow, memory conflicts, resource lifetime, explicit synchronization, control
flow, communication ordering, and conservative barriers. Every edge records its reason and source
locations. Transitive reduction may simplify execution, but diagnostics retain original reasons.

Whole-buffer pairwise conflict analysis is sufficient for the first implementation. More precise
region indexes are justified only by workload profiles.

## 8. Orchestration IR

The implemented normalized graph is an analysis snapshot associated with a source function:

```mlir
"ckl.graph"() <{source = @step, name = "step.graph", dot = "..."}> ({
  "ckl.graph_node"() <{id = 0, kind = "alloc", accesses = [...], ...}>
  "ckl.graph_node"() <{id = 1, kind = "dispatch", kernel = @update, ...}>
  "ckl.graph_edge"() <{from = 0, to = 1, reasons = [{kind = "ssa", ...}, ...]}>
})
```

The source program retains allocation, dispatch, and free operations; each dispatch produces a
completion token. The normalized graph records resource identity, ownership and lifetime,
dispatch arguments, dependencies, kernel synchronization metadata, and provenance without
copying producer kernel bodies. Launch parameters and device affinity remain on source dispatches
until executable lowering is introduced.

Function arguments and SSA results replace named logical ports. Types and ABI describe structure;
derived effects describe mutation and ordering.

## 9. Planning and runtime

### 9.1 CUDA backend

The initial backend loads generated cubins, resolves functions, manages device memory, runs kernels
and copies through streams, constructs explicit CUDA Graphs, updates supported node parameters,
caches executables, and records warm-up and steady-state measurements.

CUDA schedules ready graph nodes. CKL controls dependency topology, graph boundaries, launch
parameters, memory, batching, and supplied implementations—not internal hardware scheduling.

### 9.2 Memory planning

Temporary resources may share storage when their lifetimes cannot overlap; size, alignment,
address space, and device are compatible; and reuse does not add a cycle or extend a critical
lifetime. External and persistent resources are never implicitly recycled. Reports show peak
memory before and after planning and explain each reuse.

### 9.3 Measured search

The first search space includes ordinary launches versus graphs, graph and iteration-batch
boundaries, legal memory reuse, number of in-flight instances, and optional producer-supplied launch
variants. Every candidate is validated before execution.

Cache keys include graph and summary hashes, implementation and binary identities, shape buckets,
target architecture, driver/runtime versions, and relevant planning options.

### 9.4 Alternatives are optional

One implementation per dispatch is sufficient. Alternatives are useful only when supplied by
compiler-generated launch configurations, shape specializations, or independent implementations.
Effects are derived from each body. End-to-end measurements, not `estimatedExecutionCost`, rank
plans.

## 10. Frontend model

The existing Python tracer may author CKL kernels and repeated programs, but orchestration semantics
live in MLIR rather than Python metadata.

```python
@ckl.kernel(block_size=(256, 1, 1))
def update(a: MemRef[N, f32], b: MemRef[N, f32]) -> None:
    value = ckl.load(a, ...)
    ckl.store(value + 1.0, b, ...)

@ckl.graph
def step(a, b, c):
    update(a, b)
    update(c, a)
```

The user does not declare that `update` reads `a` and writes `b`; its body provides that
information. The frontend may express real ABI assertions such as ownership or non-aliasing, but
they are verified where possible and never silently assumed.

## 11. Initial use case

The initial application is a repeated scientific simulation or iterative solver. Candidates
include Hotspot3D, an FDTD solver, TeaLeaf, or a focused multi-field stencil mini-application.

The selected workload must have multiple small-to-medium kernels, an independent branch, temporary
buffers, a reduction or convergence calculation, many repeated iterations, and stable or
shape-bucketed topology.

```text
                         +--> update field A --> boundary A --+
previous state ----------+                                    +--> residual
                         +--> update field B --> boundary B --+       |
                                                                       v
                                                              convergence/update
```

At least one version mixes kernels from CKL and an upstream MLIR dialect such as GPU, Linalg, or
Affine, with no per-dispatch effect annotation. A workload dominated by one long saturating kernel
is unsuitable because it provides little orchestration opportunity.

## 12. Validation and evaluation

### 12.1 Correctness

Randomized differential tests compare conservative sequential execution with the CKL graph. They
cover distinct and aliased allocations, overlapping and disjoint subviews, in-place updates,
dynamic indices, allocation reuse, atomics, nested calls, control flow, and unknown operations.

Small generated graphs are checked against a dependency oracle. A missing edge is a correctness
defect; an unnecessary edge is a precision limitation.

### 12.2 Baselines

Every benchmark includes:

1. sequential single-stream CUDA launches;
2. single-stream CUDA Graph capture;
3. manually scheduled multi-stream execution;
4. a hand-written explicit CUDA Graph with minimal dependencies;
5. a CKL-derived graph with planning disabled; and
6. the complete measured CKL plan.

The explicit graph is the primary performance ceiling. IREE is an additional system baseline where
the workload can be represented without changing its character. Taskflow/cudaFlow may compare API
and runtime overhead, but does not replace the manual graph.

### 12.3 Metrics

Report steady-state device time, end-to-end latency, throughput, host submission overhead, graph
construction and instantiation, tuning cost, amortization iterations, critical path, achieved
overlap, peak temporary memory, and multi-instance tail latency.

For dialect reuse, report adapter size, interface-model count, producer source changes, manual
effect declarations, and conservatively modeled operations.

### 12.4 Acceptance criteria

The initial system must:

1. have no incorrect executions in randomized alias/effect tests;
2. run straightforward graphs within 3--5% of the hand-written explicit graph;
3. materially outperform single-stream capture on at least two workloads with genuine graph,
   memory, batching, or in-flight opportunity;
4. reduce peak temporary memory on one nontrivial workload;
5. orchestrate kernels from two dialects through interfaces; and
6. explain all conservative and optimized dependencies.

Before sophisticated analysis begins, candidate workloads get sequential, captured, and manually
optimal implementations. A workload whose explicit graph does not meaningfully beat capture is
rejected as the primary demonstration.

## 13. Roadmap

### Milestone 0: feasibility

- Build sequential, captured, and explicit-graph versions of two candidate workloads.
- Measure launch, overlap, batching, and memory-reuse opportunity.
- Select the primary workload from evidence.

**Implementation status:** complete. The harness implements sequential launches, single-stream
capture, manually coordinated streams, and explicit graphs. On an RTX 5060 Ti with CUDA 12.8, the
underfilled `multi_field` candidate's explicit graph measured 1.564x faster than capture, while the
linear negative control measured 1.000x. All modes passed checksum validation. The raw result is
stored in `benchmarks/milestone0/results/rtx5060ti.csv`; `multi_field` is the selected primary
workload. A full-width branched stencil was rejected because its branches individually saturated
the GPU and exposed no scheduling headroom.

### Milestone 1: semantic foundation

- Add memory effects to CKL memory, allocation, atomic, and synchronization operations.
- Define dispatch and optional region interfaces.
- Implement base-resource and whole-buffer alias analysis.
- Add strict verification, conservative unknown effects, and interprocedural summaries.

**Implementation status:** complete. CKL operations expose standard MLIR read, write, allocate,
and free effects; atomic and synchronization operations retain ordering scope. Reusable
`AccessRegionOpInterface` and `DispatchOpInterface` contracts isolate orchestration from producer
dialects. The `ckl-summarize-effects` pass traces CKL and upstream view-like operations to base
resources, treats compiler allocations as fresh, conservatively aliases external buffers, and
propagates summaries through calls and dispatches. Strict mode rejects missing semantics;
conservative mode emits whole-resource read/write effects, an ordering barrier, and an explicit
unknown marker. Integration tests cover CKL and upstream memory effects, views, private
allocations, calls, dispatches, atomics, synchronization, malformed dispatches, and both unknown
operation modes.

### Milestone 2: sound orchestration graph

- Introduce minimal graph IR.
- Bind summaries to dispatch operands.
- Derive SSA, conflict, lifetime, and synchronization edges.
- Emit provenance and graph visualization.
- Add randomized oracle and differential tests.

**Implementation status:** complete for the whole-buffer correctness baseline. `ckl.dispatch`
produces an SSA `!ckl.token` and accepts token dependencies, replacing symbolic launch names with
producer/consumer links. The `ckl-build-graph` pass recomputes kernel summaries, binds formal
effects to actual operands, and emits normalized `ckl.graph_node` and `ckl.graph_edge` operations.
It derives SSA, explicit-token, possible-alias memory, allocation/free lifetime, conservative
control, and unknown-effect barrier edges. Every edge records its reason and endpoint source
locations; every graph includes a DOT representation. Internal kernel synchronization remains
node metadata because it orders work inside a launch rather than separate launches.

Integration tests cover a deterministic branched graph, provenance, independent allocations,
explicit dependencies, lifetime ordering, and conservative unknown barriers. A fixed-seed
generator compares 20 dispatch graphs against an independent whole-buffer dependency oracle and
rejects both missing and unnecessary dispatch edges. This Milestone 2 check is structural; the
Milestone 3 runtime suite adds numerical differential execution.

### Milestone 3: NVIDIA runtime

- Load CKL-generated cubins and launch through CUDA streams.
- Construct, instantiate, update, and cache explicit CUDA Graphs.
- Establish correctness and overhead parity with manual baselines.

**Implementation status:** complete for the focused single-device kernel-graph runtime. The
optional `CKLNvidiaRuntime` library owns a CUDA primary context, streams, device buffers, loaded
CUBIN modules, resolved functions, argument storage, validated source-ordered plans, instantiated
graphs, and a caller-keyed executable cache. Plans run either as ordinary topological launches or
as explicit CUDA Graphs, and instantiated kernel parameters can be updated without rebuilding the
topology.

The runtime suite compiles a kernel through `ckl-opt`, extracts the resulting `gpu.binary` object,
loads the CUBIN, executes its lowered memref ABI, and validates every result. A second validation
uses the selected six-kernel multi-field topology and compares ordinary launches, the CKL graph,
and an independently constructed raw Driver API graph. On the RTX 5060 Ti, 200 cycles measured
2.718 ms, 1.850 ms, and 1.873 ms respectively; CKL graph time was 0.988x the manual graph and all
checksums matched. Cache identity reuse and `cuGraphExecKernelNodeSetParams` updates are also
tested. The runtime currently accepts a lowering-supplied `NvidiaPlan`; automatic host-code
emission and memory reuse are deliberately left to later milestones.

### Milestone 4: measured optimization

- Add temporary-buffer reuse.
- Tune graph boundaries, iteration batching, and in-flight instances.
- Store reproducible measurements and decision provenance.

**Implementation status:** complete for the focused whole-buffer, single-device search. A
device-independent planner consumes graph edges and each resource's complete user set. It shares
temporary storage only when reachability proves every use of one lifetime strictly precedes every
use of the other; it never adds serialization edges, and persistent or external resources are
never recycled. Plans record heap offsets, baseline and planned bytes, prior slot residents, and a
reason for each decision. The NVIDIA runtime exposes bounded ownership-preserving allocation views
and can repeat a plan topology inside a larger graph boundary.

`ckl-build-graph` invokes this planner automatically for compiler-owned, static, identity-layout
integer or floating-point memrefs. It materializes resources, user node IDs, launch metadata, heap
assignments, offsets, and reuse provenance on `ckl.graph`. Dynamic or unsupported layouts are
marked unavailable with a reason rather than assigned an assumed size. Translating those artifacts
into kernel ABI bindings and emitted C++ host code remains a separate lowering.

The measured validation runs an ordinary, separately allocated reference and checks every tuned
candidate numerically. It searches graph batches 1, 2, 4, and 8 with 1, 2, or 4 independent graph,
stream, and buffer-set instances. On the RTX 5060 Ti, temporary storage per instance fell from
16,384 to 8,192 bytes. For 192 logical iterations, the selected batch-8/four-instance candidate
measured 0.622 ms versus 2.755 ms for batch 1/one instance. All candidate results, graph setup
costs, checksums, and the selection reason are stored in
`benchmarks/milestone4/results/rtx5060ti.csv`. This result targets underfilled kernels and does not
imply that maximal batching or in-flight execution is universally best.

### Milestone 5: cross-dialect reuse

- Integrate a second MLIR GPU dialect through interface models.
- Run a mixed-dialect graph without per-call effect declarations.
- Measure adapter complexity.

### Milestone 6: justified precision

- Profile false dependencies.
- Add only the interval, strided, or affine regions needed to remove them.
- Compare whole-buffer and region-aware results with the manual oracle.

Later work includes conditional and programmatic CUDA Graph features, HIP Graph, multi-GPU device
placement, P2P/NCCL nodes, and topology-aware communication overlap.

## 14. Relationship to existing systems

### MLIR

MLIR supplies the extensible operations, interfaces, analyses, and lowering mechanisms that make a
dialect-neutral layer possible. CKL reuses upstream interfaces before defining its own.

### IREE

IREE Stream already models asynchronous resources, lifetimes, scheduling, and commands. CKL's
intended distinction is a smaller embeddable layer for composing already-lowered kernels from GPU
DSLs without adopting an end-to-end tensor compiler/runtime. Expanding into general tensor lowering
would erase that distinction.

### CUDA Graphs

CUDA Graphs provide reusable command graphs. CKL derives their topology and resources; it does not
replace CUDA's ready-node scheduler. Native explicit graphs remain the reference.

### XLA and framework compilers

XLA and major ML frameworks already capture tensor programs. CKL initially targets scientific and
custom DSL pipelines where adopting a complete ML framework compiler is disproportionate.

### Task graph libraries

Taskflow and similar libraries simplify manual graph construction. CKL's contribution is deriving
effects and dependencies from compiler IR, not another manual graph API.

## 15. Migration from the current prototype

Reusable pieces include Python tracing, CKL memory operations, NVIDIA binary generation, callable
bodies, graph traversal, resource lifetimes, and decision provenance.

The following are no longer foundational APIs:

- manually named ports;
- manually maintained effect lists;
- target-independent execution-cost estimates;
- required implementation alternatives; and
- layout-conversion search at every task boundary.

Layout, distribution, proof, and target-extension libraries may remain optional facilities for CKL
kernels. Other dialects do not need to adopt them.

Migration is incremental: add inferred summaries beside existing metadata, construct and validate
the effect-derived graph, implement the runtime, move callers to argument/effect dependencies, and
retire manual metadata after equivalent tests exist.

## 16. Risks and mitigations

### Duplicating IREE

Keep the scope on reusable kernel orchestration, study IREE analyses first, and compare where
feasible.

### Excessively conservative effects

Start with whole buffers, profile lost concurrency, and add only region forms justified by measured
false dependencies.

### Little CUDA Graph headroom

Run manual feasibility baselines first and reject workloads where explicit dependencies, batching,
or memory planning cannot improve capture.

### Insufficient cross-dialect semantics

Integrate a second dialect early. Strict mode makes missing semantics fail visibly.

### Runtime work overwhelms compiler work

Limit the initial runtime to one process, one NVIDIA device, kernel/copy nodes, stable topology,
and a small allocator.

### Noisy measurement

Use warm-up, repeated samples, robust statistics, fixed clocks where possible, explicit budgets,
and target-specific cache keys.

## 17. Open questions

1. Should the graph reuse upstream `async`/`gpu` concepts or define a small CKL resource layer?
2. Which upstream alias and value-bounds analyses are stable enough initially?
3. Should strict mode reject indirect calls or permit a closed callee set?
4. How should symbolic regions interact with specialization and graph caching?
5. When should graph construction occur relative to outlining and target lowering?
6. How should optional implementations prove observable effect compatibility?
7. Which mini-application exposes genuine opportunity without a large external runtime?
8. Would an IREE import/export path help reuse or blur the project boundary?

## 18. Decision record

1. CKL is an orchestration layer first and a kernel DSL second.
2. Kernel bodies, not manual effect declarations, are authoritative.
3. Function arguments and SSA results replace required named ports.
4. MLIR interfaces and external models provide cross-dialect extension.
5. Whole-buffer conservative analysis is the first correctness baseline.
6. Region precision is added only for measured false dependencies.
7. Multiple implementations and their selection are optional.
8. Complete plans are measured; static costs do not rank final performance.
9. The first backend is single-GPU NVIDIA CUDA Graph execution.
10. The first workload is a repeated scientific simulation or iterative solver.
11. Hand-written explicit CUDA Graphs are the primary performance ceiling.
12. Multi-GPU execution is a later graph extension, not the first milestone.

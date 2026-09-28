# Effect-Derived GPU Orchestration

This document describes CKL's target architecture. Milestone 0 feasibility measurement, the
Milestone 1 semantic foundation, and Milestone 2 graph construction are implemented. Planning and
runtime execution remain design targets unless explicitly noted otherwise.

The normative project scope and acceptance criteria are maintained in the
[revised project specification](../specs/mlir-layout-aware-gpu-compiler-spec.md).

## Goal

CKL will turn a program containing GPU kernel dispatches into a sound, inspectable, and measured
execution plan. Kernel bodies remain the source of truth for memory behavior. CKL derives the
dependencies between dispatches instead of requiring authors to maintain separate read/write or
port declarations.

The project is deliberately narrower than an end-to-end tensor compiler. CKL does not need to
invent kernel schedules, replace an existing GPU DSL, or predict the runtime of a kernel from a
static hardware model. It operates at the boundary between independently compiled kernels and a
GPU command runtime.

## Compilation model

```text
producer dialects
    |
    | operations implementing standard MLIR and CKL interfaces
    v
effect extraction and alias analysis
    |
    | per-kernel summaries bound to actual dispatch operands
    v
orchestration graph
    |
    +--> dependency and region analysis
    +--> resource lifetime and memory planning
    +--> graph formation and target profiling
    v
backend executable and decision report
```

Kernel bodies may remain in their producer dialects while CKL constructs a normalized graph of
dispatches and resources. A producer should not have to lower its computation into CKL's kernel
dialect to use the orchestration layer.

## Dialect interoperability contract

Supporting a new MLIR-based GPU DSL requires semantic interfaces rather than CKL-specific
operation names. Interfaces may be implemented by the dialect or attached through MLIR external
models.

### Memory effects

Operations use MLIR's memory-effect interface to report reads, writes, allocations, and frees on
specific SSA resources. Pure operations report no effects. Atomics report both read and write
effects and preserve their ordering scope.

The interface must describe every implicit effect that is relevant across dispatch boundaries,
including device globals, communication state, random-number state, and externally observable
I/O. A DSL that cannot represent such state must conservatively expose it as an unknown resource.

### Access regions

Standard memory effects identify a resource but do not always identify the precise part of that
resource that is accessed. CKL will define an optional access-region interface capable of
returning:

- a whole-resource access;
- a static byte or multidimensional interval;
- an affine region derived from an indexing map and iteration domain; or
- an unknown region.

Region information is an optimization aid. If it is absent or cannot be proven, CKL assumes the
whole resource is accessed. Disjoint-region concurrency must never be required for correctness.

### Dispatches

A dispatch interface identifies a schedulable kernel invocation and provides:

- the referenced kernel body or symbol;
- the actual arguments;
- grid, block, and dynamic shared-memory parameters;
- target device and required capabilities; and
- explicit completion or synchronization dependencies that cannot be inferred from memory.

CKL computes an interprocedural effect summary for the referenced body and substitutes the actual
arguments at each dispatch site. Nested calls are summarized to a fixed point. Recursion or an
unavailable callee body produces conservative effects.

### Aliases, views, and ownership

Alias analysis traces casts, views, and subviews to an underlying resource and transforms their
access regions accordingly. A compiler-owned allocation has a fresh identity. External pointer or
memref arguments may alias unless their type or calling convention provides a stronger guarantee.

Unknown aliasing reduces available parallelism but remains correct. CKL must not require `noalias`
annotations for correctness.

### Strict and conservative modes

Strict mode rejects a schedulable program if an operation has neither known purity nor complete
effect semantics. It is intended for compiler testing and reproducible optimization.

Conservative mode permits unknown operations but treats them as reading and writing all reachable
resources and as ordering barriers. Foreign libraries, inline assembly, and opaque binaries enter
through this path until an adapter supplies a better summary.

Compiler-derived summaries may be materialized as attributes for diagnostics and caching, but
they are never authoritative source annotations. Any transformation that changes a body
invalidates and recomputes its summary.

## Orchestration graph

Graph edges arise from:

- direct SSA dependencies;
- write-to-read, write-to-write, and read-to-write conflicts on possibly overlapping regions;
- allocation and lifetime constraints;
- explicit synchronization, communication, or control dependencies; and
- conservative barriers for unknown behavior.

Read-to-read accesses do not conflict. Accesses to provably disjoint regions do not conflict.
Function arguments and SSA results replace manually named logical ports in the target design.

The initial implementation should infer whole-buffer dependencies. Precise affine or symbolic
regions should be added only after profiles identify false dependencies that matter in a real
workload.

The implemented `ckl-build-graph` pass emits one `ckl.graph` per host function containing
dispatches. Dispatches produce SSA `!ckl.token` completion values, and consumers list token
operands for dependencies that are explicit rather than memory-derived. Graph nodes retain their
bound accesses, synchronization metadata, and source location; graph edges retain all applicable
reasons and endpoint locations. Internal kernel synchronization is descriptive metadata, not an
automatic dependency between separate launches. Unknown effects become cross-node barriers in
conservative mode.

The current conflict model is deliberately whole-buffer. Distinct CKL allocations are proven
disjoint, views retain their base identity, and external resources may alias. Nested control flow
is preserved conservatively with source-order control edges. The pass also emits a DOT rendering
for inspection; it does not execute the graph.

## Execution planning

The first backend targets a single NVIDIA GPU and can choose among ordinary CUDA launches and an
explicit CUDA Graph. Planning includes:

- graph boundaries and executable-graph caching;
- parameter updates for stable topology;
- temporary-buffer lifetime and reuse;
- legal concurrency exposed by the dependency graph;
- iteration batching or graph unrolling; and
- multiple graph instances and buffer sets for in-flight work.

CUDA retains control of ready-node scheduling inside a graph. CKL optimizes the topology,
resources, graph configuration, and supplied implementations; it does not claim to replace the
GPU driver's internal scheduler.

Static resource data is used for legality and diagnostics. Performance choices are based on
target measurements of complete plans. Search results are cached by kernel identity, shapes,
target, driver/runtime version, and relevant graph configuration.

Multiple implementations are not required. If a producer supplies compiler-generated launch
variants or independently authored CKL implementations, their effects are derived from their
bodies and their observable contracts must be compatible. Selection is empirical and operates on
end-to-end graph performance rather than a manually assigned execution cost.

## Initial workload

The initial workload should be a repeated scientific simulation or iterative solver with a
pipeline resembling:

```text
                         +--> update field A --> boundary A --+
previous state ----------+                                    +--> residual
                         +--> update field B --> boundary B --+       |
                                                                       v
                                                              convergence/update
```

Candidates include Hotspot3D, an FDTD solver, TeaLeaf, or a focused multi-field stencil
mini-application. The chosen program should contain multiple sub-millisecond kernels, temporary
buffers, at least one independent branch, a reduction, and many repeated iterations. A program
dominated by a single saturating kernel is not an orchestration benchmark.

At least one evaluation program should mix kernels originating in two dialects, initially CKL and
an upstream MLIR dialect such as GPU, Linalg, or Affine. The graph must be constructed through
interfaces with no per-call effect annotation.

## Validation plan

### Correctness

Randomized differential tests compare a conservative sequential execution with the derived graph.
They cover:

- aliased arguments and overlapping subviews;
- disjoint subviews;
- in-place updates;
- allocation, reuse, and deallocation;
- atomics and synchronization;
- nested calls and control flow; and
- unknown or foreign operations in both strict and conservative modes.

Small generated graphs are also checked against a dependency oracle. A missing edge is a
correctness failure; an unnecessary edge is a precision limitation.

### Baselines

Each workload is compared with:

1. sequential single-stream CUDA launches;
2. single-stream CUDA Graph capture;
3. a manually scheduled multi-stream implementation;
4. a hand-written explicit CUDA Graph with minimal dependencies;
5. a CKL-derived graph without profile-guided choices; and
6. the complete CKL plan.

The explicit graph is the performance ceiling for graph construction. Where the workload can be
represented without changing its character, IREE's asynchronous resource scheduling is a useful
system-level comparison.

### Measurements

Report:

- steady-state device time and end-to-end latency;
- host submission overhead;
- graph construction, instantiation, and tuning cost;
- peak temporary memory;
- achieved concurrency and critical path;
- throughput with multiple instances in flight; and
- the number of iterations required to amortize setup.

Dialect integration is measured separately by adapter size, the number of external interface
models, required source changes, and the number of manual effect declarations.

### Success criteria

The initial system should:

- produce no incorrect executions in randomized alias and effect tests;
- run straightforward graphs within 3--5% of a hand-written explicit CUDA Graph;
- demonstrate a material improvement over single-stream capture on at least two workloads where
  dependency precision, memory reuse, batching, or in-flight execution exposes real opportunity;
- show a concrete reduction in temporary memory on a workload with reusable lifetimes; and
- orchestrate kernels from at least two dialects through interfaces.

Before building sophisticated analyses, manually implement the sequential, captured, and optimal
explicit-graph versions of candidate workloads. If the explicit graph provides no meaningful
advantage over capture, that workload cannot validate effect-aware orchestration and should be
replaced.

## Scope and roadmap

### Phase 1: sound graph construction

Milestones 1 and 2 are implemented, completing the initial sound whole-buffer graph construction
phase.

- Define and document the dispatch and optional access-region interfaces.
- Add memory effects to CKL operations.
- Infer whole-buffer, interprocedural summaries.
- Implement strict verification and conservative unknown effects.
- Add randomized dependency and alias tests.
- Bind summaries to actual dispatch operands and materialize graph nodes and provenance edges.
- Check generated dispatch graphs against a fixed-seed dependency oracle.

### Phase 2: executable runtime

- Load and launch generated NVIDIA binaries.
- Lower the orchestration graph to ordinary CUDA launches and explicit CUDA Graphs.
- Support parameter update, caching, and measurement.
- Establish the manual baseline on the initial mini-application.

### Phase 3: useful optimization

- Add buffer lifetime planning and reuse.
- Add only the region precision needed by measured false dependencies.
- Tune batching, graph boundaries, and in-flight instances.
- Integrate a second producer dialect through external interface models.

### Later work

- Additional CUDA Graph features such as conditional or programmatic dependencies.
- HIP Graph or another command-graph backend.
- Multi-GPU placement and P2P/NCCL nodes.
- Communication/computation overlap and topology-aware planning.

Generating communication kernels, implementing a general distributed tensor compiler, and
building a static model of intra-kernel performance are not initial goals.

## Migration from the current prototype

Reusable pieces include Python tracing, GPU binary generation, task-graph discovery, callable
implementations, resource lifetimes, and decision provenance. The layout and proof libraries may
remain useful to CKL-authored kernels, but they are not prerequisites for the orchestration layer.

The following concepts should be removed from the future public orchestration API or made
optional:

- manually named input and output ports;
- manually maintained per-kernel memory-effect lists;
- target-independent estimated execution costs; and
- layout-conversion search as the primary reason for composing tasks.

This separation allows the current layout work to remain available without requiring every
producer dialect to adopt it.

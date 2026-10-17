# IREE Stream and CKL execution planning

These are source-reading notes and proposed directions, not an approved implementation plan.
Reviewed the local sibling checkout at `~/Documents/workspace/iree`, revision
`5cf0a123b0c53ea8cb6a02ae923f46f7b3224eb8`. No IREE code was changed or dependency added.
The relative source links below assume that checkout remains next to CKL.

## Is CKL a graph compiler?

CKL is a GPU language/compiler with a graph-compilation layer. It owns both computation inside
kernels and composition between kernels: source lowering, effects, aliases, dependency inference,
temporary storage, device code, executable plans, host JIT, and execution. Calling it a graph
compiler describes its program-level responsibilities but omits its kernel-language/backend work.

The graph is an execution graph derived from an imperative program's memory effects and ordering.
It is not a requirement to express the application as a high-level tensor/operator graph. Program
order supplies reference semantics; inferred independence permits concurrent execution. A CUDA
Graph is one execution mechanism for that plan; ordinary launches execute the same compiled work.
See [orchestration](orchestration.md), [CUDA execution](cuda-backend.md), and the
[specification](../specs/mlir-layout-aware-gpu-compiler-spec.md).

## Where Stream sits

IREE's [dialect definition](../../iree/compiler/src/iree/compiler/Dialect/Stream/IR/StreamDialect.td)
places Stream between Flow and HAL:

```text
Flow: dispatchable tensor computation and host dataflow
  → Stream: resources, asynchronous dependencies, execution groups, storage
  → HAL: device buffers and command submission through the runtime abstraction
```

Stream is closest to CKL's effect/graph/storage planning and executable-plan layers, rather than
to CKL's Python frontend or device arithmetic lowering. Its
[pipeline](../../iree/compiler/src/iree/compiler/Dialect/Stream/Transforms/Passes.cpp) progresses
from `stream.tensor.*` through `stream.async.*` to `stream.cmd.*`. Resource contents, scheduling,
and storage are made explicit in stages before lowering to HAL.

## What the implementation provides

| Facility | Evidence in the local source | Relation to CKL |
| --- | --- | --- |
| Resource classes and sizes | [StreamTypes.td](../../iree/compiler/src/iree/compiler/Dialect/Stream/IR/StreamTypes.td): external, staging, transient, variable, constant; ops carry resource sizes | CKL already distinguishes external resources and planned temporaries; persistent state and staging deserve explicit treatment as workloads grow. Stream sizes can be runtime SSA values. |
| Access ranges and views | [StreamOps.td](../../iree/compiler/src/iree/compiler/Dialect/Stream/IR/StreamOps.td): dispatch resource offsets/ends/lengths and `resource.subview`; [ResourceHazards.cpp](../../iree/compiler/src/iree/compiler/Dialect/Stream/Analysis/ResourceHazards.cpp) checks overlapping accesses and excludes read/read hazards | CKL currently uses whole-buffer graph conflicts and rejects views in executable packing. Byte ranges must agree across analysis, ABI packing, and execution before enabling finer concurrency. |
| Execution grouping and concurrency | [Passes.td](../../iree/compiler/src/iree/compiler/Dialect/Stream/Transforms/Passes.td): ScheduleExecution and ScheduleConcurrency; `async.execute`, `async.concurrent`, `cmd.serial`, `cmd.concurrent` in StreamOps | CKL already derives a DAG. A useful next abstraction could be groups of commands with explicit dependencies and submission boundaries. A concurrency permission does not guarantee hardware overlap. |
| Completion dependencies across the program | `!stream.timepoint` in StreamTypes; propagate/elide timepoint passes in Passes.td | A timepoint means completion including transitive predecessors. Propagation moves waits toward consumers across calls, globals, and control flow; elision removes redundant waits. CKL's dispatch tokens currently describe internal ordering, while its public Python execution synchronizes every run. |
| Transfers as planned work | `async.transfer`, copy/fill ops, and PlaceTransferExecutions in Passes.cpp | CKL's current Python copies are synchronous outside the compiled plan. Explicit transfer nodes would let a later pipeline overlap movement and computation while retaining ownership/completion rules. |
| Copy and destination handling | MaterializeCopyOnWrite, ElideAsyncCopies, EmplaceAllocations in Passes.cpp | Stream first preserves value semantics, then removes provably unnecessary copies. CKL's explicit mutable buffers have different semantics; output placement/copy elimination may transfer as ideas, but the entire copy-on-write model is not an automatic fit. |
| Lifetime-aware allocation | ScheduleAllocation, ReuseAllocations, LayoutSlices, EmplaceTransients, and AutomaticReferenceCounting in Passes.cpp; `resource.pack` in StreamOps | Allocation/deallocation participates in the timeline. Packing supports dynamic slice sizes and lifetime intervals. CKL already reuses static temporary storage using graph reachability; the major extensions are dynamic/control-flow lifetimes and asynchronous ownership. |
| Placement and encoding | Affinity attributes and host/device encoding stages in Passes.cpp | Useful later for memory spaces or multiple devices. The local pipeline still contains placement TODOs; these interfaces are not evidence of an automatic optimal placement algorithm. |

Stream also models collectives, parameter/file operations, and dispatch argument/binding
optimization. Those are useful reference material, but do not justify expanding CKL's initial
single-device scope.

## Two distinctions that matter

**Grouping dispatches does not fuse kernel computation.** A `stream.cmd.execute` region may
contain multiple dispatches and copies, retaining their individual computation and storage
boundaries. Likewise, the
[FuseDispatchBindings pass](../../iree/compiler/src/iree/compiler/Dialect/Stream/Transforms/FuseDispatchBindings.cpp)
combines correlated bindings to storage; it does not replace a producer/consumer pair with one
GPU kernel. CKL pointwise fusion changes device bodies and eliminates temporary store/load
boundaries. Both optimizations can coexist, but solve different costs.

**Rich IR does not imply an optimal scheduler.** In this checkout,
[Partitioning.cpp](../../iree/compiler/src/iree/compiler/Dialect/Stream/Analysis/Partitioning.cpp)
selects the reference algorithm for both execution partitioning and concurrency partitioning.
[Partitioning.h](../../iree/compiler/src/iree/compiler/Dialect/Stream/Analysis/Partitioning.h)
describes these as correctness-driven algorithms without a cost model. Several passes support
control flow, but the pipeline also lowers SCF to CFG because some later passes do not handle
SCF directly. Adopt explicit semantics and verification ideas without assuming universal
scheduling or control-flow support.

## Recommended sequence for CKL

1. **Establish application headroom before extending optimization.** Implement
   [SGD](../benchmarks/momentum-sgd/README.md) and the
   [preprocessing component](../benchmarks/image-preprocessing/README.md), with natural operation
   boundaries and manually optimized controls. These determine whether broader pointwise fusion,
   better kernels, or submission overhead is the most valuable next target.
2. **Add a small asynchronous composition API if measured submission overhead warrants it.**
   Separate submission from waiting and represent completion across invocations. Retain code,
   buffers, and temporary heaps until completion; prevent simultaneous reuse of one invocation's
   mutable storage or provide separate instances. Keep dependencies explicit and device-side
   where possible. Existing internal dispatch tokens and native stream/runtime support are a
   starting point, not proof that the public ownership contract is complete.
3. **Add bounded program repetition driven by scientific workloads.** Represent trip counts and
   loop-carried persistent buffers in executable IR; implement frontend lowering, verification,
   and runtime behavior together. Start with fixed counts for miniWeather or CG, then handle
   convergence separately. Replaying a fixed plan must not freeze optimizer counters or scalar
   bindings. Command grouping can reduce submission cost without changing kernel computations.
4. **Add offset-aware bindings and region precision when traces show false dependencies or
   copies.** Preserve base identity, byte offset, byte length, and access kind through effects,
   graph analysis, memory planning, physical arguments, and runtime bounds checks. Begin with
   contiguous subranges. Unknown or externally aliased ranges remain conservative.
5. **Bring transfers and asynchronous allocation into plans when pipelines need them.** Reuse
   the same completion/ownership model rather than adding separate special cases for uploads,
   downloads, and storage recycling. Defer multi-device affinity and tensor encoding machinery
   until an actual CKL workload needs them.

Broader fusion is a separate branch of this sequence: multi-input/output same-index kernels,
escaping persistent outputs, and branches/joins need explicit legality and alias handling.
Reductions and stencils require additional kernel semantics and cannot be solved by execution
grouping alone. The [application benchmark proposals](../benchmarks/README.md) are intended to
decide which branch deserves implementation effort first.

## Source-reading scope

Read the dialect/types/operation definitions, pipeline and pass descriptions, resource hazards,
partitioning entry points, binding fusion implementation, and representative execution-scheduling
and SCF transient-allocation tests. This was a static source review: no IREE build, runtime
benchmark, or end-to-end capability validation was performed. Conclusions apply to this local
revision, not necessarily upstream HEAD.

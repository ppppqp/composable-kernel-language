# Execution planning inside the CKL DSL

CKL owns both kernels and their composition. Execution planning is a compiler stage of the new
language. The language direction is described in [dsl.md](dsl.md), and the normative requirements
are in the [project specification](../specs/mlir-layout-aware-gpu-compiler-spec.md).

## Compilation model

```text
CKL source
    -> typed CKL kernel/program IR
    -> kernel effect summaries and base-resource aliases
    -> dispatch dependency graph with provenance
    -> resource lifetimes and memory plan
    -> verified ckl_exec plan + CKL-generated device artifacts
    -> in-process LLVM host JIT with embedded device code
    -> CUDA executor (HIP runtime retained separately)
```

Upstream MLIR dialects are internal tools for arithmetic, structured control flow, memory
semantics, and target lowering. CKL's access-region and dispatch interfaces organize its own
compiler passes. There is no external DSL adapter, external dispatch registration, or JSON
launch-plan import path.

## Effect inference

Kernel bodies are authoritative. Standard MLIR memory effects and CKL region semantics describe
reads, writes, allocations, frees, atomics, and synchronization. Calls substitute actual arguments
into interprocedural summaries. Private, nonescaping allocations do not become program resources.
Compiler-owned summary attributes are recomputed after body changes.

Alias analysis traces views and casts to base resources. Compiler allocations have fresh
identities; caller-supplied buffers may alias. The first correctness baseline is whole-buffer
analysis. Region precision is added only when measured false dependencies justify it.

Strict mode rejects missing semantics. Conservative mode marks unknown operations as reading and
writing reachable resources and imposing an ordering barrier. This supports diagnostics and
analysis; it does not make opaque operations executable.

## Graph and storage planning

`ckl.dispatch` references a CKL kernel body and returns an SSA completion token. The graph builder
binds formal effects to actual arguments and adds SSA, memory-conflict, explicit-token, lifetime,
control, and unknown-effect barrier edges. Each edge retains its reasons and source locations;
graphs include DOT output. Internal kernel synchronization remains node metadata.

The memory planner uses graph reachability and complete resource user sets. Temporaries share
storage only when all uses of one lifetime precede all uses of the other and size, alignment,
address space, and device permit reuse. It never adds ordering to force reuse. External and
persistent resources are excluded. Reports retain original and planned bytes and reuse reasons.

Static identity-layout integer and float memrefs are planned automatically. Dynamic or
unsupported resources receive an unavailable reason rather than an assumed size.

## Analysis-to-execution boundary

`ckl-lower-graph-to-exec` converts analyzed graphs to verified `ckl_exec.plan` operations with heaps,
resource slices, launch metadata, CKL backend ABI identities, physical argument slots, and reduced
dependencies. The native compiler consumes this executable IR and JIT-compiles an LLVM host
entry point with the device binary embedded.

The current static subset requires unconditional host-entry-block dispatches. Conditional or
repeated host nodes are rejected until executable IR can represent predicates and trip counts.
Base identity-layout pointer arguments are supported; views/subviews require offset-aware packing
and currently fail executable lowering. Supported scalar bindings/constants are i32, i64, f32,
and f64. Computed host scalars and dynamic launch/resource metadata remain unsupported.

CKL CUDA preparation generates ABI identities and artifact metadata automatically. Some
low-level MLIR fixtures carry these attributes explicitly. They are compiler plumbing, not the
intended source-language API.

## Runtime

`KernelArtifact`, `ArtifactRegistry`, `KernelInvocation`, and `ExecutionPlan` are internal compiler
and runtime handoff types. The executor loads CUBIN or HSACO objects, resolves entry points, and
passes packed arguments to the device driver. The JIT host entry point allocates declared heaps,
binds external buffers, and constructs native launches. Python invocation objects retain code,
buffers, heap allocations, streams, and graphs through completion.

The optional CUDA and HIP executors support ordinary launches, explicit command graphs,
executable caching, and owned or borrowed buffers and streams. CUDA additionally has parameter
update and topology batching support. Borrowed storage and streams must remain alive during
execution and belong to the selected device/context.

CKL controls topology, resources, graph boundaries, batching, and instances in flight. The driver
schedules ready nodes. Performance choices use measurements of complete plans; static resource
information establishes legality rather than predicted speed.

## Validation plan

Correctness checks compare sequential execution with derived graphs and cover aliasing, views,
in-place updates, allocations, calls, synchronization, and conservative unknown effects. Small
generated graphs are checked against a dependency oracle. Executable validation rejects unsupported
control flow or argument packing before host code is emitted.

Benchmarks compare sequential launches, single-stream capture, manual streams, a hand-written
explicit graph, an unoptimized CKL graph, and the measured CKL plan. Report steady-state device
time, end-to-end latency, host overhead, setup/tuning cost, peak temporary memory, overlap,
throughput, and amortization iterations.

The retained Milestone 0--4 benchmarks establish orchestration feasibility and runtime behavior.
Their hand-written device kernels do not establish language completeness. The initial
[CUDA acceptance test](cuda-backend.md#acceptance-test) now executes a CKL-authored two-kernel
program whose analyzed bodies generate the binary. The scientific multi-field workload is next.
See [the language implementation sequence](dsl.md#implementation-sequence).

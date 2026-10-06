# FlyDSL and HIP benchmark plan

## Purpose

This benchmark program should establish three claims separately:

1. CKL adds negligible steady-state overhead when it executes an already-resolved HIP plan or a
   retained HIP Graph.
2. CKL can execute real multi-kernel FlyDSL operators correctly and improve launch-sensitive
   workloads without changing their kernels, buffers, or streams.
3. Improvements remain visible in an application using FlyDSL through AITER and a model-serving
   framework.

Setup cost, steady-state overhead, scheduling benefit, and end-to-end impact must not be collapsed
into one number.

## Runtime overhead model

Manifest parsing, HSACO loading, artifact registration, function lookup, plan resolution, and graph
instantiation are setup operations. A retained `HipGraphExecutable` does not traverse the manifest
or `ExecutionPlan` during replay. Its hot path performs CKL ownership/context checks, selects the
HIP device, and calls `hipGraphLaunch` on the supplied stream.

This path should approach native HIP Graph or PyTorch HIP Graph performance after setup is
amortized, but exact equality is not assumed. `hipSetDevice`, the CKL wrapper, a future Python FFI
boundary, and differences between explicit graph construction and stream capture may be measurable
for very small workloads.

Ordinary execution has an important API distinction:

```cpp
// Setup plus execution: do not put this overload in the timed loop.
runtime.launchOrdinary(executionPlan, artifacts, stream);

// Resolve once, then measure only repeated submission.
HipPlan resolved = runtime.resolve(executionPlan, artifacts);
runtime.launchOrdinary(resolved, stream);
```

The first overload resolves artifacts on every call. All steady-state ordinary-launch benchmarks
must use a retained `HipPlan`. All graph benchmarks must retain the instantiated
`HipGraphExecutable` rather than perform a cache lookup or instantiation inside the timed loop.

## Stage 0: isolate CKL runtime overhead

Status: implemented in [`hip-runtime-overhead`](hip-runtime-overhead/README.md), with raw RX 9060
XT samples retained in the repository. The result establishes native-runtime parity for retained
C++ plans and graphs; the FlyDSL and Python integration claims remain for later stages.

Use one HSACO, one imported stream, identical buffers and arguments, and linear graphs containing
2, 4, 8, and 16 small kernels. Compare:

1. direct `hipModuleLaunchKernel` calls;
2. CKL ordinary submission from a pre-resolved `HipPlan`;
3. a manually constructed HIP Graph launched with `hipGraphLaunch`; and
4. a retained CKL `HipGraphExecutable`.

Measure both GPU event time and CPU submission time. Run enough launches per sample that the timer
resolution is not material, report medians plus tail percentiles, and retain the raw samples. File
I/O and module loading belong in separately reported setup measurements.

The initial acceptance criteria are:

- identical numerical results in all four modes;
- CKL graph replay within 5% of the manual HIP Graph median;
- CKL pre-resolved ordinary submission within 5% of direct HIP submission for graphs whose kernel
  time is large enough to measure reliably; and
- no allocations, module loads, function lookup, or graph construction in a timed loop.

This experiment validates runtime parity only. It does not demonstrate a scheduling benefit.

## Stage 1: same FlyDSL artifact, four execution modes

Status: implemented for the 512x256 two-stage RMSNorm specialization in
[`flydsl-stage1`](flydsl-stage1/README.md). CKL ordinary execution matches native FlyDSL GPU time;
CKL and PyTorch graph replay also match each other, but both graphs are slower than eager submission
for this two-launch case. This validates integration parity and identifies a negative control, not
a scheduling win.

Compile one multi-launch FlyDSL program once and reuse its copied HSACO, launch geometry, physical
arguments, PyTorch allocations, and current PyTorch stream in every mode:

1. FlyDSL native eager launches;
2. PyTorch `torch.cuda.CUDAGraph` capture and replay of the FlyDSL callable;
3. CKL ordinary execution from a retained resolved plan; and
4. CKL explicit HIP Graph replay.

The PyTorch graph is the most important practical baseline: it represents what a FlyDSL/PyTorch
user can already obtain without adopting CKL. The manual raw HIP Graph from Stage 0 remains a
runtime ceiling rather than a required baseline for every operator.

The in-process integration should register FlyDSL's copied HSACO bytes directly, import
`torch.Tensor.data_ptr()` storage without ownership, and import the current PyTorch HIP stream. The
file bundle remains useful for deployment and debugging but must not be used in steady-state timing.

FlyDSL should expose a compile-without-launch operation and an optional, producer-neutral executor
hook. CKL should implement that hook in its own package; FlyDSL should not acquire a hard dependency
on CKL.

## Real FlyDSL workload ladder

### Two-stage RMSNorm backward

Start with FlyDSL's two-stage RMSNorm backward. It launches a partial-gradient kernel followed by a
weight-gradient reduction kernel in one host launcher. This is a single-GPU, training-relevant
correctness target with a simple true dependency and an intermediate buffer. See the
[upstream implementation](https://github.com/ROCm/FlyDSL/blob/main/kernels/norm/rmsnorm_bwd_kernel.py).

Sweep small, medium, and large row counts. Small shapes test launch sensitivity; large shapes are a
negative control where device work should dominate and CKL should neither help nor materially hurt.

### MoE token sorting

MoE token sorting is the primary single-GPU scheduling benchmark. FlyDSL uses it to reorganize
router top-k results for expert GEMMs and provides three token-dependent paths:

- one kernel with all phases in LDS;
- two kernels through an HBM workspace; and
- four kernels for workspace clearing, scatter, counting, and final prefix/scatter work.

The implementation explicitly describes use by DeepSeek-style MoE models in the
[FlyDSL MoE sorting source](https://github.com/ROCm/FlyDSL/blob/main/kernels/moe/moe_sorting_kernel.py).
It exercises launch overhead, temporary storage, graph specialization, and eventually dependency
analysis while remaining runnable on one GPU.

Prefer a separate static plan for each supported token bucket initially. Do not add general dynamic
control flow merely to reproduce the producer's runtime selection. Report every tested token count,
expert count, top-k value, selected path, workspace size, and graph node count.

### Standard two-stage MoE

After sorting works, compose the complete single-GPU operator:

```text
optional sorting
    -> stage-1 expert GEMM
    -> optional split-K activation/quantization
    -> stage-2 expert GEMM
    -> optional top-k or masked reduction
```

AITER's proposed FlyDSL compile-plan inventory names these units and their conditional variants in
its [CPU-only compile-plan and manifest RFC](https://github.com/ROCm/aiter/issues/4333). This is a
strong integration target because the producer already wants explicit artifacts, ABI metadata,
deduplicated compile units, and strict AOT behavior; CKL adds execution planning rather than
reimplementing kernel compilation.

### Multi-GPU communication and MegaMoE

Defer custom all-reduce and MegaMoE until the single-GPU path is stable. They introduce symmetric
allocations, cross-device pointer tables, GPU-side synchronization, multiple streams, post-load
initialization, and topology requirements. They are important later for validating
communication/computation overlap.

The application relevance is established: AITER uses FlyDSL kernels across GEMM and MoE
([AITER repository](https://github.com/ROCm/aiter)), and AMD has published a
[Kimi-K2.5 FlyDSL MoE case study](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html).

## Measurement contract

Every result must record:

- CKL, FlyDSL, LLVM/MLIR, ROCm, PyTorch, AITER, and framework revisions;
- GPU model, architecture, driver/runtime version, clock policy, and visible-device topology;
- operator dimensions, dtypes, launch count, grid/block dimensions, and workspace bytes;
- warmup count, measured iterations, repetitions, and whether arguments were rotated;
- ordinary, graph, and reference checksums or numerical tolerances;
- compile, module-load, plan-resolution, graph-capture, and graph-instantiation setup times;
- median GPU time and p10/p90 or p5/p95 distribution;
- median CPU submission time and tail distribution; and
- break-even replay count when setup cost is included.

Use the same tensors and stream for paired modes. Synchronize outside measured regions except where
synchronization is part of the workload contract. Reset mutable outputs identically. Avoid timing a
Python loop when another mode performs the same repetitions inside C++; host-overhead comparisons
must cross equivalent language boundaries.

CKL should be evaluated over both favorable and unfavorable shapes. Results must retain negative
and neutral cases rather than publishing only the token buckets where graph replay wins.

## Path to Hugging Face model coverage

A general FlyDSL TorchDynamo/Inductor backend is not a prerequisite and would confound CKL results
with graph ingestion, partitioning, operator coverage, fusion, fallback, and Inductor's own graph
policies.

The preferred application route is:

```text
Hugging Face model
    -> vLLM, SGLang, or ATOM
    -> AITER operator selection
    -> FlyDSL artifacts and compile plan
    -> optional CKL execution provider
```

This permits controlled CKL-off/CKL-on comparisons for an existing operator implementation. Begin
with MoE models whose selected path contains FlyDSL sorting or two-stage MoE. Report operator
latency alongside request throughput, time to first token, time per output token, inter-token
latency, and tail latency so a small operator gain is not overstated as a model-wide result.

## Required implementation sequence

1. Add the raw HIP-versus-CKL overhead benchmark and retain raw samples.
2. Make pre-resolved ordinary execution the obvious benchmark/API path.
3. Add an in-process FlyDSL compile-without-launch and optional executor boundary.
4. Support the producer-described contiguous tensor/base-pointer ABI needed by RMSNorm; continue to
   reject views and ambiguous memref expansion.
5. Validate two-stage RMSNorm backward through all four execution modes.
6. Support statically specialized two- and four-kernel MoE sorting plans.
7. Compose multiple FlyDSL artifacts for standard two-stage MoE.
8. Add effect-derived dependencies only after source-ordered execution is correct and measured.
9. Integrate the operator path into AITER and a serving framework for model-level evaluation.
10. Add multi-GPU communication and MegaMoE only after the preceding baselines are reproducible.

## Non-goals for the first benchmark cycle

- Building a general PyTorch Inductor backend.
- Supporting arbitrary strided tensor views or guessing a producer ABI.
- Treating compilation, file loading, or graph instantiation as steady-state execution.
- Claiming value from single large GEMM or attention kernels where orchestration is not expected to
  help.
- Starting with an eight-GPU workload before single-GPU runtime parity is established.

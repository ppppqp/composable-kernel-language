# Composable Kernel Language

> **Status:** CKL is an experimental project undergoing an architectural pivot. The
> retired layout-aware prototype has been removed. Milestones 0 through 4 of the effect-derived
> orchestration design are implemented: feasibility, semantic inference, graph construction, the
> focused NVIDIA runtime, and measured plan optimization. Milestone 5 now includes a producer
> artifact boundary, FlyDSL launch-plan export, and an optional HIP executor.

Composable Kernel Language (CKL) is an experimental, MLIR-based orchestration layer for
repeated GPU programs. It derives memory effects and dependencies from kernel IR, constructs
an executable task graph, and will use target measurements to choose how that graph is
executed. CKL is intended to compose kernels that are already well implemented, rather than
predict or synthesize their intra-kernel schedules.

CKL's own kernel frontend is one possible producer. Other MLIR-based GPU DSLs should be able
to participate by implementing the required operation interfaces or registering external
interface models. See [the orchestration design](docs/orchestration.md) for an overview and the
[revised project specification](specs/mlir-layout-aware-gpu-compiler-spec.md) for the complete
contract, scope, validation criteria, and roadmap.

## Motivation

Kernel DSLs generally have enough information to describe loads, stores, allocations, views,
atomics, and synchronization, but that information is often lost at the runtime boundary.
The application then reconstructs dependencies manually with streams, events, and graph APIs.
Those declarations can be incomplete or become stale as kernels change.

CKL instead treats the kernel body as the source of truth:

1. **Infer effects:** derive reads, writes, allocation lifetimes, synchronization, and aliases
   from operations in the kernel IR.
2. **Build a sound graph:** add only dependencies required by SSA flow and inferred effects.
   When an effect cannot be proven precisely, use a conservative dependency.
3. **Keep kernels independent:** consume kernels from multiple dialects through interfaces
   instead of teaching CKL about every operation class.
4. **Measure orchestration:** evaluate complete execution plans on the target GPU rather than
   relying on a static GPU performance model.
5. **Explain decisions:** retain inferred effects, rejected transformations, measured results,
   and the final schedule as inspectable compiler artifacts.

## Target architecture

The intended compilation flow is:

```text
CKL kernels -----------+
MLIR GPU/Linalg -------+--> effect and dispatch interfaces --> orchestration IR
other GPU dialects ----+                                      |
                                                              +--> CUDA Graph
                                                              +--> CUDA streams
                                                              +--> future backends
```

The interoperability boundary has four parts:

- standard MLIR memory effects for reads, writes, allocations, and frees;
- an optional CKL access-region interface for sub-buffer precision;
- a dispatch interface describing a kernel invocation and its launch configuration; and
- alias/view information that traces an accessed value back to an underlying resource.

Whole-buffer effects are the conservative fallback when region information is unavailable.
Unknown operations are rejected in strict mode or treated as ordering barriers in conservative
mode. Manual effect annotations are reserved for foreign or opaque calls; CKL-authored kernels
derive their effects from their bodies.

Function arguments and SSA results form the kernel boundary. Manually named ports are not a
requirement of the target design. Multiple kernel implementations are likewise optional: they
become useful only when a DSL or author supplies genuinely different implementations with the
same observable contract.

## Initial use case

The first target workload is a repeated scientific simulation or iterative solver composed of
multiple small-to-medium kernels, such as Hotspot3D, FDTD, or a multi-field stencil pipeline.
These programs offer analyzable memory access, repeated launch sequences, temporary-buffer
lifetimes, reductions, and independent branches. They also provide a natural path to later
multi-GPU halo exchange without making distributed execution part of the initial scope.

CKL will be evaluated against ordinary sequential launches, single-stream CUDA Graph capture,
manually scheduled streams, and a hand-written explicit CUDA Graph. The manual graph is the
performance ceiling; CKL should approach it without hand-maintained dependencies and outperform
capture only where inferred independence, memory reuse, batching, or multiple in-flight instances
create a real opportunity. The complete validation plan is in
[docs/orchestration.md](docs/orchestration.md#validation-plan).

## Current implementation

The repository contains the reusable bootstrap and the first five milestones of the new design:

- a CKL MLIR dialect with allocation, free, load, store, atomic, synchronization, view, and
  interface-based dispatch operations;
- reusable access-region and dispatch operation interfaces;
- an external dispatch model for upstream `gpu.launch_func` (the boundary emitted by FlyDSL), with
  effects derived from nested `gpu.func` bodies and no per-launch effect declarations;
- whole-buffer base-resource/alias analysis and an interprocedural effect-summary pass with strict
  and conservative modes;
- a minimal normalized graph IR and a graph-construction pass that derives SSA, token, memory,
  lifetime, control, and unknown-effect barrier dependencies with provenance and DOT output;
- compiler-derived static resource descriptors and memory plans, including heap assignments,
  reuse explanations, and explicit diagnostics when a dynamic layout cannot yet be planned;
- a verified `ckl_exec` dialect and `ckl-lower-graph-to-exec` transformation that materialize
  heaps, resource bindings, producer-owned ABIs, physical argument slots, and reduced kernel
  dependencies;
- an optional NVIDIA Driver API runtime for ordinary launches, explicit CUDA Graphs, parameter
  updates, executable caching, suballocated device-memory views, topology batching, borrowed
  framework buffers/streams, and externally generated CUBIN loading;
- a backend-neutral runtime handoff consisting of immutable producer artifacts, byte-exact
  physical ABI arguments, and topologically ordered `ExecutionPlan` nodes; the NVIDIA executor
  resolves this handoff to CUDA modules and graph nodes;
- a device-independent memory planner that derives lifetime compatibility from graph reachability,
  excludes external and persistent resources from reuse, and records every storage decision;
- a `ckl-hostgen` tool that consumes verified `ckl_exec.plan` operations and emits owning C++
  unresolved `ExecutionPlan` builders for static direct-pointer kernel ABIs, including planned
  heaps, retained external buffers, packed arguments, artifact identities, and dependencies;
- a versioned `ckl-import-manifest` boundary through which an external DSL can provide resources,
  launch metadata, dependencies, artifact identities, and physical ABI slots without linking to
  CKL's compiler libraries;
- a `ckl-opt` driver that registers CKL alongside upstream MLIR dialects and GPU translations;
- Python utilities for invoking the optimizer, describing an NVIDIA target, extracting generated
  GPU objects, forming compilation cache keys, and emitting a FlyDSL HSACO/C++ bundle; and
- a CUDA Milestone 0 feasibility harness with two candidate workloads and four execution modes.

Milestone 0 selected the underfilled `multi_field` workload after its explicit graph ran 1.56x
faster than single-stream capture; the linear negative control showed no meaningful improvement.
Milestone 1 derives whole-buffer effects from kernel bodies and materializes inspectable summaries.
Milestone 2 binds those summaries to dispatch operands and constructs an inspectable orchestration
graph. Milestone 3 provides the first executable NVIDIA runtime and validates compiler-generated
CUBIN loading plus graph correctness, updates, caching, and overhead. Milestone 4 reuses ordered
temporaries without changing graph dependencies and measures graph batching and multiple
instances in flight. Static memory plans are attached automatically to `ckl.graph`, and the
direct-pointer ABI path now emits and executes C++ host-plan builders. Those builders no longer
resolve CUDA functions: a producer registers compiled artifacts and CKL's NVIDIA executor imports
them when it prepares the plan. The optional HIP executor consumes the same `ArtifactRegistry` and
`ExecutionPlan`; it accepts `rocm.hsaco` artifacts with the `rocm.bare_ptr` ABI, supports owned or
borrowed buffers and streams, and executes ordinary launches or HIP Graphs. General
producer-specific ABI lowering, including arbitrary lowered memref conventions, remains later
work.

Executable lowering is deliberately fail-closed: dispatches nested in host control flow and
direct-pointer arguments derived from views/subviews or non-identity layouts remain valid analysis
inputs but are rejected before host code or an unconditional executable plan can be emitted.

The optional cross-dialect test mixes an upstream GPU launch with a CKL launch over the same
resource and derives the required write/read edge from their kernel bodies. Static GPU launch
metadata is normalized through the dispatch interface; dynamic launch geometry is rejected
explicitly until the executable-plan IR can represent symbolic parameters. FlyDSL is the preferred
first external integration target because its `gpu.launch_func` boundary and explicit MLIR ABI are
closer to CKL. External DSL integration is evidence of reuse, not a prerequisite for CKL's own DSL
and thin runtime to be useful.

To inspect the graph for the deterministic test program:

```bash
build/tools/ckl-opt/ckl-opt tests/MLIR/graph.mlir --ckl-build-graph
```

To transform a statically planned direct-ABI program into executable IR:

```bash
build/tools/ckl-opt/ckl-opt tests/Runtime/host-plan.mlir \
  --ckl-build-graph --ckl-lower-graph-to-exec
```

To emit a C++ builder for a compatible executable plan:

```bash
build/tools/ckl-opt/ckl-opt input.mlir \
  --ckl-build-graph --ckl-lower-graph-to-exec -o plan.mlir
build/tools/ckl-hostgen/ckl-hostgen plan.mlir -o generated-plan.cpp
```

`ckl-hostgen` deliberately rejects `ckl.graph`; executable verification must happen first. CUDA
plans use `cuda.direct`/`direct_pointer`, while ROCm plans use
`rocm.bare_ptr`/`bare_pointer`.

An external compiler can construct the same verified executable IR through the v1 JSON manifest:

```bash
build/tools/ckl-import-manifest/ckl-import-manifest \
  tests/MLIR/external-plan.json -o external-plan.mlir
build/tools/ckl-hostgen/ckl-hostgen external-plan.mlir -o external-plan.cpp
```

The [executable manifest contract](docs/executable-manifest.md) is post-analysis: CKL validates its
explicit physical ABI and topology but does not guess missing effects, argument packing, or
dependencies.

The emitted `ckl.graph` contains normalized nodes, dependency reasons and source locations, plus a
DOT string suitable for visualization.

## Repository layout

```text
include/ckl/     public dialect, interface, and analysis headers
lib/             CKL dialect, analysis, core, and optional runtime implementation
tools/ckl-opt/   CKL-aware MLIR optimizer driver
python/ckl/      compilation and GPU-artifact utilities
tests/           Python utilities and MLIR semantic integration tests
benchmarks/      CUDA feasibility workloads and analysis scripts
docs/            architecture overview and roadmap
specs/           normative project specification
```

## Running the bootstrap tests

Requirements:

- CMake 3.20 or newer
- Python 3.12
- Ninja, when using the commands below

Configure, build, and run the validation suite:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

The default configuration does not build native code and does not require MLIR.

## Building with a local MLIR checkout

The optional generic optimizer requires an LLVM/MLIR build tree containing `MLIRConfig.cmake`.
Point `LLVM_BUILD` at the LLVM build directory and pass its MLIR package directory to CMake:

```bash
export LLVM_BUILD=/path/to/llvm-project/build

cmake -S . -B build -G Ninja \
   -DCMAKE_BUILD_TYPE=Debug \
   -DCKL_ENABLE_MLIR=ON \
   -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir"

cmake --build build
ctest --test-dir build --output-on-failure
```

The resulting optimizer is available at:

```text
build/tools/ckl-opt/ckl-opt
```

`MLIR_DIR` selects the MLIR package. To derive summaries in strict mode:

```bash
build/tools/ckl-opt/ckl-opt input.mlir --ckl-summarize-effects
```

For foreign operations without semantic adapters, conservative mode records whole-resource
read/write effects, an unknown-effect marker, and an ordering barrier:

```bash
build/tools/ckl-opt/ckl-opt input.mlir --allow-unregistered-dialect \
  '--ckl-summarize-effects=strict=false'
```

Summaries are emitted as compiler-owned `ckl.effect_summary` function attributes. They are always
recomputed from bodies; pre-existing attributes are not treated as authoritative.

## Python utilities

The retained Python API contains compilation and artifact utilities. Install it in a local
environment with:

```bash
uv venv --python 3.12
uv pip install -e .
uv run python -m unittest discover -s tests/Python -p "test_*.py"
```

```python
target = ckl.NVIDIATarget(
    chip="sm_120",
    features="+ptx87",
    toolkit_root=os.environ["CUDA_HOME"],
)
options = ckl.CompilerOptions(target=target)
```

`compile_module` remains available in `ckl.compiler` for raw MLIR input. It requires a built
`ckl-opt`; extracting embedded GPU objects additionally requires matching MLIR Python bindings.

FlyDSL's patched public boundary can be imported without intercepting its executor internals:

```python
compiled = flyc.compile(program, *arguments)
exported = compiled.artifact.export_for_orchestration()
artifact = ckl.import_flydsl_artifact(exported)
layouts = {"input": packed_input_shape_and_stride}
manifest = ckl.flydsl_executable_manifest(artifact, resource_layouts=layouts)
bundle = ckl.emit_flydsl_cpp_bundle(
    artifact, "build/flydsl-bundle", resource_layouts=layouts
)
```

FlyDSL copies its embedded `gpu.binary` payloads into plain Python objects while its own MLIR
runtime owns the module. CKL normalizes those bytes and assigns a deterministic artifact identity,
without loading a second MLIR Python extension in the same process. FlyDSL currently identifies
their physical device ABI as `rocm.bare_ptr`. For straight-line launches with static dimensions,
direct raw global pointers or contiguous memrefs, and basic scalar arguments, FlyDSL also exports a
verified plain-data launch plan. CKL expands memrefs into a pointer plus caller-supplied packed
shape/stride bytes and converts the result to executable manifest v1. `emit_flydsl_cpp_bundle` runs that
manifest through CKL verification and host generation, writes the HSACO beside it, and emits one
C++ include containing a typed `HipRuntime` plan builder and an `ArtifactRegistry` loader. The
generated loader takes the bundle directory, so artifact placement remains explicit and the
binary is not copied into C++ source. The returned bundle reports the stable generated
`plan_builder` and `artifact_loader` symbol names, which are derived from FlyDSL's host entry rather
than the content hash. More complex control flow, computed views, implicit memref ABI
descriptors, and genuinely distinct multi-object artifacts are rejected.

## Milestone 0 feasibility benchmarks

The optional CUDA benchmark compares sequential launches, single-stream capture, manually
coordinated streams, and an explicit minimal-dependency graph on two workloads. It is disabled by
default so the project remains configurable without a CUDA toolkit.

```bash
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCKL_BUILD_CUDA_BENCHMARKS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build build-cuda --target ckl-milestone0

python3 benchmarks/milestone0/run.py \
  --executable build-cuda/benchmarks/milestone0/ckl-milestone0 \
  --output benchmarks/milestone0/results/rtx5060ti.csv
python3 benchmarks/milestone0/analyze.py benchmarks/milestone0/results/rtx5060ti.csv
```

See [the benchmark documentation](benchmarks/milestone0/README.md) for workload definitions and
the evidence-based selection rule.

## Milestone 3 NVIDIA runtime

The runtime is optional and uses the CUDA Driver API. A combined MLIR/runtime build also compiles a
kernel through `ckl-opt`, extracts its `gpu.binary` CUBIN, and executes it as an end-to-end test.

```bash
cmake -S . -B build-runtime -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCKL_ENABLE_MLIR=ON \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DCKL_ENABLE_CUDA_RUNTIME=ON \
  -DCKL_RUNTIME_TEST_ARCH=120 \
  -DCKL_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build build-runtime
ctest --test-dir build-runtime --output-on-failure
```

The reusable API is declared in `include/ckl/Runtime/NvidiaRuntime.h`. The validation executable
compares ordinary launches, a cached CKL graph executable, and an independently constructed raw
CUDA Driver graph over the same six-kernel workload. Reproduction details and checked-in results
are in [benchmarks/milestone3](benchmarks/milestone3/README.md).

## Milestone 5 HIP runtime

The HIP runtime uses only the host API (`hip_runtime_api.h` and `libamdhip64`), avoiding HIP's
device-compilation CMake package in applications that only embed CKL. Build it with:

```bash
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCKL_ENABLE_HIP_RUNTIME=ON \
  -DCKL_HIP_RUNTIME_TEST_ARCH=gfx942
cmake --build build-hip
ctest --test-dir build-hip --output-on-failure
```

The API is declared in `include/ckl/Runtime/HipRuntime.h`. The validation compiles a small HSACO,
imports a ROCm manifest, compiles the generated host builder, registers the artifact, and checks
ordinary launches, HIP Graph execution, graph-cache reuse, and borrowed-buffer offset
preservation. It is reported as skipped when no HIP device is accessible.

## Acknowledgments

The target design is informed by:

- [MLIR](https://mlir.llvm.org/) for extensible IR, operation interfaces, and GPU lowering;
- [IREE](https://iree.dev/) for asynchronous resource and command scheduling;
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
  for reusable GPU command graphs;
- [Taskflow](https://taskflow.github.io/) for heterogeneous task-graph programming.

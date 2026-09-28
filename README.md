# Composable Kernel Language

> **Status:** CKL is an experimental project undergoing an architectural pivot. The
> retired layout-aware prototype has been removed. Milestones 0 and 1 of the effect-derived
> orchestration design are implemented; graph construction and runtime execution remain future
> work.

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

The repository contains the reusable bootstrap and the first three milestones of the new design:

- a CKL MLIR dialect with allocation, free, load, store, atomic, synchronization, view, and
  interface-based dispatch operations;
- reusable access-region and dispatch operation interfaces;
- whole-buffer base-resource/alias analysis and an interprocedural effect-summary pass with strict
  and conservative modes;
- a minimal normalized graph IR and a graph-construction pass that derives SSA, token, memory,
  lifetime, control, and unknown-effect barrier dependencies with provenance and DOT output;
- a `ckl-opt` driver that registers CKL alongside upstream MLIR dialects and GPU translations;
- Python utilities for invoking the optimizer, describing an NVIDIA target, extracting generated
  GPU objects, and forming compilation cache keys; and
- a CUDA Milestone 0 feasibility harness with two candidate workloads and four execution modes.

Milestone 0 selected the underfilled `multi_field` workload after its explicit graph ran 1.56x
faster than single-stream capture; the linear negative control showed no meaningful improvement.
Milestone 1 derives whole-buffer effects from kernel bodies and materializes inspectable summaries.
Milestone 2 binds those summaries to dispatch operands and constructs an inspectable orchestration
graph. No CUDA Graph runtime or kernel frontend is implemented yet; runtime execution begins in
Milestone 3.

To inspect the graph for the deterministic test program:

```bash
build/tools/ckl-opt/ckl-opt tests/MLIR/graph.mlir --ckl-build-graph
```

The emitted `ckl.graph` contains normalized nodes, dependency reasons and source locations, plus a
DOT string suitable for visualization.

## Repository layout

```text
include/ckl/     public dialect, interface, and analysis headers
lib/             CKL dialect and semantic analysis implementation
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

## Acknowledgments

The target design is informed by:

- [MLIR](https://mlir.llvm.org/) for extensible IR, operation interfaces, and GPU lowering;
- [IREE](https://iree.dev/) for asynchronous resource and command scheduling;
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
  for reusable GPU command graphs;
- [Taskflow](https://taskflow.github.io/) for heterogeneous task-graph programming.

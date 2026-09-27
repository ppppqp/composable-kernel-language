# Composable Kernel Language

> **Status:** CKL is an experimental project undergoing an architectural pivot. The
> retired layout-aware prototype has been removed. The repository is now a minimal bootstrap
> for the effect-derived orchestration design; its dialect, analyses, and runtime are not yet
> implemented.

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

The repository intentionally contains only the reusable bootstrap after removal of the previous
prototype:

- a generic `ckl-opt` driver that registers upstream MLIR dialects, passes, extensions, and GPU
  translations;
- Python utilities for invoking the optimizer, describing an NVIDIA target, extracting generated
  GPU objects, and forming compilation cache keys; and
- small tests for those Python utilities.

No CKL dialect, effect analysis, orchestration IR, CUDA Graph runtime, or kernel frontend is
currently implemented. The next code milestone is the semantic-interface and effect foundation
described in the specification.

## Repository layout

```text
tools/ckl-opt/   generic MLIR optimizer bootstrap
python/ckl/      compilation and GPU-artifact utilities
tests/Python/    bootstrap utility tests
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

`MLIR_DIR` selects the MLIR package. The current driver intentionally registers no CKL dialect or
pass; it is the executable scaffold for the first orchestration interfaces and analyses.

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

## Acknowledgments

The target design is informed by:

- [MLIR](https://mlir.llvm.org/) for extensible IR, operation interfaces, and GPU lowering;
- [IREE](https://iree.dev/) for asynchronous resource and command scheduling;
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
  for reusable GPU command graphs;
- [Taskflow](https://taskflow.github.io/) for heterogeneous task-graph programming.

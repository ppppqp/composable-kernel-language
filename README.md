# Composable Kernel Language

> **Status:** CKL is an experimental project undergoing an architectural pivot. The
> repository currently implements the layout-aware task-composition prototype described
> below under [Current implementation](#current-implementation). The effect-derived
> orchestration runtime is the target design, not yet an implemented feature.

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

The checked-in prototype predates this pivot. It contains a target-independent layout,
composition, proof, and conversion-planning core; an initial CKL MLIR dialect and optimizer;
and a Python frontend. The compiler can lower a boxed NVIDIA path to a device binary, but a
general runtime launch API and CUDA Graph backend are not yet implemented.

The existing task graph, callable implementations, provenance, resource lifetimes, Python
tracing, and NVIDIA binary generation are expected to inform the orchestration work. The manual
task-port descriptions, static execution costs, and layout-conversion search should not be
treated as the future public orchestration API.


## Repository layout

```text
include/ckl/Core/       public core APIs
lib/Core/               core implementations
include/ckl/Dialect/    MLIR dialect definitions and public APIs
lib/Dialect/            dialect implementations and transformations
include/ckl/Extensions/ target-specific extension APIs
lib/Extensions/         target-specific extension implementations
tools/ckl-opt/          CKL optimizer driver
docs/                   target architecture and project roadmap
specs/                  detailed project specification
tests/Core/             semantic and property-style validation
tests/Dialect/          MLIR round-trip, verification, and transformation tests
python/ckl/             dependency-free Python frontend
tests/Python/           frontend validation and generated-MLIR round trips
```

## Building the standalone core

Requirements:

- CMake 3.20 or newer
- A C++17 compiler
- Ninja, when using the commands below

Configure, build, and run the validation suite:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

MLIR is not required for the default build.

## Building with a local MLIR checkout

The optional compiler build requires an LLVM/MLIR build tree containing
`MLIRConfig.cmake`, the MLIR libraries, and TableGen tools. Point `LLVM_BUILD` at the LLVM
build directory and pass its MLIR package directory to CMake:

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

`MLIR_DIR` selects the MLIR package. Its `MLIRConfig.cmake` locates the matching LLVM
package and supplies the LLVM/MLIR include directories, libraries, and CMake build helpers.

## Python frontend

The Python API builds generic CKL types, index maps, distributions, tasks, tile memory operations,
and invocations using MLIR Python bindings. Compiler passes select task alternatives and introduce
target-specific operations. It can be used directly with `uv`:

```bash
uv venv --python 3.12
uv pip install -e .
export MLIR_PYTHON_ROOT="$LLVM_BUILD/tools/mlir/python_packages/mlir_core"
PYTHONPATH="$PWD/python:$PWD/build/python:$MLIR_PYTHON_ROOT" \
  uv run python tests/Python/emit_traced_memory.py
```

```python
@ckl.jit(tasks=[copy_task], passes=["--ckl-select-alternatives"])
def kernel(tile: tile_type) -> tile_type:
    return ckl.invoke(copy_task, [tile])

compiled = kernel.compile()
print(compiled.mlir)
```

Device kernels use an explicit GPU container:
```python
@ckl.jit(device=True, module_name="kernels", block_size=(32, 1, 1))
def kernel(source: source_type, target: target_type) -> None:
    ...
```

Compilation can target an NVIDIA device binary explicitly:

```python
target = ckl.NVIDIATarget(
    chip="sm_120",
    features="+ptx87",
    toolkit_root=os.environ["CUDA_HOME"],
)
compiled = kernel.compile(ckl.CompilerOptions(target=target))
```

## Acknowledgments

The current prototype and target design are informed by:

- [MLIR](https://mlir.llvm.org/) for extensible IR, operation interfaces, and GPU lowering;
- [IREE](https://iree.dev/) for asynchronous resource and command scheduling;
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
  for reusable GPU command graphs;
- [AMD Composable Kernel and CK Tile](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel)
  for composable GPU kernels;
- [NVIDIA CUTLASS/CuTe](https://github.com/nvidia/cutlass) and
  [ROCm FlyDSL](https://github.com/ROCm/FlyDSL) for layout representations;
- Colfax Research's [*Categorical Foundations for CuTe Layouts*](https://arxiv.org/pdf/2601.05972)
  for formal treatment of CuTe layouts; and
- [TileLang](https://github.com/tile-ai/tilelang) for layout inference.

# Composable Kernel Language

Composable Kernel Language (CKL) is an experimental compiler project for composing
layout-aware GPU tile tasks. Its goal is to let independently authored tile operations
negotiate physical data layouts, memory placement, and communication without hiding the
resulting costs or forcing intermediate global-memory materialization.

## Motivation

Existing systems such as AMD Composable Kernel, CuTe, and FlyDSL provide strong layout
algebras and tile-programming abstractions. CKL builds on that prior art and focuses on a
different boundary: compiler-mediated composition between separately specified tile tasks.

CKL emphasizes:

1. **Layout Abstraction**: Separate representations for execution ownership, per-executor values, logical tile
   coordinates, and storage addressing. This lays the groundwork for layout inference and optimization.
2. **Visibility**: Compiler-visible layout composition, equivalence proofs, conversion plans, and decision
   provenance. We consider this important for experts and KDA to work on kernel optimization.
3. **Composability**: Logical tasks that can use compiler-derived, target-provided, imported, or explicitly
   authored alternatives, with automatic, partially constrained, or pinned selection. Any arch-specific kernels can fit in the optimization pipeline seamlessly with just a declaration on contract and a tunable cost model.

## Architecture

The project is divided into two conceptual layers.

### Core

The core contains the target-independent layout, composition, planning, verification, and
provenance model. It is currently a semantic validation library rather than a production
compiler or performance model.

### DSL and compiler

The compiler layer currently contains an initial MLIR dialect and an optimizer driver. It
uses the standalone core for semantic verification and layout-conversion planning. The
Python frontend and boxed NVIDIA lowering are under development. Runtime integration is not yet
implemented.


## Repository layout

```text
include/ckl/Core/       public core APIs
lib/Core/               core implementations
include/ckl/Dialect/    MLIR dialect definitions and public APIs
lib/Dialect/            dialect implementations and transformations
include/ckl/Extensions/ target-specific extension APIs
lib/Extensions/         target-specific extension implementations
tools/ckl-opt/          CKL optimizer driver
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
   -DCKL_ENABLE_MLIR=ON \  # specify this to enable DSL build
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

The initial Python API builds generic CKL types, index maps, distributions, tasks, symbol bindings,
tile memory operations, and invocations. It emits MLIR text that is verified by `ckl-opt`; compiler
passes introduce target-specific operations. The package has no runtime dependencies and can be
used directly with `uv`:

```bash
uv venv --python 3.12
uv pip install -e .
export MLIR_PYTHON_ROOT="$LLVM_BUILD/tools/mlir/python_packages/mlir_core"
PYTHONPATH="$PWD/python:$PWD/build/python:$MLIR_PYTHON_ROOT" \
  uv run python tests/Python/emit_traced_memory.py
```

Python 3.12 is currently required because Python extension modules must match the interpreter used
to build the local MLIR bindings.

The MLIR-enabled CMake build generates CKL Python operation classes directly from `CKLOps.td`.
`@ckl.func` traces against those generated classes and upstream MLIR insertion points and SSA
values; it does not assemble operation text. Logical task contracts can be declared with
`@ckl.task(alternatives=[...])`, listed as dependencies of `@ckl.func(tasks=[...])`, and called with
`ckl.invoke(...)` during tracing.

`@ckl.jit` adds compilation and deterministic in-process caching:

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

This emits a native `gpu.module` containing a `gpu.func ... kernel`. Device kernels currently return
through memory arguments and therefore must have a `None` return annotation.

Compilation can target an NVIDIA device binary explicitly:

```python
target = ckl.NVIDIATarget(
    chip="sm_120",
    features="+ptx87",
    toolkit_root=os.environ["CUDA_HOME"],
)
compiled = kernel.compile(ckl.CompilerOptions(target=target))
```

The target architecture, PTX feature level, binary format, and resolved toolkit path participate in
the JIT cache identity. GPU objects are extracted through the MLIR Python binding and exposed as
bytes when the configured lowering passes produce `gpu.binary`; target selection alone does not
turn a host function into a GPU kernel. Calling a JIT function as an executable kernel remains
rejected until general runtime argument marshalling is implemented.

The current NVIDIA proof point traces the fixed one-warp `m16n8k16` MMA kernel entirely from Python,
compiles it to CUBIN, and validates its result on hardware. This is an end-to-end correctness gate,
not yet a general matrix-multiplication API. Its task declares an MMA implementation with cost 1
and a scalar fallback with cost 100; the invocation is unpinned, so the compiler's alternative
selection pass chooses the MMA implementation before target lowering.

With the CUDA environment active, compile the example with:

```bash
PYTHONPATH="$PWD/python:$PWD/build/python:$MLIR_PYTHON_ROOT" \
  uv run python examples/nvidia_mma.py
```

## Acknowledgments

The design is informed by the following awesome projects:
- [AMD Composable Kernel and CK Tile](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel) for the idea of composable kernel.
- [NVIDIA CUTLASS/CuTe](https://github.com/nvidia/cutlass) for the design of data layout model.
- [ROCm FlyDSL](https://github.com/ROCm/FlyDSL) for a mlir implementation of CUTLASS's data layout model.
- Colfax Research's [*Categorical Foundations for CuTe Layouts*](https://arxiv.org/pdf/2601.05972) for the indepth intro to category theory.
- [TileLang](https://github.com/tile-ai/tilelang) for the idea of layout inference.

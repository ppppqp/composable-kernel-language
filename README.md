# Composable Kernel Language

> NOTE: This project is still under active development.
> Read the [blog post](https://retepy.com/posts/til/2026-08-29-ckl/) for problem statement and design.

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
   provenance for kernel authors and compiler developers.
3. **Composability**: Logical tasks that can use compiler-derived, target-provided, imported, or explicitly
   authored alternatives, with automatic, partially constrained, or pinned selection. Target-specific
   implementations participate through declared layout contracts and costs.

## Architecture

The project is divided into two conceptual layers.

### Core

The core contains the target-independent layout, composition, planning, verification, and
provenance model. It is currently a semantic validation library rather than a production
compiler or performance model.

### DSL and compiler

The compiler layer currently contains an initial MLIR dialect and an optimizer driver. It
uses the standalone core for semantic verification and layout-conversion planning. The
Python frontend constructs native MLIR operations and can compile a boxed NVIDIA path to a device
binary. A general runtime launch API is not yet implemented.


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

The design is informed by the following awesome projects:
- [AMD Composable Kernel and CK Tile](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel) for the idea of composable kernel.
- [NVIDIA CUTLASS/CuTe](https://github.com/nvidia/cutlass) for the design of data layout model.
- [ROCm FlyDSL](https://github.com/ROCm/FlyDSL) for a mlir implementation of CUTLASS's data layout model.
- Colfax Research's [*Categorical Foundations for CuTe Layouts*](https://arxiv.org/pdf/2601.05972) for the indepth intro to category theory.
- [TileLang](https://github.com/tile-ai/tilelang) for the idea of layout inference.

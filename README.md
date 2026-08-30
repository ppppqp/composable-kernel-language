# Composable Kernel Language

Composable Kernel Language (CKL) is an experimental compiler project for composing
layout-aware GPU tile tasks. Its goal is to let independently authored tile operations
negotiate physical data layouts, memory placement, and communication without hiding the
resulting costs or forcing intermediate global-memory materialization.

## Motivation

Existing systems such as AMD Composable Kernel, CuTe, and FlyDSL provide strong layout
algebras and tile-programming abstractions. CKL builds on that prior art and focuses on a
different boundary: compiler-mediated composition between separately specified tile tasks.

The intended design emphasizes:

1. Separate representations for execution ownership, per-executor values, logical tile
   coordinates, and storage addressing.
2. Compiler-visible layout composition, equivalence proofs, conversion plans, and decision
   provenance.
3. A reusable core library whose semantics can be tested independently of frontend syntax
   and MLIR dialect design.
4. Explicit task alternatives that allow automatic, partially constrained, or pinned
   implementation choices.

## Architecture

The project is divided into two conceptual layers.

### Core

The core contains the target-independent layout, composition, planning, verification, and
provenance model. It is currently a semantic validation library rather than a production
compiler or performance model.

### DSL and compiler

The compiler layer currently contains an initial MLIR dialect and an optimizer driver. It
uses the standalone core for semantic verification and layout-conversion planning. The
frontend DSL, target lowering, and runtime integration are still under development.

The DSL and dialect follow a few design principles:

1. The DSL is a thin, inspectable frontend over MLIR rather than a second layout algebra.
2. Logical tile shape is separate from physical ownership, local storage, and memory
   placement.
3. Layout negotiation and communication costs remain explicit in compiler IR.
4. Generic composition semantics stay target-independent; hardware-specific layouts and
   lowering live in extensions.
5. Dialect verification and planning reuse the same semantics validated by the core.

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
build-mlir/tools/ckl-opt/ckl-opt
```

`MLIR_DIR` selects the MLIR package. Its `MLIRConfig.cmake` locates the matching LLVM
package and supplies the LLVM/MLIR include directories, libraries, and CMake build helpers.

## Acknowledgments

The design is informed by the following awesome projects:
- [AMD Composable Kernel and CK Tile](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel) for the idea of composable kernel.
- [NVIDIA CUTLASS/CuTe](https://github.com/nvidia/cutlass) for the design of data layout model.
- [ROCm FlyDSL](https://github.com/ROCm/FlyDSL) for a mlir implementation of CUTLASS's data layout model.
- Colfax Research's [*Categorical Foundations for CuTe Layouts*](https://arxiv.org/pdf/2601.05972) for the indepth intro to category theory.
- [TileLang](https://github.com/tile-ai/tilelang) for the idea of layout inference.

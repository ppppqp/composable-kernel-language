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

Not implemented yet. This layer is expected to contain the frontend, MLIR dialects,
inference and transformation passes, target lowering, and runtime integration.

The eventual DSL and dialects should encode semantics established by the core rather than
define a competing layout model.

## Repository layout

```text
include/ckl/Core/       public core APIs
lib/Core/               core implementations
tests/Core/             semantic and property-style validation
tests/Core/fixtures/    checked-in reference coordinate tables
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

## Acknowledgments

The design is informed by AMD Composable Kernel and CK Tile, NVIDIA CUTLASS/CuTe, ROCm
FlyDSL, and Colfax Research's *Categorical Foundations for CuTe Layouts*.

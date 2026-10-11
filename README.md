# Composable Kernel Language

CKL is an experimental, self-contained GPU DSL built on MLIR. The target is one language for
writing kernels and composing them into repeated GPU programs, with a CKL compiler and thin
CUDA/HIP runtimes. Kernel bodies provide the memory effects used to infer dependencies, plan
storage, and build reusable command graphs.

The project has pivoted away from integrating existing GPU DSLs. External compiler adapters,
launch-plan manifests, and the mixed-dialect dispatch path have been removed. Upstream MLIR
arithmetic, control-flow, memory, and GPU lowering remain compiler infrastructure.

## Current status

The repository implements the compiler/runtime foundation:

- CKL memory, allocation, view, atomic, synchronization, and dispatch operations;
- interprocedural effect and alias analysis with strict and conservative modes;
- dependency graphs with provenance and DOT output;
- static memory planning with safe temporary-buffer reuse;
- verified `ckl_exec` plans and generated C++ host builders;
- artifact loading, ordinary launches, command graphs, caching, and borrowed buffers/streams in
  optional CUDA and HIP runtimes; and
- feasibility and runtime benchmarks with correctness checks and recorded measurements.

The user-facing kernel/program frontend and complete CKL kernel-to-device lowering are still to
be implemented. Python currently exposes raw-MLIR compilation and GPU-object utilities; it does
not yet provide kernel decorators or a tracing frontend. Runtime fixtures use hand-written
CUDA/HIP kernels, and one NVIDIA test validates upstream GPU-to-CUBIN lowering. These validate
infrastructure, not a completed CKL source-to-device pipeline.

See [the language design](docs/dsl.md), [compiler architecture](docs/orchestration.md), and
[normative specification](specs/mlir-layout-aware-gpu-compiler-spec.md).

## Intended compilation flow

```text
CKL source: kernels + repeated programs
    -> CKL kernel and program IR
    -> inferred effects, aliases, and dependency graph
    -> memory planning and verified executable plan
    -> CKL device lowering + compiled CUBIN/HSACO artifacts
    -> generated host builder + CUDA/HIP executor
```

Effect analysis runs while kernel bodies remain available. CKL owns argument packing, launch
metadata, artifact identity, and device code generation. Complete execution plans are measured on
the target GPU to choose graph boundaries, iteration batches, and instances in flight.

The initial workload is a repeated scientific solver or multi-field stencil with small kernels,
independent branches, temporary buffers, and reductions. Sequential launches, single-stream
capture, manually scheduled streams, and a hand-written explicit graph remain the baselines.

## Build and test

The default build requires CMake 3.20+, a C++17 compiler, and Python. The Python package requires
Python 3.12. Native core tests do not require MLIR or a GPU toolkit.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

For the MLIR compiler tools, point `MLIR_DIR` at a compatible LLVM/MLIR build:

```bash
cmake -S . -B build-mlir -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DCKL_ENABLE_MLIR=ON \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build-mlir
ctest --test-dir build-mlir --output-on-failure
```

Inspect an inferred graph or generate a host builder from CKL IR:

```bash
build-mlir/tools/ckl-opt/ckl-opt tests/MLIR/graph.mlir --ckl-build-graph
build-mlir/tools/ckl-opt/ckl-opt tests/Runtime/host-plan.mlir \
  --ckl-build-graph --ckl-lower-graph-to-exec -o /tmp/host-plan.exec.mlir
build-mlir/tools/ckl-hostgen/ckl-hostgen /tmp/host-plan.exec.mlir -o /tmp/host-plan.cpp
```

Host generation requires verified `ckl_exec.plan` IR. The current executable subset requires
static launch dimensions and base identity-layout resources. Conditional dispatches, dynamic
resources, view-derived pointer arguments, and unsupported ABI packing fail with diagnostics.
Compiler-owned effect summaries are always recomputed from kernel bodies.

Install and test the Python utilities:

```bash
uv venv --python 3.12
uv pip install -e .
uv run python -m unittest discover -s tests/Python -p "test_*.py"
```

`ckl.compiler.compile_module` accepts raw MLIR through a built `ckl-opt`. `NVIDIATarget` configures
upstream GPU-to-NVVM compilation; embedded object extraction requires compatible MLIR Python
bindings.

## Optional runtimes and benchmarks

Enable CUDA validation with `-DCKL_ENABLE_CUDA_RUNTIME=ON`. Combining it with MLIR enables generated
CUBIN and CKL host-plan tests. `CKL_RUNTIME_TEST_ARCH` selects the SM architecture and
`CKL_CUDA_HOST_COMPILER` selects the nvcc host compiler. See the
[Milestone 3](benchmarks/milestone3/README.md) and
[Milestone 4](benchmarks/milestone4/README.md) benchmarks.

Enable HIP validation with `-DCKL_ENABLE_HIP_RUNTIME=ON`. The host runtime requires
`hip_runtime_api.h` and `libamdhip64`; validation additionally needs `hipcc` and the offload bundler.
`CKL_HIP_RUNTIME_TEST_ARCH` selects the GFX architecture. With MLIR enabled, the test derives a HIP
host plan from CKL kernel effects. Device execution is skipped when no HIP device is accessible.

`CKL_BUILD_CUDA_BENCHMARKS` enables the [feasibility harness](benchmarks/milestone0/README.md).
`CKL_BUILD_HIP_BENCHMARKS` enables the [HIP overhead benchmark](benchmarks/hip-runtime-overhead/README.md).

## Repository layout

```text
include/ckl/      dialect, analysis, core, and runtime headers
lib/             compiler analyses, transforms, and runtime implementations
tools/           ckl-opt and ckl-hostgen
python/ckl/      raw-MLIR compilation and GPU-artifact utilities
tests/          core, Python, MLIR semantics, and optional GPU validation
benchmarks/      feasibility, planning, and runtime measurements
docs/            language and compiler architecture
specs/           normative scope and acceptance criteria
```

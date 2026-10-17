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

- verified indexed kernel loads/stores and thread/block/global coordinates with typed buffers;
- CKL memory, allocation, view, atomic, synchronization, and dispatch operations;
- interprocedural effect and alias analysis with strict and conservative modes;
- dependency graphs with provenance and DOT output;
- static memory planning with safe temporary-buffer reuse;
- verified `ckl_exec` plans and an in-process LLVM host JIT;
- artifact loading, ordinary launches, command graphs, caching, and borrowed buffers/streams in
  optional CUDA and HIP runtimes; and
- feasibility and runtime benchmarks with correctness checks and recorded measurements.

The initial Python frontend provides `@ckl.kernel`, `@ckl.program`, typed shape specialization,
indexed computation, guarded stores, temporaries, and explicit launches. It compiles source ASTs
to MLIR without executing bodies or requiring a GPU. The initial CUDA backend compiles these
bodies to embedded CUBINs and JIT-compiles host entry points in process. A two-kernel CKL program
passes sequential and CUDA graph numerical checks through Python. Broader language features remain
subsequent work; the older hand-written CUDA/HIP fixtures continue to validate infrastructure.

See [the language design](docs/dsl.md), [compiler architecture](docs/orchestration.md), and
[normative specification](specs/mlir-layout-aware-gpu-compiler-spec.md).

## Intended compilation flow

```text
CKL source: kernels + repeated programs
    -> CKL kernel and program IR
    -> inferred effects, aliases, and dependency graph
    -> memory planning and verified executable plan
    -> CKL device lowering + compiled CUBIN/HSACO artifacts
    -> LLVM host JIT + CUDA executor (HIP runtime retained separately)
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

Inspect an inferred graph or executable plan from CKL IR:

```bash
build-mlir/tools/ckl-opt/ckl-opt tests/MLIR/graph.mlir --ckl-build-graph
build-mlir/tools/ckl-opt/ckl-opt tests/Runtime/host-plan.mlir \
  --ckl-build-graph --ckl-lower-graph-to-exec -o /tmp/host-plan.exec.mlir
```

The host JIT consumes verified `ckl_exec.plan` IR. The current executable subset requires
static launch dimensions and base identity-layout resources. Conditional dispatches, dynamic
resources, view-derived pointer arguments, and unsupported ABI packing fail with diagnostics.
Compiler-owned effect summaries are always recomputed from kernel bodies.

Install and test the Python utilities:

```bash
uv venv --python 3.12
uv pip install -e .
uv run python -m unittest discover -s tests/Python -p "test_*.py"
```

Lower the [two-kernel Python example](examples/vector_pipeline.py) without GPU dependencies:

```bash
PYTHONPATH=python python3 examples/vector_pipeline.py > /tmp/vector-pipeline.mlir
build-mlir/tools/ckl-opt/ckl-opt /tmp/vector-pipeline.mlir --ckl-build-graph
```

`step.lower(signature=...)` returns a `LoweredProgram`. Use `.verify(ckl_opt=...)` or
`.analyze("--ckl-build-graph", ckl_opt=...)` to run the native verifier/analyses. The frontend
emits a logical device with no ABI capability, so executable lowering fails until backend
compilation supplies real device artifacts and packing metadata.

Use `lowered.compile(target=ckl.NVIDIATarget(...))` to compile device code and its host launcher
in process. The result supports `.run(...)` and `.bind(...).run(graph=True)`, with no generated
files or MLIR Python bindings. See [CUDA compilation and execution](docs/cuda-backend.md).

`ckl.compiler.compile_module` accepts raw MLIR through a built `ckl-opt`. `NVIDIATarget` configures
upstream GPU-to-NVVM compilation; embedded object extraction requires compatible MLIR Python
bindings.

## Optional runtimes and benchmarks

Enable CUDA validation with `-DCKL_ENABLE_CUDA_RUNTIME=ON`. Combining it with MLIR enables
the in-process `ckl-cuda-frontend-runtime` test, compiled directly from the Python example.
`CKL_RUNTIME_TEST_ARCH` selects the SM architecture and
`CKL_CUDA_HOST_COMPILER` selects the nvcc host compiler. See the
[Milestone 3](benchmarks/milestone3/README.md) and
[Milestone 4](benchmarks/milestone4/README.md) benchmarks.

Enable HIP validation with `-DCKL_ENABLE_HIP_RUNTIME=ON`. The host runtime requires
`hip_runtime_api.h` and `libamdhip64`; validation additionally needs `hipcc` and the offload bundler.
`CKL_HIP_RUNTIME_TEST_ARCH` selects the GFX architecture. MLIR analysis tests separately verify HIP
plans and their inferred dependencies. Device execution is skipped when no HIP device is accessible.

`CKL_BUILD_CUDA_BENCHMARKS` enables the [feasibility harness](benchmarks/milestone0/README.md).
`CKL_BUILD_HIP_BENCHMARKS` enables the [HIP overhead benchmark](benchmarks/hip-runtime-overhead/README.md).

The [standalone workload suite](benchmarks/standalone/README.md) benchmarks five CKL-authored
examples through the Python/JIT path. It retains correctness checks, raw samples, setup costs,
temporary-memory measurements, and recommendations for the next compiler optimization.
The initial automatic pointwise-chain fusion pass is enabled by default, with an unfused fallback
for overlapping buffers. See the [fusion measurements](benchmarks/standalone/results/rtx5060ti-fusion.md).

## Repository layout

```text
include/ckl/      dialect, analysis, core, and runtime headers
lib/             compiler analyses, transforms, and runtime implementations
tools/           ckl-opt for optional verification and inspection
python/ckl/      typed Python frontend, MLIR compilation, and GPU-artifact utilities
tests/          core, Python, MLIR semantics, and optional GPU validation
benchmarks/      feasibility, planning, and runtime measurements
docs/            language and compiler architecture
specs/           normative scope and acceptance criteria
```

# Composable Kernel Language

**Write GPU kernels. Let the compiler derive their execution graph.**

CKL is an experimental GPU DSL built on MLIR. Define kernels and compose them in Python;
CKL uses memory effects and alias analysis to infer dependencies, reuse temporary storage,
and build replayable CUDA graphs.

On a complete autoregressive decoder, inferred graphs run **1.04–1.22× faster than
single-stream graph capture**, with unchanged unfused FP32 kernels. Temporary storage reuse
saves **up to 94% of scratch memory** in the measured cases.

[Language design](docs/dsl.md) · [Architecture](docs/orchestration.md) ·
[Python frontend](docs/python-frontend.md) · [CUDA setup](docs/cuda-backend.md) ·
[Benchmarks](benchmarks/decoder/README.md)

## Measured on a complete decoder

Each sample generates 16 successive tokens through embedding, RMSNorm, projections, RoPE,
KV-cache updates, attention, an MLP, and greedy selection. Tokens feed back into the next step;
the context grows throughout the sequence.

| Configuration | Ordinary launches | Stream capture | CKL inferred graph | Speedup over capture | Scratch with reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small: 2 layers, hidden 64 | 62.08 µs | 54.82 µs | 46.71 µs | **1.17×** | 14,080 → 3,072 B |
| Wide: 2 layers, hidden 256 | 154.06 µs | 149.06 µs | 122.47 µs | **1.22×** | 47,360 → 10,240 B |
| Deep: 8 layers, hidden 64 | 230.72 µs | 194.67 µs | 162.48 µs | **1.20×** | 54,016 → 3,072 B |

Times are median native GPU-stream time per complete decode step on an RTX 5060 Ti,
Release build, seven samples. Compilation, transfers, prefill, and graph setup are excluded.
The scratch column is a separate storage-reuse ablation; reuse has little effect on latency.

![GPU timelines showing serial capture versus compiler-derived overlap of independent projection kernels](benchmarks/decoder/results/deep-dive/figures/small-timeline.png)

*Inferred dependencies expose Q/K/V and gate/up projection overlap. The same 48 kernels still
execute per step. Grouping repeats dependent steps in one submission, reducing host overhead.*

These are small models with random weights and naive kernels, evaluated to study orchestration.
Long-context softmax and large-vocabulary selection limit the gains. Independent-request
concurrency is supplied by the benchmark and is separate from CKL's dependency inference.

See the [nine-case evaluation and GPU traces](benchmarks/decoder/results/deep-dive/REPORT.md)
for controls, setup costs, correctness checks, raw samples, and limitations.

## Kernels describe computation; programs describe composition

```python
import ckl

N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]

@ckl.kernel
def update(src: Vector, dst: Vector, bias: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < N:
        dst[i] = src[i] + bias

@ckl.program
def step(src: Vector, dst: Vector, bias: ckl.f32) -> None:
    temporary = ckl.empty_like(src)
    grid = (ckl.cdiv(N, 256),)
    ckl.launch(update, src, temporary, bias, grid=grid, block=(256,))
    ckl.launch(update, temporary, dst, bias, grid=grid, block=(256,))
```

CKL derives the producer–consumer edge from the write/read conflict on `temporary`.
Static contiguous subviews expose byte ranges, allowing disjoint accesses to one allocation
to avoid false dependencies. Temporary storage is reused only when graph order proves it safe.

The [runnable example](examples/vector_pipeline.py) shows shape specialization and IR inspection.
The [CUDA guide](docs/cuda-backend.md) shows in-process compilation, buffer binding, and
ordinary, captured, or inferred graph execution. Use `fuse_pointwise=False` to preserve the
unfused computation when comparing orchestration strategies.

## Build and explore

The Python frontend requires Python 3.12 and matching upstream MLIR Python bindings.
The native compiler additionally requires a compatible LLVM/MLIR build and nanobind.
CUDA execution requires a compatible toolkit and GPU; see [setup and ownership](docs/cuda-backend.md#build-and-ownership).

The core planner and execution-plan tests can be built without MLIR or a GPU toolkit
(CMake 3.20+, a C++17 compiler, and Python):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

To build the MLIR compiler, add `-DCKL_ENABLE_MLIR=ON`,
`-DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir`, and
`-DPython_EXECUTABLE=/path/to/python-with-nanobind` to configuration.
Add `-DCKL_ENABLE_CUDA_RUNTIME=ON` for CUDA execution.

CKL currently supports static shapes and launches, structured kernel control flow, effect/alias
analysis, subview-aware dependency graphs, storage planning, and an in-process device/host JIT.
The source compiler targets CUDA; the optional HIP runtime validates infrastructure separately.
Dynamic host control flow and dynamic/strided views remain future work. Public execution is synchronous.

[Compiler sources](lib/) · [Tests](tests/) · [Examples](examples/) ·
[Design documents](docs/) · [Specification](specs/mlir-layout-aware-gpu-compiler-spec.md)

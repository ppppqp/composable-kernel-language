# Python-driven compilation and ckl_exec → LLVM validation

The compiler pipeline and host JIT now live in Python. `Compiler.cpp` is removed;
`NativeBindings.cpp` supplies MLIR registration/inspection and `RuntimeBindings.cpp`
retains native CUDA execution. The new `ckl-lower-exec-to-llvm` pass generates the
LLVM dialect module consumed by upstream MLIR's Python `ExecutionEngine`.

Steady-state decode medians are essentially unchanged in this run. Repeated compilation
costs 5.7–7.3% more for the measured small/deep shapes; first binding is about 4% faster.
These are descriptive before/after measurements, not statistical equivalence guarantees.

## Validation

- Debug and Release builds both pass all 10 CTest suites. See [Debug log](debug-tests.log)
  and [Release log](release-tests.log).
- All 126 decode case/variant pairs pass the independent NumPy reference before and after.
  Tokens/history/position, numerical logits/KV, untouched buffers and allocation guards are checked.
- Prompts, generated tokens, dependency topologies, dispatch counts, temporary counts and heap sizes
  match exactly across the complete matrix.
- The repeated setup benchmark additionally checks exact prepared-kernel IR hashes and execution
  plans across both implementations for small/deep cases with reuse on/off.
- New lowering tests cover byte-exact integer/float/raw-byte constants, embedded NUL bytes,
  argument slot order, resource bindings, dependency calls and invalid handoffs.
  Python integration tests cover the LLVM dialect inspection API and source/signature mismatch.
  Existing tests cover closed-parent ownership, alias fallback, CUDA graphs, fusion and loops.

## Steady-state decode

NVIDIA RTX 5060 Ti, Release builds, CUDA 12.8, `sm_120` / `+ptx87`, FP32 with fusion disabled.
Nine shapes × fourteen variants; each sample generates 16 successive tokens. Seven samples
and three warmups per variant, with rotating/reversed variant order. Native timings average
one complete step; Python timings measure completion of the full sequence. Compilation,
binding, graph setup, prefill, transfers and resets are excluded from steady-state timing.
Grouping changes host observation granularity and is not streamed token latency.

Positive changes mean slower execution:

| Metric | Minimum change | Median change | Maximum change |
|---|---:|---:|---:|
| stream_us | -0.85% | +0.13% | +0.92% |
| wall_us | -0.85% | +0.13% | +0.95% |
| generation_ms | -1.20% | +0.09% | +0.57% |

[Full per-variant comparison](comparison.md) · [Before samples](before.json) · [After samples](after.json)

## Repeated compilation and binding

[Setup benchmark](../../host_setup.py) lowers the frontend once outside timing, then measures
independent compile calls, first binding (including device-module loading) and repeated binding
with a loaded module. Each shape/reuse case uses two warmup compilations followed by seven
measured compilations. Each measured program contributes five rebind samples (35 per row).
No decode kernels run in this setup benchmark. Baseline and final processes ran serially after
build completion; GPU tests and decode measurements also ran serially.

| Case | Reuse | Compile before → after (ms) | Compile change | First bind change | Rebind change |
|---|---|---:|---:|---:|---:|
| small | False | 236.10 → 250.12 | +5.94% | -3.51% | -5.56% |
| small | True | 235.16 → 248.59 | +5.71% | -4.53% | -5.44% |
| deep | False | 712.55 → 760.57 | +6.74% | -4.20% | +1.61% |
| deep | True | 707.40 → 758.81 | +7.27% | -4.38% | +1.00% |

[Before setup samples](setup-before.json) · [After setup samples](setup-after.json)

The host entry is invoked during binding, not each decode step. JIT optimization level 0 avoids
optimizing a plan-construction function outside the execution hot path. The added MLIR lowering
and Python orchestration still add measurable compilation cost; compilation should be amortized
across bound executions. Native execution and graph replay remain unchanged.

## Reproduce

The baseline is repository revision `47428a67976e1da7d7b52c0902b7389a2db2420b` with its original
native compiler. A pre-existing README edit was retained and does not affect compilation.
Its library and Python package were saved before refactoring; see [validation metadata](validation.json).
Run each phase with its matching Python package and native library, never both in one interpreter.
Matching upstream MLIR Python bindings, NumPy and nanobind are required in both environments.

```bash
cmake -S . -B /tmp/ckl-exec-llvm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCKL_ENABLE_MLIR=ON -DCKL_ENABLE_CUDA_RUNTIME=ON \
  -DCKL_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir \
  -DPython_EXECUTABLE=/path/to/python -DCUDAToolkit_ROOT=/path/to/cuda
cmake --build /tmp/ckl-exec-llvm -j 8
```

Set `PYTHONPATH` to the phase's `python` directory and the matching upstream MLIR Python package;
set `CKL_NATIVE_LIBRARY` to that phase's built extension. Use a writable, consistent CUDA cache.
Run the existing matrix runner from the matching checkout and the setup helper from this change:

```bash
python benchmarks/decoder/run.py --toolkit /path/to/cuda --suite extended \
  --steps 16 --samples 7 --warmup 3 --output /path/to/phase.json
python benchmarks/decoder/host_setup.py --toolkit /path/to/cuda \
  --samples 7 --warmup 2 --binds 5 --output /path/to/setup-phase.json
python benchmarks/decoder/compare.py /path/to/before.json /path/to/after.json \
  --output /path/to/comparison.md
ctest --test-dir /tmp/ckl-exec-llvm --output-on-failure
```

GPU clocks are unlocked. Library order was before then after, rather than randomized across
processes. Small differences may reflect run-to-run variation; this experiment is limited to
this GPU, the decoder workloads and the retained regression suite. The extension now shares
`libMLIRPythonCAPI` with Python and requires matching bindings; its smaller file size is not a
measurement of total process memory.

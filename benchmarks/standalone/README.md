# Standalone CKL workload benchmarks

These examples compile and execute entirely through CKL's Python frontend, in-process compiler,
LLVM host JIT, and CUDA executor. They compare ordinary launches and CUDA graphs using the same
compiled kernels, arguments, dependencies, and planned temporary storage.

| Case | Source use case | Kernel nodes | What it tests |
|---|---|---:|---|
| `single_affine` | Scale/bias vector preprocessing | 1 | Launch-overhead control |
| `chain_six` | Six successive pointwise transforms | 6 | Sequential launch and intermediate-memory costs |
| `fused_chain_six` | The same six transforms in one kernel | 1 | A manually authored fusion control |
| `fork_join` | Two fields, join, two branches, final join | 6 | Inferred independence, joins, and temporary reuse |
| `stencil_two` | Two steps of a three-point smoothing stencil | 2 | Neighbor accesses, explicit boundaries, and intermediate storage |

All kernels and programs are in [workloads.py](workloads.py). There are no hand-written CUDA
kernels and no frontend extensions for these cases. `fused_chain_six` is a separate CKL source
implementation; CKL does **not** perform automatic kernel fusion yet.

## Reproduce

Build a Release compiler/runtime against an LLVM/MLIR build with native JIT and NVPTX support:

```bash
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCKL_ENABLE_MLIR=ON -DCKL_ENABLE_CUDA_RUNTIME=ON \
  -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir \
  -DCUDAToolkit_ROOT=/path/to/cuda
cmake --build build-bench --target CKLCompiler

CKL_NATIVE_LIBRARY=$PWD/build-bench/python/ckl/libCKLCompiler.so \
PYTHONPATH=python python3.12 benchmarks/standalone/run.py \
  --toolkit /path/to/cuda --chip sm_120 \
  --sizes 257 4096 65536 1048576 \
  --iterations 200 --samples 7 --warmup 10 --latency-iterations 20 \
  --output benchmarks/standalone/results/local.json
python3.12 benchmarks/standalone/analyze.py \
  benchmarks/standalone/results/local.json \
  --output benchmarks/standalone/results/local.md
```

Choose the chip for your GPU; the current runner uses PTX 8.7 features and a compatible toolkit.
Only Python's standard library is needed in addition to CKL. The runner retains partial JSON
if a case fails; the analyzer rejects incomplete matrices.

## Measurement and correctness

Each size is a separate specialization. AST lowering, native compilation/JIT, first binding,
rebinding, and graph instantiation are measured separately from steady execution. Device/runtime
initialization is recorded once. The first compile includes process-local compiler initialization;
there is no persistent disk cache. Binding includes temporary allocation and, on first bind,
module loading and function resolution.

The steady-state measurements cover two distinct calling patterns:

- **Native burst:** submit 200 complete programs from C++, synchronize once, and divide by 200.
  CUDA events measure stream elapsed time; a CPU clock measures submission plus completion wall
  time. Stream elapsed time includes native submission gaps, so it is not isolated kernel time.
- **Synchronous Python latency:** call the existing `Invocation.run()` 20 times and divide the
  wall time by 20. Each call synchronizes. This includes Python/FFI and synchronization costs.

Seven paired samples alternate ordinary/graph order after ten warmup calls per mode. CPU
reference scans stay outside timing samples. The saved JSON includes every sample and range,
maximum error, graph topology, allocated heap sizes, and build/environment metadata. The report
also estimates graph setup amortization against native burst savings.

Every case is checked against a CPU reference with f32 rounding, before and after measurements.
The checks cover every logical output, an extra 256-element poisoned output tail, and unchanged
inputs. The length 257 exercises an incomplete block. All outputs in the recorded run matched
exactly, though the checker allows a small floating-point tolerance. Inputs stay fixed and source
and output buffers are distinct, so repeated submissions have the same expected result.

These are warm repeated-buffer measurements on one GPU with unlocked clocks. They do not measure
cold-memory bandwidth, transfer-inclusive application time, stream capture, manual concurrent
streams, or other DSL/framework implementations. The largest working sets may benefit from GPU
caches. Plan reductions report allocated bytes, not a measured no-reuse performance baseline.
Independent graph branches alone do not establish that useful kernel overlap occurred.

## Recorded findings and next step

The [full report](results/rtx5060ti.md) and [raw samples](results/rtx5060ti.json) cover an RTX 5060 Ti,
CUDA 12.8, SM 120, Python 3.12, and Release native code. All 20 cases passed.

For 257 elements, median native stream times were:

| Case | Ordinary µs | Graph µs |
|---|---:|---:|
| Single affine | 1.32 | 2.05 |
| Six-stage chain | 7.87 | 6.15 |
| Manually fused chain | 1.31 | 2.04 |
| Fork/join | 7.87 | 6.18 |
| Two-step stencil | 2.54 | 4.09 |

**Prioritize a conservative pointwise-chain fusion pass.** Across the four sizes, the manually
fused control is 5.76–6.17x faster than the unfused chain with ordinary launches and 3.01–4.98x
faster when both use graphs. It removes all five logical temporaries and five launches while
retaining the six arithmetic stages. This establishes headroom, not an automatic-fusion result.
Start with single-use, static, contiguous pointwise intermediates and matching index/bounds
semantics. Preserve floating-point operations and reject neighbor-dependent stencil fusion,
side effects, and uncertain aliasing. Use these cases as acceptance tests for the pass.

**Keep graph selection explicit.** Graphs improve native burst throughput by 1.27–1.29x for the
small six-node cases, but are slower for one/two-node cases and do not improve the million-element
cases here. At 65,536 elements, fork/join benefits while the linear chain does not. Synchronous
Python latency often improves even where native burst throughput worsens, so a universal
node-count heuristic would mix different objectives. First graph setup amortizes after roughly
22–34 native repeats in the beneficial cases in this run; do not apply graphs unconditionally.

**The existing memory planner is already useful.** The chain uses two slots for five logical
temporaries (60% fewer allocated bytes than separate aligned allocations); fork/join uses three
(40% fewer). At one million elements, these are 20 MiB → 8 MiB and 20 MiB → 12 MiB. The branched
plan retains two independent roots. More complex scheduling/reuse heuristics are a lower priority
than eliminating these pointwise intermediates altogether.

**Revisit asynchronous submission/batching after fusion.** Tiny single-kernel programs take
about 1.3 µs in native bursts versus 5–6 µs per synchronous Python call. The public API has
measurable overhead, but the native-burst path is currently benchmark instrumentation rather than
an application API. Preserve compiled programs and bound invocations today; follow fusion with a
small explicit submission/synchronization API if real calling patterns need that throughput.
Compilation/JIT costs were about 18–26 ms per specialization and are excluded from steady times.

The larger cases also exposed a correctness issue before measurements were accepted: pageable
host-to-device copies could finish staging before the device transfer completed, while CKL's
nonblocking execution stream proceeded independently. Buffer copies now wait for transfer
completion, and clearing a buffer also waits before returning. A million-element alternating
input/poison-output regression was added to the CUDA frontend acceptance test. The full recorded
matrix was rerun after this fix; the initial failed run is not used in the report.

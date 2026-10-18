# Complete autoregressive decoder: orchestration experiment

**Status:** implemented and measured. [GPU trace deep dive](results/deep-dive/REPORT.md),
[interactive trace viewer](results/deep-dive/traces/viewer.html), [initial report](results/rtx5060ti.md),
[raw samples/topology](results/rtx5060ti.json), [CKL kernels/program](workloads.py),
[independent NumPy reference](reference.py), [runner](run.py).

The experiment asks whether generic effect-derived orchestration improves a complete decoding
program over reasonable sequential execution. Every configuration executes identical CKL device
kernels with pointwise fusion disabled; only dependencies, temporary placement, and steps per
submission change. Outperforming an optimized serving engine is not an acceptance requirement.

## Model and timed boundary

The default randomly initialized transformer has two pre-norm layers, hidden size 64, four MHA
heads of dimension 16, SwiGLU width 128, and vocabulary 256. Storage/math is f32, with i32 token
IDs and positions. It uses adjacent-pair RoPE, a contiguous per-layer KV cache, stable causal
softmax, residuals, an untied vocabulary projection, and greedy selection with first-index tie
breaking. There are no biases or early EOS. This is a small model computation, not a trained
checkpoint or a text-quality evaluation.

```text
token embedding
→ [RMSNorm → Q/K/V projections → RoPE → KV append → causal attention
   → output projection → residual → RMSNorm → gated MLP → residual] × layers
→ final RMSNorm → vocabulary projection → greedy selection → position advance
```

There are 48 dispatches per step for two layers. Dot products/reductions use straightforward
per-thread loops rather than tuned shared-memory/tensor-core implementations. Kernels compile
from CKL source; weights and persistent buffers are allocated/bound once. The baseline has no
per-kernel host synchronization or intermediate host transfers.

The runner prepares deterministic prompts and reference-prefills `context - 1` tokens outside
timing. The last prompt token is the first measured input. Each sample generates 16 successive
tokens per request: token IDs, history, position, and K/V update on device, and contexts genuinely
grow. Timing excludes prefill, initialization, transfers, resets, compilation, and graph setup.
No requests arrive or leave during a sample.

## Ablations

| Variant | Temporary reuse | Execution |
| --- | --- | --- |
| `sequential` | Off | Native ordinary launches in source/topological order on one stream |
| `capture` | Off | Actual single-stream capture of those launches, replayed per step |
| `inferred` | Off | Explicit graph from inferred dependencies, replayed per step |
| `inferred_reuse` | On | Same inferred graph with lifetime-based temporary reuse |
| `sequential_reuse`, `capture_reuse` | On | Matched controls for storage placement |
| `group{2,4,8}_{capture,inferred}` | On | Replay a graph containing a partial group of complete steps |
| `batched_capture` | On | Capture all 16 complete steps in one graph |
| `batched_inferred` | On | Repeat the inferred plan 16 times in one graph |

Batched inferred plans order every next-iteration root after all previous leaves, allowing safe
temporary reuse. No model-specific dependency annotations are provided. The runner checks that
storage reuse leaves prepared kernel IR identical and records topology, dispatch counts, and
heap sizes. The public API remains synchronous; batching is not an asynchronous serving scheduler.

Native stream/wall times average a complete step across a growing-context sequence and include
native submission gaps. Python generation latency covers completion of all 16 steps, with one
wait per submission. Batched execution exposes outputs to the host only after the group finishes;
its speedup must not be presented as streamed inter-token latency. Setup times are separate.

## Reproduce

Use a Release CKL build with CUDA runtime, NVPTX, and ExecutionEngine enabled:

```bash
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCKL_ENABLE_MLIR=ON -DCKL_ENABLE_CUDA_RUNTIME=ON \
  -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir \
  -DCUDAToolkit_ROOT=/path/to/cuda
cmake --build build-bench --target CKLCompiler ckl-opt -j 8
```

Only NumPy is required by this runner; there is no PyTorch, FlashInfer, or model-download dependency.

```bash
cmake --build /tmp/ckl-standalone-bench --target CKLCompiler ckl-opt -j 8
PYTHONPATH=python \
CKL_NATIVE_LIBRARY=/tmp/ckl-standalone-bench/python/ckl/libCKLCompiler.so \
CUDA_CACHE_PATH=/tmp/ckl-decoder-cuda \
/path/to/numpy-env/bin/python benchmarks/decoder/run.py \
  --toolkit /path/to/cuda --chip sm_120 \
  --cases 1x1 1x8 8x8 1x64 8x64 32x64 \
  --steps 16 --samples 7 --warmup 3 \
  --output benchmarks/decoder/results/local.json
```

Cases are `batch x initial context`. Recorded results use RTX 5060 Ti, CUDA toolkit 12.8,
driver 595.91.07, and Python 3.12.13; raw JSON records NumPy/platform versions and source/native
hashes. Adjust paths/target for your environment. Incomplete JSON is retained on failure;
reports require a complete correctness-checked matrix.

### Extended cases, independent requests, and GPU traces

`--suite extended` selects nine named configurations: small, awkward dimensions, context 512,
hidden 256 with FFN 512 and vocabulary 2048, eight layers, vocabulary 8192, batch 64,
and matched batch-four controls for the small/wide models. The wide preset changes several
model dimensions together; it is not an isolated hidden-width ablation. `--groups 2 4 8`
adds partial step grouping when a group divides `--steps`; `--groups` alone disables those extras.
The default six custom shapes remain available through `--cases`.

The request experiment runs two, four, or eight independent invocations with shared read-only
model weights and distinct prompts, tokens, positions, KV caches, histories, and scratch heaps.
It compares serialized bursts with simultaneous native bursts on separate streams using a
persistent Python thread pool; ctypes releases the GIL during each burst. Reported completion
latency includes worker wakeup/barrier costs. This is an explicit concurrency experiment using
isolated state, not compiler-inferred cross-request scheduling or continuous batching. Matched
batch-four controls quantify the advantage of ordinary batching when requests can be batched.

With the environment variables above set, collect all measurements **serially**:

```bash
python benchmarks/decoder/collect.py --toolkit /path/to/cuda --chip sm_120 \
  --nsys /path/to/recent/nsys --matrix --output benchmarks/decoder/results/local-deep-dive
python3 benchmarks/decoder/analyze.py benchmarks/decoder/results/local-deep-dive/traces
```

Use a recent Nsight Systems version supporting your GPU/CUDA driver; these traces use
2026.3.1, rather than the older system `nsys` 2023.4. `collect.py` performs unprofiled timings
before profiling and records its exact profile commands. CUDA profiler API ranges surround
only warmed decode bursts. `--cuda-graph-trace=node` records individual graph kernels.
Every profile is checked against the NumPy reference after capture, and the analyzer checks
kernel/submission counts, API success, known kernel names, and absence of transfer/reset work.

The analyzer uses only the Python standard library. It exports compressed Chrome/Perfetto
traces, standalone SVG timelines, a self-contained HTML viewer (open locally in a modern
browser), and timestamp-based overlap/gap statistics. GPU packing lanes are visualization
rows, not SM occupancy measurements. Original stream/graph IDs and launch geometry remain
in trace event arguments. Native `.nsys-rep`/SQLite files remain local and ignored by git;
rerun collection to regenerate them. `--skip-requests --resume` retries an interrupted trace
collection without discarding already exported profiles. Profiling substantially affects
submission gaps, so performance ratios always come from the separate unprofiled runs.

Additional awkward-size validation, with the same environment:

```bash
python benchmarks/decoder/run.py --toolkit /path/to/cuda \
  --cases 3x3 --hidden 48 --heads 3 --ffn 80 --vocab 129 --layers 1 \
  --steps 3 --samples 2 --warmup 1 --output /tmp/decoder-awkward.json
```

## Correctness

The independent reference uses vectorized NumPy linear algebra. Every native timing sample and
Python generation sample checks exact tokens, complete history, and position, plus numerical
final logits and full K/V caches. Prefilled cache and unused suffixes must remain bitwise unchanged.
Weights/metadata and 256-element allocation tails are checked for unintended writes. Each
unbatched execution mode also receives a one-step check.

Tolerance is `atol=2e-5, rtol=3e-4`; retained maximum absolute error is below 1e-6. Context 1
exercises a zero-trip softmax maximum loop, greedy selection carries multiple scalar values,
and the awkward-size run covers partial blocks and a non-power-of-two vocabulary. Valid token
IDs and `0 <= position < capacity` are kernel input preconditions. The runner ensures enough
capacity for every reset sequence.

## Findings and limits

- Inferred graphs are **1.16–1.38x** faster than native sequential launches and **1.10–1.18x**
  faster than single-stream capture with identical unreused storage and kernels.
- Reuse reduces temporary heap bytes **71–78%**. Timing changes are small/mixed; this establishes
  a memory benefit, not an additional speedup claim.
- Batched inferred execution is **1.18–1.41x** faster than ordinary native bursts. Python
  completion latency improves **1.23–1.52x**, with the changed observation granularity above.
- At batch 1/context 8: sequential, capture, inferred, and batched-inferred native times are
  approximately 62.8, 54.9, 46.7, and 45.4 µs/step. Temporary storage is 14,080 → 3,072 bytes.

This demonstrates orchestration benefits for one small decode architecture. It does not establish
trained large-model serving performance or meet the broader goal of gains across multiple
distinct applications. No trace-based hardware-overlap claim is made: dependencies establish
legal independence, and timings establish elapsed-time differences. Graph setup and unlocked
clocks limit extrapolation.

Next experiments should vary model/kernel size, mixed context lengths, independent request
groups, and setup amortization. Continuous batching, paged cache, GQA, half precision, stochastic
sampling, cancellation/EOS, and asynchronous ownership remain separate extensions.

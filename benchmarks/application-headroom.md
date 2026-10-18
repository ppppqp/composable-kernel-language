# Application optimization headroom: first measurements

Measured on an RTX 5060 Ti (16 GB, 32 MiB L2), Release CKL, CUDA toolkit 12.8,
driver 595.91.07, CUDA PyTorch 2.13.0+cu130, and Python 3.12.13. These are single-GPU,
resident-device component measurements with synthetic deterministic inputs and unlocked clocks.
See [SGD results](momentum-sgd/results/rtx5060ti.md),
[normalization results](image-preprocessing/results/rtx5060ti.md), and their raw JSON files for
source/library hashes, exact configurations, individual samples, errors, setup costs, and topology.

## What was implemented

- Dense f32 SGD with weight decay and established momentum: four natural reference operations
  update parameters and momentum in place. The manual control performs the same update in one
  kernel and removes the adjusted-gradient temporary. The single parameter tensor is deliberately
  not a multi-tensor batching benchmark.
- Float RGB normalization with per-channel statistics and NCHW layout: a three-kernel
  copy/subtract/divide sequence, a two-kernel copy-elided sequence, and a one-kernel manual control.
  This is the normalization component, not resize/crop/conversion, image inference, or training.
- CKL fusion off/on, manual CKL controls, ordinary/graph execution, PyTorch eager and compiled
  controls, and PyTorch foreach/fused SGD. Normalization also has a PyTorch copy-elided control.
  Neither the CKL frontend nor compiler/runtime needed new features or dependencies.

Five sizes per workload × seven backend variants = 70 backend/shape combinations, each measured
in both execution modes with seven paired/rotated samples. Every variant passed independent
NumPy numerical checks, immutable-input checks, and tail-sentinel checks for one step, 17 steps,
100-step bursts, and 10-step latency sequences. SGD checks both persistent outputs. Maximum
observed absolute errors were below 1e-6 in the retained matrices; acceptance uses
`atol=3e-6, rtol=3e-5`, permitting floating-point contraction differences across implementations.

## Complete-step headroom

Rounded medians below use CKL-to-CKL **graph stream elapsed time**, with the same submission
and timing mechanism for all controls:

| Workload | Staged | Copy-elided | Manual single kernel | Staged/manual |
| --- | ---: | ---: | ---: | ---: |
| SGD, 257 parameters | 4.11 µs | — | 2.04 µs | 2.01x |
| SGD, 1,048,576 parameters | 28.68 µs | — | 10.25 µs | 2.80x |
| SGD, 16,777,216 parameters | 1.859 ms | — | 0.857 ms | 2.17x |
| RGB normalization, 8 × 224 × 224 | 28.64 µs | 18.47 µs | 12.28 µs | 2.33x |
| RGB normalization, 8 × 1080 × 1920 | 3.052 ms | 2.040 ms | 1.021 ms | 2.99x |

The large working sets exceed L2: SGD alone has 192 MiB of parameter/gradient/momentum state
plus a 64 MiB staged temporary; normalization has about 380 MiB of source/output data. Thus the
large-case headroom does not depend on fitting the complete working set into cache.

**The existing automatic pass provides no structural improvement on either workload.** It leaves
SGD at four dispatches and normalization at three. Small on/off timing differences are measurement
variation, not evidence of optimization. The manual implementations establish attainable headroom;
they are not new automatic transformations.

**Copy elimination alone leaves useful headroom.** For the large normalization batch, removing
the copy provides 1.50x, and combining subtraction/division provides another 2.00x. There are no
compiler-owned temporary heaps in that normalization program, so temporary-storage packing cannot
remove these extra full-array passes. SGD's one temporary also offers no opportunity to reduce
its own storage through reuse with another temporary.

**Graphs do not recover the large-case gap.** Large ordinary and graph times are nearly equal
for these bandwidth-dominated programs. For tiny manually fused SGD, ordinary native bursts take
1.34 µs versus 2.04 µs with a graph. Keep execution mode explicit; fewer dispatches do not imply
that a graph is beneficial.

## External controls and timing limits

At the largest SGD size, PyTorch fused and compiled graph stream medians are approximately
0.859 ms and 0.857 ms, compared with CKL manual at 0.857 ms. For large normalization, compiled
PyTorch is approximately 1.037 ms versus CKL manual at 1.021 ms. These support the interpretation
that the manual CKL versions are reasonable headroom controls, not deliberately weak baselines.
The small percentage differences are not grounds to claim a general framework advantage.

CKL event bursts submit from native C++; PyTorch event bursts submit from Python. Both include
submission gaps, so raw event-time ratios between frameworks are not isolated kernel-speed
comparisons. Reports also provide synchronous Python latency, with one invocation and completion
wait per step for both frameworks. PyTorch eager normalization reuses a preallocated output
instead of timing clone allocation. Compiled PyTorch SGD uses the direct update computation,
while its optimizer baselines include optimizer bookkeeping.

State/output resets, transfers, compilation, and graph setup are outside steady timing. Their
available setup measurements are retained separately. Gradients remain fixed during each SGD
sequence, and first momentum initialization is excluded. There are no claims about realistic
gradient evolution, many-tensor models, full image preprocessing, useful branch overlap, or
whole-training speedups.

## Implications for the next step

1. Keep these authored kernels as benchmarks and examples of expressing fused computation.
   The measured benefit does not require that CKL automatically merge arbitrary explicit kernels.
2. Avoid extending the current strict unary pass on the strength of the synthetic chain alone.
   Multi-input/output fusion would address SGD, but its compiler complexity should be weighed
   against simply authoring a suitable kernel and preserving a small language implementation.
3. Prioritize a small explicit submission/wait API and safe repeated execution if application
   throughput is the next objective. Tiny manual SGD is about 1.34 µs in native bursts versus
   5.65 µs per synchronous Python invocation; removing an unconditional per-call host wait is
   worth evaluating. Completion ownership and temporary reuse must remain correct across calls.
4. Add many-tensor SGD and a scientific stage before choosing more scheduling or storage work.
   They can expose batching, independence, loop-carried state, and region precision that these
   single-tensor pointwise components do not evaluate.

IREE's Stream layer groups commands and storage, but its earlier DispatchCreation pipeline also
fuses computation. See the [source-reading correction](../docs/iree-stream-notes.md#two-distinctions-that-matter).
The appropriate CKL boundary between authored kernels and automatic computation fusion remains
a design choice; IREE's division into compiler stages does not settle it.

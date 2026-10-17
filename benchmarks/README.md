# CKL benchmarks

The application workloads below are proposals, not implemented benchmarks or performance results.
Their purpose is to evaluate complete groups of kernels and let measurements guide language,
kernel, and execution-planning work. Keep the existing synthetic cases as compiler controls.

| Order | Workload | Timed unit | Main question |
| --- | --- | --- | --- |
| 1 | [Momentum SGD](momentum-sgd/README.md) | Complete dense optimizer step | Is multi-input/output pointwise fusion worthwhile? |
| 2 | [Image preprocessing](image-preprocessing/README.md) | Complete batch preprocessing | Does tail fusion improve a pipeline with indexing and conversion? |
| 3 | [AdamW](adamw/README.md) | Complete dense optimizer step | Can fusion handle branches, persistent state, and register pressure? |
| 4 | [miniWeather](miniweather/README.md) | Directional stage, then full timestep | What helps stencil pipelines when unary fusion cannot apply? |
| 5 | [Conjugate gradient](conjugate-gradient/README.md) | Complete iteration and solve | What helps repeated work with reductions and device-produced scalars? |

See the [IREE Stream comparison](../docs/iree-stream-notes.md) for related execution-planning ideas.

## Shared evaluation protocol

- Port the algorithm and preserve natural operation boundaries. Do not split an already fused
  reference operation into extra kernels to inflate an optimization opportunity. Record any
  differences in dtype, layout, state ownership, arithmetic, or algorithm.
- Pin upstream source revisions and record licenses/attribution before copying code. Links in
  these initial proposals identify references; they are not pinned reproduction inputs yet.
- Compare CKL fusion off/on with ordinary launches and explicit graphs where supported. Include
  a manually optimized CKL version to establish attainable headroom and a strong external GPU
  implementation. Record unsupported configurations rather than silently substituting another.
- Report complete-step time first. Component timings, actual dispatch counts, dependency graphs,
  logical temporary bytes, and allocated heap bytes explain the result. Allocation counts alone
  are not measurements of memory traffic or useful overlap; use profiling for those claims.
- Measure steady device/stream elapsed time and application-facing wall latency separately.
  Report compile/JIT, binding/allocation, graph setup, first execution, and amortization separately.
  Label submission gaps included in event timing. Exclude transfers from resident-device results;
  report transfer-inclusive timing separately where relevant.
- Sweep launch-bound, intermediate, and large working sets, including working sets larger than
  the measured GPU's last-level cache. Record cache/repeated-buffer behavior. Use paired samples,
  warmup, dispersion, device/toolchain metadata, and identical timing boundaries across baselines.
- Validate against an independent reference before accepting timings. Check every persistent
  output, unchanged inputs where applicable, awkward sizes, boundaries, and repeated execution.
  Specify numerical tolerances and floating-point policy. Stateful samples must start from
  equivalent state; reset outside timing and validate the actual number of repeated steps.
- Retain negative results: no fusion, graph slowdown, memory-only improvements, and faster
  components with negligible whole-program benefit are useful outcomes.

## Existing measurements

- [Standalone source-to-CUDA controls](standalone/README.md): affine, pointwise chain, fork/join,
  and stencil programs with ordinary/graph execution and automatic-fusion comparisons.
- [Milestone 0](milestone0/README.md), [Milestone 3](milestone3/README.md), and
  [Milestone 4](milestone4/README.md): historical native orchestration/runtime experiments.

The application proposals have no runner or reproduction command yet. Each directory will retain
its source, reference/checker, measurement driver, and results when implemented.

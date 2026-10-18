# Momentum SGD with weight decay

**Status:** implemented and measured for a single dense f32 tensor with established momentum.
See [the retained report](results/rtx5060ti.md) and [raw samples](results/rtx5060ti.json).

## Source and scope

Port the dense, non-Nesterov update from [PyTorch SGD](https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py).
Use nonzero momentum and weight decay, zero dampening, and no sparse gradients or mixed-precision
gradient scaling. The JSON pins the installed PyTorch source revision and file hash.

For an established momentum buffer, one complete step is:

```text
g_adjusted = gradient + weight_decay * parameters
velocity_new = momentum * velocity + g_adjusted
parameters_new = parameters - learning_rate * velocity_new
```

Preserve both updated parameters and velocity. PyTorch initializes the first momentum buffer
from the adjusted gradient; that initialization is outside this established-state benchmark.
This measures an optimizer step, not forward/backward propagation or complete training.

## Why this workload

The update is pointwise but has multiple inputs, a value with multiple uses, and persistent
outputs. CKL's current unary-chain recognizer cannot fuse the full update. This makes it a
direct test of the value of extending fusion, without first needing reductions or math intrinsics.
The basic arithmetic fits today's frontend. Both CKL and PyTorch update parameters and momentum
in place, retaining the original gradient. All variants borrow the same PyTorch-owned allocations.

## Implementation and comparisons

The [workloads](workloads.py) contain the reference's natural four operations: adjusted gradient,
in-place momentum scaling, momentum accumulation, and parameter update. The manual control
computes the same update in one kernel with two stores. This is an authored control, not a new
automatic compiler optimization.

The [runner](run.py) compares CKL fusion off/on and the manual control, each with ordinary and
graph execution, plus PyTorch ordinary, foreach, fused SGD, and `torch.compile`. The compiled
PyTorch control uses the same update equations directly, excluding optimizer-object bookkeeping.
There is one parameter tensor, so these results do not evaluate multi-tensor batching.

Start with f32 vectors of 257, 4,096, 65,536, 1,048,576, and 16,777,216 elements where memory
allows. Collections of varied tensor sizes remain a future extension; no implicit flattening or
multi-tensor speedup is claimed.

## Reproduce

Use a CUDA PyTorch + NumPy Python environment and a Release CKL native build. No TorchVision
installation is required. Build instructions are in the [standalone benchmark](../standalone/README.md).

```bash
PYTHONPATH=python \
CKL_NATIVE_LIBRARY=/tmp/ckl-standalone-bench/python/ckl/libCKLCompiler.so \
TORCHINDUCTOR_CACHE_DIR=/tmp/ckl-app-inductor \
TRITON_CACHE_DIR=/tmp/ckl-app-triton \
CUDA_CACHE_PATH=/tmp/ckl-app-cuda \
XDG_CACHE_HOME=/tmp/ckl-app-cache \
/path/to/cuda-pytorch-env/bin/python benchmarks/momentum-sgd/run.py \
  --toolkit /path/to/cuda --chip sm_120 \
  --sizes 257 4096 65536 1048576 16777216 \
  --iterations 100 --samples 7 --warmup 10 --latency-iterations 10 \
  --output benchmarks/momentum-sgd/results/local.json
```

The runner writes raw JSON and a Markdown report, retaining incomplete JSON on failure. Use
`--no-torch-compile` if a compiled baseline is unavailable; the omitted baseline is recorded.

## Correctness and decisions

The independent NumPy reference rounds each materialized arithmetic stage to f32. Checks cover
both persistent outputs, unchanged gradients, and 256-element tail sentinels on every buffer.
Each backend/mode is checked for one step, 17 steps, and the measured repeat counts. Inputs vary
across elements, with gradients fixed during a repeated sequence. State resets happen outside
timing. Tolerances are `atol=3e-6, rtol=3e-5`; bitwise agreement across FMA implementations is not
required. Other coefficient combinations, shifted aliases, and first-step initialization remain
additional cases rather than claims of this benchmark.

Native CKL bursts submit from C++; PyTorch bursts submit from Python, so their event times have
different submission overhead. Use CKL-to-CKL event comparisons for optimization headroom and
synchronous Python latency for application-facing comparisons. See the raw distributions as well
as medians. The single-tensor scope and unlocked clocks limit broader conclusions.

Follow the [shared protocol](../README.md#shared-evaluation-protocol). Report whole-step latency,
throughput in updated elements/s, dispatch count, and temporary/heap bytes. If manual fusion
provides substantial headroom, prioritize multi-input/output fusion; if it does not, investigate
kernel quality or submission cost before broadening the compiler.

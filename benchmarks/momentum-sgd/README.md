# Momentum SGD with weight decay

**Status:** proposed; no CKL implementation or results yet. First workload to implement.

## Source and scope

Port the dense, non-Nesterov update from [PyTorch SGD](https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py).
Use nonzero momentum and weight decay, zero dampening, and no sparse gradients or mixed-precision
gradient scaling. Pin the reference revision when implementing.

For an established momentum buffer, one complete step is:

```text
g_adjusted = gradient + weight_decay * parameters
velocity_new = momentum * velocity + g_adjusted
parameters_new = parameters - learning_rate * velocity_new
```

Preserve both updated parameters and velocity. PyTorch initializes the first momentum buffer
from the adjusted gradient; test that initialization separately from the established-state path.
This measures an optimizer step, not forward/backward propagation or complete training.

## Why this workload

The update is pointwise but has multiple inputs, a value with multiple uses, and persistent
outputs. CKL's current unary-chain recognizer cannot fuse the full update. This makes it a
direct test of the value of extending fusion, without first needing reductions or math intrinsics.
The basic arithmetic fits today's frontend. A first CKL port can use distinct old/new state
buffers, but that is a material difference from an in-place optimizer and must be reported.

## Implementation and comparisons

1. Implement the natural sequence of reference operations in a composed CKL program.
2. Implement a manually fused CKL kernel that computes and stores both outputs. Preserve the
   arithmetic/rounding contract when comparing with the staged version.
3. Measure CKL fusion off/on and ordinary/graph execution. Record the selected dispatch count;
   an unchanged graph is an expected initial result.
4. Compare PyTorch ordinary (`foreach=False, fused=False`), foreach, and fused paths where
   supported. Match math, state initialization, and timing boundaries. Report in-place and
   separate-output storage choices explicitly.

Start with f32 vectors of 257, 4,096, 65,536, 1,048,576, and 16,777,216 elements where memory
allows. Add collections of varied tensor sizes; record the exact distribution and distinguish
independent per-tensor dispatching from multi-tensor batching. Do not flatten tensors silently.

## Correctness and decisions

Check parameters and velocity after one and many steps against an independent reference, using
deterministic varying gradients. Reset state consistently outside each timed sample. Include
zero/nonzero coefficients, partial blocks, and supported in-place/alias configurations separately.
Account for both state buffers if the first implementation uses ping-pong storage.

Follow the [shared protocol](../README.md#shared-evaluation-protocol). Report whole-step latency,
throughput in updated elements/s, dispatch count, and temporary/heap bytes. If manual fusion
provides substantial headroom, prioritize multi-input/output fusion; if it does not, investigate
kernel quality or submission cost before broadening the compiler.

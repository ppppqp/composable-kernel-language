# Dense AdamW optimizer step

**Status:** proposed; no CKL implementation or results yet. Follow momentum SGD.

## Source and scope

Reference [PyTorch AdamW](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py) and the
[Adam implementation](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py) with decoupled
weight decay. Pin the source revision. Initially use dense f32, no AMSGrad, no sparse gradients,
no gradient scaling, and an explicit step counter.

The mathematical update is:

```text
m_new = beta1 * m + (1 - beta1) * gradient
v_new = beta2 * v + (1 - beta2) * gradient²
m_hat = m_new / (1 - beta1^step)
v_hat = v_new / (1 - beta2^step)
parameters_new = (1 - learning_rate * weight_decay) * parameters
                 - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

These equations specify the algorithm, not a required kernel decomposition or exact floating-point
evaluation order. Match the pinned reference's epsilon placement, decay, bias correction, and
operation order for strict comparisons; report any permitted numerical differences.

## Why this workload

One update contains independent moment branches, joins, persistent outputs, and nontrivial live
values. Current unary fusion cannot combine the full step. A manually fused kernel establishes
whether lower launch count and memory traffic compensate for increased register pressure.
The frontend needs at least a square-root intrinsic. Broader fusion and efficient multi-tensor
execution are separate potential improvements.

## Implementation and comparisons

Port natural reference operations and preserve parameters, both moments, and step state. Add a
manual CKL implementation, then compare CKL fusion off/on and ordinary/graph modes with PyTorch
ordinary, foreach, and fused AdamW paths where supported. Report old/new-buffer versus in-place
state handling. Bias correction may initially be supplied as host-computed scalars for a component
benchmark, but a complete-step result must include its computation and counter update consistently
across baselines; do not present a fixed-step graph replay as advancing optimizer state.

Use the [SGD size sweep](../momentum-sgd/README.md), including varied tensor collections. Start
with deterministic inputs and established state, then cover the first step and many updates.
Validate all moments, parameters, and counters, including small/zero gradients and epsilon-sensitive
values. Reset equivalent state outside each timing sample.

Follow the [shared protocol](../README.md#shared-evaluation-protocol). Report complete optimizer
step latency, updated elements/s, dispatch count, memory use, and profiled register/spill behavior
for manual and automatic fusion. This is an optimizer benchmark; training-speed claims need
separate whole-training measurements.

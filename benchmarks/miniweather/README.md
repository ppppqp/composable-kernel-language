# miniWeather timestep

**Status:** proposed; no CKL implementation or results yet. Scientific stencil workload.

## Source and scope

Port the numerical scheme from [miniWeather](https://github.com/mrnorman/miniWeather), a compact
atmospheric-flow mini-application. Pin its revision, configuration, precision, initial condition,
and reference accelerator implementation. The first scope is a single GPU with no MPI or file I/O.

```text
boundary/halo handling
→ reconstruct interface states and compute fluxes
→ compute flux divergence/tendencies
→ update the fluid state
→ repeat across Runge–Kutta stages and directional sweeps
```

Preserve the four physical state fields, halo widths, boundary conditions, hyperviscosity,
three-stage Runge–Kutta scheme, and directional-sweep order. Do not replace the physics with an
affine chain or independent heat-equation stencil while retaining the miniWeather name.

## Staged port and compiler questions

1. Implement one complete directional stage with an independent reference checker. Label this a
   component benchmark, including boundary handling and every resulting physical field.
2. Extend to the full timestep, then repeated timesteps with correct persistent state.
3. Compare natural CKL dispatches with a manually optimized implementation and the pinned
   upstream accelerator implementation, matching work, precision, and timing scope.

The port needs richer indexing, boundary control flow, math operations, and eventually explicit
program iteration beyond today's Python subset. The current unary-chain fusion pass is expected
to leave the important stencil boundaries intact. Opportunities include safe pointwise work
around stencils, temporary reuse, command grouping, and repeated submission. Temporal/stencil
fusion needs its own legality and redundant-work analysis; it cannot reuse unary forwarding blindly.

## Cases, correctness, and measurement

Use the upstream rising-thermal configuration first. Start with 128 × 64, 512 × 256, and
2,048 × 1,024 cells where memory permits, preserving the physical domain/aspect ratio and
appropriate timestep for each grid. Record the exact timestep and number of steps. Changing
grid resolution changes numerical work and stable timestep, so separate fixed-step throughput
from simulation-to-fixed-physical-time results.

Check every state field after a stage, a full timestep, and many timesteps. Compare boundary
cells, numerical norms, and upstream conservation/correctness diagnostics with a stated tolerance.
Account for expected floating-point drift without accepting unstable evolution.

Follow the [shared protocol](../README.md#shared-evaluation-protocol). Compare CKL fusion off/on
and ordinary/graph modes; report complete-step latency, cell updates/s, temporary heaps, setup,
and repeated execution overhead. Establish useful kernel overlap with a trace, not graph shape.
Use results to choose between kernel improvements, sub-buffer dependency precision, repeated
execution support, and orchestration changes.

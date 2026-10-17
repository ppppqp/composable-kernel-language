# Conjugate-gradient solver

**Status:** proposed; no CKL implementation or results yet. Reduction and iteration workload.

## Source and scope

Use [HPCG's CG routine](https://github.com/hpcg-benchmark/hpcg/blob/master/src/CG.cpp) as a reference
for the computational structure. Implement a defined single-GPU, unpreconditioned CG solver;
this is not an implementation or compliant result of the full HPCG benchmark. Pin references
and document the exact matrix generator, storage, precision, and stopping rule.

For an established iteration with rho = dot(r, r):

```text
Ap = A * p
alpha = rho / dot(p, Ap)
x_new = x + alpha * p
r_new = r - alpha * Ap
rho_new = dot(r_new, r_new)
beta = rho_new / rho
p_new = r_new + beta * p
```

The solution and residual updates can run independently after alpha is available. Reductions
create real synchronization boundaries. Alpha, beta, and reduction results must stay on the
device for the resident iteration benchmark; a host-readback implementation is a separate baseline.

## Staged port and cases

1. Start with an explicitly stored CSR positive-definite Poisson matrix with Dirichlet
   boundaries, a manufactured solution, and a documented right-hand side and initial guess.
   This needs indirect indexing/loops and an efficient reduction implementation beyond today's
   frontend. A matrix-free stencil version may be an additional, separately labeled case.
2. Implement a complete iteration with device-resident scalars and validate each intermediate.
3. Add repeated fixed-count iterations, then a solve with a specified convergence tolerance and
   maximum iteration count. Do not claim that a fixed-count component implements convergence.

Start with 2D grids of 64 × 64, 256 × 256, 1,024 × 1,024, and larger sizes as memory permits.
Record matrix dimension, nonzeros, conditioning-relevant choices, precision, and working-set size.
Use f64 when feasible for the scientific comparison; report f32 as a separate configuration.

## Baselines, correctness, and decisions

Compare CKL ordinary/graph execution and fusion off/on, a manually optimized CKL implementation,
and a GPU solver using the same matrix representation and algorithm, such as a CuPy/cupyx
implementation. Compare external solvers only after checking preconditioning, precision, and
stopping criteria. Use an independent CPU reference for correctness.

Validate one/many iterations, solution error, and the true residual norm ||b - Ax||, rather than
only the recursively updated residual. Include a converged initial guess and guard breakdown
or zero denominators. Report time per iteration and time to the same solution tolerance with
iteration counts; a faster iteration that takes more iterations is not necessarily a faster solve.

Follow the [shared protocol](../README.md#shared-evaluation-protocol). Report reduction and sparse
matvec costs, device/host synchronization, dispatch topology, temporary memory, setup, and total
solve latency. This workload guides reduction support, device-scalar dependencies, loop execution,
and command grouping even if pointwise fusion provides no measurable benefit.

# Automatic pointwise fusion measurements

RTX 5060 Ti, SM 120, CUDA 12.8, Python 3.12, Release native compiler. Each entry is the median of seven samples, each containing 200 native program submissions. Buffers are warm and reused; CUDA stream time includes native submission gaps.

| Elements | Ordinary, off → on µs | Ordinary speedup | Graph, off → on µs | Graph speedup | Planned temporary bytes, off → on |
|---|---:|---:|---:|---:|---:|
| 257 | 8.15 → 1.31 | 6.23x | 6.15 → 2.04 | 3.01x | 2,560 → 0 |
| 4,096 | 8.03 → 1.32 | 6.09x | 6.14 → 2.04 | 3.00x | 32,768 → 0 |
| 65,536 | 7.93 → 1.29 | 6.13x | 8.19 → 2.05 | 3.99x | 524,288 → 0 |
| 1,048,576 | 39.09 → 6.85 | 5.71x | 40.66 → 8.20 | 4.96x | 8,388,608 → 0 |

The six-stage chain becomes one kernel, eliminating all five logical temporaries. The manually fused control takes 1.25–1.31 µs for the three smaller sizes and 6.77 µs for 1,048,576 elements with ordinary launches; automatic fusion takes 1.29–1.32 µs and 6.85 µs respectively in the fusion-enabled run.

All 40 workload/size/fusion configurations passed numerical, input-preservation, and output-tail checks. The stencil and fork/join retain their original two- and six-node topologies. CPU structural tests cover launch/bounds mismatches, escaping and multiply consumed temporaries, explicit tokens, synchronization barriers, relaxed arithmetic, and pass ordering. GPU regression tests additionally verify shifted-overlap fallback, exact alias fallback, adjacent disjoint slices, and rounding-sensitive multiply/add behavior.

For the eligible chain, compile/JIT time increases from roughly 20–22 ms to 40–42 ms because the original program is compiled and retained for overlap fallback. These costs are excluded from steady-state times. Other patterns do not build a fallback when no fusion occurs. There is no generic profitability model for long chains; fusion can be disabled with `fuse_pointwise=False`.

The two matrices ran consecutively with alternating ordinary/graph sample order within each case. Clock settings were not locked; modest differences in unchanged controls should not be attributed to fusion. The initial benchmark report remains available as historical evidence motivating this optimization.

- [Fusion disabled: full report](rtx5060ti-fusion-off.md), [raw samples](rtx5060ti-fusion-off.json).
- [Fusion enabled: full report](rtx5060ti-fusion-on.md), [raw samples](rtx5060ti-fusion-on.json).
- [Runtime and legality contract](../../../docs/cuda-backend.md#pointwise-chain-fusion).

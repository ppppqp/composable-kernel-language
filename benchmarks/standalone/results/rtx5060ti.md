# Standalone CKL measurements

GPU: NVIDIA GeForce RTX 5060 Ti. Python 3.12.13. Native build: Release.

Times below are medians in microseconds per complete program. CUDA-event stream time includes native host submission gaps. These are warm repeated-buffer measurements, not isolated kernel execution or cold-memory bandwidth measurements.

| Case | Elements | Ordinary stream µs | Graph stream µs | Graph speedup | Python ordinary µs | Python graph µs |
|---|---:|---:|---:|---:|---:|---:|
| single_affine | 257 | 1.32 | 2.05 | 0.65x | 5.76 | 5.12 |
| chain_six | 257 | 7.87 | 6.15 | 1.28x | 12.53 | 8.49 |
| fused_chain_six | 257 | 1.31 | 2.04 | 0.64x | 5.45 | 5.17 |
| fork_join | 257 | 7.87 | 6.18 | 1.27x | 11.82 | 8.22 |
| stencil_two | 257 | 2.54 | 4.09 | 0.62x | 7.15 | 6.31 |
| single_affine | 4,096 | 1.31 | 2.05 | 0.64x | 5.51 | 5.28 |
| chain_six | 4,096 | 7.90 | 6.15 | 1.29x | 12.00 | 8.53 |
| fused_chain_six | 4,096 | 1.30 | 2.04 | 0.64x | 5.65 | 5.20 |
| fork_join | 4,096 | 7.84 | 6.17 | 1.27x | 12.23 | 8.36 |
| stencil_two | 4,096 | 2.48 | 4.09 | 0.61x | 6.85 | 6.16 |
| single_affine | 65,536 | 1.32 | 2.05 | 0.64x | 5.81 | 5.36 |
| chain_six | 65,536 | 7.85 | 8.19 | 0.96x | 13.01 | 10.53 |
| fused_chain_six | 65,536 | 1.27 | 2.05 | 0.62x | 5.74 | 5.54 |
| fork_join | 65,536 | 7.91 | 6.21 | 1.27x | 12.99 | 9.97 |
| stencil_two | 65,536 | 2.49 | 4.09 | 0.61x | 7.39 | 6.76 |
| single_affine | 1,048,576 | 6.49 | 8.19 | 0.79x | 11.21 | 10.84 |
| chain_six | 1,048,576 | 39.09 | 40.83 | 0.96x | 43.64 | 42.90 |
| fused_chain_six | 1,048,576 | 6.78 | 8.20 | 0.83x | 11.54 | 11.15 |
| fork_join | 1,048,576 | 39.82 | 40.95 | 0.97x | 44.84 | 43.52 |
| stencil_two | 1,048,576 | 14.57 | 16.38 | 0.89x | 19.31 | 18.82 |

| Case | Elements | Lower ms | Compile/JIT ms | First bind ms | Rebind ms | Graph instantiate ms | Estimated graph break-even repeats | Temporary bytes, baseline → planned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_affine | 257 | 0.48 | 25.72 | 0.095 | 0.068 | 0.052 | no steady saving | 0 → 0 |
| chain_six | 257 | 0.66 | 20.70 | 0.086 | 0.010 | 0.030 | 24 | 6,400 → 2,560 |
| fused_chain_six | 257 | 0.44 | 19.07 | 0.083 | 0.008 | 0.023 | no steady saving | 0 → 0 |
| fork_join | 257 | 0.73 | 22.78 | 0.094 | 0.011 | 0.042 | 32 | 6,400 → 3,840 |
| stencil_two | 257 | 0.56 | 21.05 | 0.085 | 0.009 | 0.021 | no steady saving | 1,280 → 1,280 |
| single_affine | 4,096 | 0.28 | 18.31 | 0.082 | 0.008 | 0.018 | no steady saving | 0 → 0 |
| chain_six | 4,096 | 0.64 | 20.89 | 0.087 | 0.012 | 0.029 | 22 | 81,920 → 32,768 |
| fused_chain_six | 4,096 | 0.44 | 19.14 | 0.085 | 0.007 | 0.018 | no steady saving | 0 → 0 |
| fork_join | 4,096 | 0.72 | 22.82 | 0.095 | 0.012 | 0.044 | 32 | 81,920 → 49,152 |
| stencil_two | 4,096 | 0.57 | 20.98 | 0.086 | 0.009 | 0.021 | no steady saving | 16,384 → 16,384 |
| single_affine | 65,536 | 0.29 | 19.05 | 0.096 | 0.008 | 0.020 | no steady saving | 0 → 0 |
| chain_six | 65,536 | 0.65 | 19.99 | 0.099 | 0.011 | 0.030 | no steady saving | 1,310,720 → 524,288 |
| fused_chain_six | 65,536 | 0.47 | 19.45 | 0.092 | 0.008 | 0.019 | no steady saving | 0 → 0 |
| fork_join | 65,536 | 0.73 | 23.77 | 0.114 | 0.065 | 0.044 | 34 | 1,310,720 → 786,432 |
| stencil_two | 65,536 | 0.57 | 20.91 | 0.094 | 0.010 | 0.022 | no steady saving | 262,144 → 262,144 |
| single_affine | 1,048,576 | 0.33 | 18.91 | 0.105 | 0.009 | 0.029 | no steady saving | 0 → 0 |
| chain_six | 1,048,576 | 0.70 | 20.81 | 0.257 | 0.064 | 0.041 | no steady saving | 20,971,520 → 8,388,608 |
| fused_chain_six | 1,048,576 | 0.47 | 18.95 | 0.111 | 0.010 | 0.030 | no steady saving | 0 → 0 |
| fork_join | 1,048,576 | 0.77 | 24.41 | 0.281 | 0.060 | 0.063 | no steady saving | 20,971,520 → 12,582,912 |
| stencil_two | 1,048,576 | 0.60 | 21.66 | 0.238 | 0.089 | 0.033 | no steady saving | 4,194,304 → 4,194,304 |

Estimated break-even includes measured graph instantiation and excess first-launch wall time, amortized against native burst wall-time savings. It excludes compilation and binding, which both execution modes share. It is an estimate from separate measurements.

All 20 cases passed full output, tail-sentinel, and unchanged-input checks. Raw JSON retains individual samples, ranges, topology, setup costs, and maximum errors.

| Elements | Chain graph / manually fused graph |
|---|---:|
| 257 | 3.01x |
| 4,096 | 3.01x |
| 65,536 | 3.99x |
| 1,048,576 | 4.98x |

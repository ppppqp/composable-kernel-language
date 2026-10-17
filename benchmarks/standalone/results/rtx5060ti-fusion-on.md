# Standalone CKL measurements

GPU: NVIDIA GeForce RTX 5060 Ti. Python 3.12.13. Native build: Release.

Automatic pointwise fusion: on.

Times below are medians in microseconds per complete program. CUDA-event stream time includes native host submission gaps. These are warm repeated-buffer measurements, not isolated kernel execution or cold-memory bandwidth measurements.

| Case | Elements | Ordinary stream µs | Graph stream µs | Graph speedup | Python ordinary µs | Python graph µs |
|---|---:|---:|---:|---:|---:|---:|
| single_affine | 257 | 1.32 | 2.04 | 0.65x | 5.44 | 5.14 |
| chain_six | 257 | 1.31 | 2.04 | 0.64x | 5.48 | 5.04 |
| fused_chain_six | 257 | 1.31 | 2.05 | 0.64x | 5.72 | 5.14 |
| fork_join | 257 | 7.87 | 6.18 | 1.27x | 12.31 | 8.19 |
| stencil_two | 257 | 2.57 | 4.09 | 0.63x | 7.27 | 6.24 |
| single_affine | 4,096 | 1.35 | 2.05 | 0.66x | 5.44 | 5.08 |
| chain_six | 4,096 | 1.32 | 2.04 | 0.64x | 5.37 | 5.11 |
| fused_chain_six | 4,096 | 1.25 | 2.05 | 0.61x | 5.73 | 5.05 |
| fork_join | 4,096 | 7.90 | 6.15 | 1.28x | 12.21 | 8.23 |
| stencil_two | 4,096 | 2.51 | 4.09 | 0.61x | 6.84 | 6.15 |
| single_affine | 65,536 | 1.32 | 2.05 | 0.64x | 6.03 | 5.50 |
| chain_six | 65,536 | 1.29 | 2.05 | 0.63x | 6.03 | 5.49 |
| fused_chain_six | 65,536 | 1.28 | 2.05 | 0.62x | 6.01 | 5.64 |
| fork_join | 65,536 | 7.88 | 6.26 | 1.26x | 13.02 | 10.03 |
| stencil_two | 65,536 | 2.54 | 4.10 | 0.62x | 7.40 | 6.81 |
| single_affine | 1,048,576 | 6.49 | 8.19 | 0.79x | 11.48 | 10.78 |
| chain_six | 1,048,576 | 6.85 | 8.20 | 0.84x | 11.63 | 11.21 |
| fused_chain_six | 1,048,576 | 6.77 | 8.20 | 0.83x | 11.50 | 11.21 |
| fork_join | 1,048,576 | 39.82 | 40.95 | 0.97x | 44.88 | 43.48 |
| stencil_two | 1,048,576 | 14.56 | 16.38 | 0.89x | 19.32 | 18.88 |

| Case | Elements | Lower ms | Compile/JIT ms | First bind ms | Rebind ms | Graph instantiate ms | Estimated graph break-even repeats | Temporary bytes, baseline → planned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_affine | 257 | 0.51 | 24.49 | 0.094 | 0.067 | 0.049 | no steady saving | 0 → 0 |
| chain_six | 257 | 0.66 | 39.92 | 0.084 | 0.008 | 0.018 | no steady saving | 0 → 0 |
| fused_chain_six | 257 | 0.44 | 18.59 | 0.083 | 0.008 | 0.018 | no steady saving | 0 → 0 |
| fork_join | 257 | 0.72 | 23.65 | 0.096 | 0.012 | 0.044 | 35 | 6,400 → 3,840 |
| stencil_two | 257 | 0.57 | 21.24 | 0.089 | 0.010 | 0.021 | no steady saving | 1,280 → 1,280 |
| single_affine | 4,096 | 0.28 | 18.70 | 0.082 | 0.008 | 0.018 | no steady saving | 0 → 0 |
| chain_six | 4,096 | 0.65 | 40.37 | 0.088 | 0.009 | 0.025 | no steady saving | 0 → 0 |
| fused_chain_six | 4,096 | 0.45 | 19.32 | 0.093 | 0.009 | 0.020 | no steady saving | 0 → 0 |
| fork_join | 4,096 | 0.74 | 23.21 | 0.096 | 0.011 | 0.047 | 40 | 81,920 → 49,152 |
| stencil_two | 4,096 | 0.58 | 21.09 | 0.090 | 0.009 | 0.027 | no steady saving | 16,384 → 16,384 |
| single_affine | 65,536 | 0.32 | 19.55 | 0.097 | 0.008 | 0.021 | no steady saving | 0 → 0 |
| chain_six | 65,536 | 0.68 | 41.84 | 0.105 | 0.008 | 0.023 | no steady saving | 0 → 0 |
| fused_chain_six | 65,536 | 0.45 | 19.23 | 0.100 | 0.008 | 0.023 | no steady saving | 0 → 0 |
| fork_join | 65,536 | 0.73 | 23.25 | 0.109 | 0.068 | 0.045 | 38 | 1,310,720 → 786,432 |
| stencil_two | 65,536 | 0.58 | 21.32 | 0.106 | 0.011 | 0.024 | no steady saving | 262,144 → 262,144 |
| single_affine | 1,048,576 | 0.40 | 19.97 | 0.107 | 0.009 | 0.029 | no steady saving | 0 → 0 |
| chain_six | 1,048,576 | 0.65 | 41.43 | 0.109 | 0.010 | 0.034 | no steady saving | 0 → 0 |
| fused_chain_six | 1,048,576 | 0.45 | 18.62 | 0.104 | 0.011 | 0.031 | no steady saving | 0 → 0 |
| fork_join | 1,048,576 | 0.75 | 23.43 | 0.276 | 0.058 | 0.062 | no steady saving | 20,971,520 → 12,582,912 |
| stencil_two | 1,048,576 | 0.59 | 21.54 | 0.214 | 0.094 | 0.032 | no steady saving | 4,194,304 → 4,194,304 |

Estimated break-even includes measured graph instantiation and excess first-launch wall time, amortized against native burst wall-time savings. It excludes compilation and binding, which both execution modes share. It is an estimate from separate measurements.

All 20 cases passed full output, tail-sentinel, and unchanged-input checks. Raw JSON retains individual samples, ranges, topology, setup costs, and maximum errors.

| Elements | Chain graph / manually fused graph |
|---|---:|
| 257 | 1.00x |
| 4,096 | 1.00x |
| 65,536 | 1.00x |
| 1,048,576 | 1.00x |

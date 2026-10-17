# Standalone CKL measurements

GPU: NVIDIA GeForce RTX 5060 Ti. Python 3.12.13. Native build: Release.

Automatic pointwise fusion: off.

Times below are medians in microseconds per complete program. CUDA-event stream time includes native host submission gaps. These are warm repeated-buffer measurements, not isolated kernel execution or cold-memory bandwidth measurements.

| Case | Elements | Ordinary stream µs | Graph stream µs | Graph speedup | Python ordinary µs | Python graph µs |
|---|---:|---:|---:|---:|---:|---:|
| single_affine | 257 | 1.41 | 2.04 | 0.69x | 5.71 | 5.36 |
| chain_six | 257 | 8.15 | 6.15 | 1.33x | 12.88 | 8.81 |
| fused_chain_six | 257 | 1.35 | 2.05 | 0.66x | 5.67 | 5.50 |
| fork_join | 257 | 8.19 | 6.17 | 1.33x | 12.87 | 8.35 |
| stencil_two | 257 | 2.66 | 4.10 | 0.65x | 7.46 | 6.39 |
| single_affine | 4,096 | 1.41 | 2.04 | 0.69x | 5.76 | 5.42 |
| chain_six | 4,096 | 8.03 | 6.14 | 1.31x | 12.67 | 8.62 |
| fused_chain_six | 4,096 | 1.32 | 2.05 | 0.64x | 5.93 | 5.31 |
| fork_join | 4,096 | 8.22 | 6.17 | 1.33x | 12.87 | 8.37 |
| stencil_two | 4,096 | 2.57 | 4.09 | 0.63x | 6.89 | 6.16 |
| single_affine | 65,536 | 1.33 | 2.05 | 0.65x | 5.92 | 5.45 |
| chain_six | 65,536 | 7.93 | 8.19 | 0.97x | 13.12 | 10.37 |
| fused_chain_six | 65,536 | 1.29 | 2.05 | 0.63x | 5.82 | 5.55 |
| fork_join | 65,536 | 7.92 | 6.22 | 1.27x | 13.01 | 10.10 |
| stencil_two | 65,536 | 2.54 | 4.10 | 0.62x | 7.42 | 6.81 |
| single_affine | 1,048,576 | 6.48 | 8.19 | 0.79x | 11.43 | 10.76 |
| chain_six | 1,048,576 | 39.09 | 40.66 | 0.96x | 43.86 | 42.89 |
| fused_chain_six | 1,048,576 | 6.76 | 8.19 | 0.83x | 11.60 | 11.21 |
| fork_join | 1,048,576 | 39.81 | 40.95 | 0.97x | 44.91 | 43.43 |
| stencil_two | 1,048,576 | 14.56 | 16.38 | 0.89x | 19.43 | 18.89 |

| Case | Elements | Lower ms | Compile/JIT ms | First bind ms | Rebind ms | Graph instantiate ms | Estimated graph break-even repeats | Temporary bytes, baseline → planned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_affine | 257 | 0.62 | 26.27 | 0.100 | 0.075 | 0.066 | no steady saving | 0 → 0 |
| chain_six | 257 | 0.75 | 21.71 | 0.090 | 0.012 | 0.028 | 19 | 6,400 → 2,560 |
| fused_chain_six | 257 | 0.49 | 20.00 | 0.086 | 0.009 | 0.018 | no steady saving | 0 → 0 |
| fork_join | 257 | 0.82 | 25.01 | 0.102 | 0.013 | 0.045 | 28 | 6,400 → 3,840 |
| stencil_two | 257 | 0.64 | 23.19 | 0.092 | 0.011 | 0.021 | no steady saving | 1,280 → 1,280 |
| single_affine | 4,096 | 0.32 | 19.62 | 0.087 | 0.009 | 0.020 | no steady saving | 0 → 0 |
| chain_six | 4,096 | 0.74 | 21.90 | 0.093 | 0.013 | 0.029 | 21 | 81,920 → 32,768 |
| fused_chain_six | 4,096 | 0.51 | 20.92 | 0.089 | 0.008 | 0.020 | no steady saving | 0 → 0 |
| fork_join | 4,096 | 0.82 | 25.69 | 0.100 | 0.013 | 0.047 | 29 | 81,920 → 49,152 |
| stencil_two | 4,096 | 0.65 | 21.93 | 0.088 | 0.010 | 0.022 | no steady saving | 16,384 → 16,384 |
| single_affine | 65,536 | 0.31 | 19.62 | 0.086 | 0.011 | 0.021 | no steady saving | 0 → 0 |
| chain_six | 65,536 | 0.64 | 20.17 | 0.104 | 0.012 | 0.032 | no steady saving | 1,310,720 → 524,288 |
| fused_chain_six | 65,536 | 0.46 | 19.10 | 0.096 | 0.019 | 0.022 | no steady saving | 0 → 0 |
| fork_join | 65,536 | 0.74 | 22.99 | 0.116 | 0.067 | 0.048 | 37 | 1,310,720 → 786,432 |
| stencil_two | 65,536 | 0.61 | 21.56 | 0.153 | 0.011 | 0.023 | no steady saving | 262,144 → 262,144 |
| single_affine | 1,048,576 | 0.39 | 20.30 | 0.103 | 0.008 | 0.030 | no steady saving | 0 → 0 |
| chain_six | 1,048,576 | 0.67 | 20.00 | 0.257 | 0.058 | 0.043 | no steady saving | 20,971,520 → 8,388,608 |
| fused_chain_six | 1,048,576 | 0.46 | 19.27 | 0.109 | 0.011 | 0.030 | no steady saving | 0 → 0 |
| fork_join | 1,048,576 | 0.76 | 24.12 | 0.267 | 0.059 | 0.063 | no steady saving | 20,971,520 → 12,582,912 |
| stencil_two | 1,048,576 | 0.60 | 20.81 | 0.237 | 0.095 | 0.030 | no steady saving | 4,194,304 → 4,194,304 |

Estimated break-even includes measured graph instantiation and excess first-launch wall time, amortized against native burst wall-time savings. It excludes compilation and binding, which both execution modes share. It is an estimate from separate measurements.

All 20 cases passed full output, tail-sentinel, and unchanged-input checks. Raw JSON retains individual samples, ranges, topology, setup costs, and maximum errors.

| Elements | Chain graph / manually fused graph |
|---|---:|
| 257 | 3.01x |
| 4,096 | 3.00x |
| 65,536 | 3.99x |
| 1,048,576 | 4.96x |

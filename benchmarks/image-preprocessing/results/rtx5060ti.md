# image-preprocessing measurements

GPU: NVIDIA GeForce RTX 5060 Ti; PyTorch 2.13.0+cu130; Python 3.12.13.

Times are medians in µs per complete component/step. CKL bursts submit from C++; PyTorch bursts submit from Python. Stream elapsed times include submission gaps and must not be used as a direct cross-framework kernel-speed comparison. Synchronous Python latency uses one submission and synchronization per step for both frameworks.

| Shape | Backend | Nodes (CKL) | Ordinary stream µs | Graph stream µs | Python ordinary µs | Python graph µs |
|---|---|---:|---:|---:|---:|---:|
| 1×257×259 | ckl_off | 3 | 6.707 | 8.195 | 12.512 | 10.900 |
| 1×257×259 | ckl_on | 3 | 6.684 | 8.198 | 12.448 | 10.826 |
| 1×257×259 | ckl_manual | 1 | 2.475 | 4.094 | 7.289 | 6.934 |
| 1×257×259 | ckl_copy_elided | 2 | 4.574 | 6.142 | 10.028 | 8.850 |
| 1×257×259 | torch_eager | — | 8.603 | 8.231 | 14.194 | 10.853 |
| 1×257×259 | torch_copy_elided | — | 5.747 | 6.175 | 11.427 | 9.028 |
| 1×257×259 | torch_compiled | — | 15.453 | 3.545 | 21.490 | 6.556 |
| 1×224×224 | ckl_off | 3 | 5.681 | 6.159 | 10.803 | 10.013 |
| 1×224×224 | ckl_on | 3 | 5.642 | 6.158 | 10.887 | 9.883 |
| 1×224×224 | ckl_manual | 1 | 2.083 | 4.090 | 6.826 | 6.467 |
| 1×224×224 | ckl_copy_elided | 2 | 3.860 | 4.113 | 8.683 | 8.051 |
| 1×224×224 | torch_eager | — | 8.402 | 6.180 | 13.416 | 9.873 |
| 1×224×224 | torch_copy_elided | — | 5.598 | 4.136 | 10.970 | 8.412 |
| 1×224×224 | torch_compiled | — | 17.556 | 4.126 | 24.668 | 6.608 |
| 8×224×224 | ckl_off | 3 | 26.549 | 28.642 | 31.456 | 31.053 |
| 8×224×224 | ckl_on | 3 | 26.546 | 28.645 | 31.410 | 30.996 |
| 8×224×224 | ckl_manual | 1 | 10.020 | 12.283 | 14.962 | 14.479 |
| 8×224×224 | ckl_copy_elided | 2 | 18.127 | 18.470 | 22.970 | 22.503 |
| 8×224×224 | torch_eager | — | 23.847 | 24.646 | 29.386 | 28.465 |
| 8×224×224 | torch_copy_elided | — | 18.484 | 20.521 | 24.791 | 23.473 |
| 8×224×224 | torch_compiled | — | 18.315 | 12.331 | 33.869 | 15.677 |
| 32×224×224 | ckl_off | 3 | 140.069 | 139.461 | 142.530 | 141.804 |
| 32×224×224 | ckl_on | 3 | 140.026 | 139.451 | 142.347 | 141.892 |
| 32×224×224 | ckl_manual | 1 | 69.903 | 70.604 | 75.012 | 74.279 |
| 32×224×224 | ckl_copy_elided | 2 | 103.052 | 102.644 | 108.313 | 107.193 |
| 32×224×224 | torch_eager | — | 137.076 | 134.365 | 134.216 | 133.784 |
| 32×224×224 | torch_copy_elided | — | 100.225 | 100.360 | 105.824 | 104.632 |
| 32×224×224 | torch_compiled | — | 71.494 | 71.315 | 87.917 | 72.251 |
| 8×1080×1920 | ckl_off | 3 | 3052.297 | 3052.050 | 3051.338 | 3052.228 |
| 8×1080×1920 | ckl_on | 3 | 3056.895 | 3058.911 | 3056.529 | 3056.759 |
| 8×1080×1920 | ckl_manual | 1 | 1021.319 | 1020.960 | 1021.213 | 1021.830 |
| 8×1080×1920 | ckl_copy_elided | 2 | 2039.709 | 2040.255 | 2039.998 | 2038.980 |
| 8×1080×1920 | torch_eager | — | 3084.705 | 3085.745 | 3086.941 | 3089.464 |
| 8×1080×1920 | torch_copy_elided | — | 2059.493 | 2059.221 | 2061.146 | 2061.166 |
| 8×1080×1920 | torch_compiled | — | 1038.682 | 1037.073 | 1063.222 | 1039.646 |

Manual optimization headroom below compares CKL against CKL with identical timing boundaries. Values above 1 favor the manually fused control.

| Shape | Ordinary off/manual | Graph off/manual | Graph on/off | Staged temp bytes |
|---|---:|---:|---:|---:|
| 1×257×259 | 2.71x | 2.00x | 1.00x | 0 |
| 1×224×224 | 2.73x | 1.51x | 1.00x | 0 |
| 8×224×224 | 2.65x | 2.33x | 1.00x | 0 |
| 32×224×224 | 2.00x | 1.98x | 1.00x | 0 |
| 8×1080×1920 | 2.99x | 2.99x | 1.00x | 0 |

Copy-elision versus arithmetic fusion (CKL graph stream medians):

| Shape | Staged/copy-elided | Copy-elided/manual |
|---|---:|---:|
| 1×257×259 | 1.33x | 1.50x |
| 1×224×224 | 1.50x | 1.01x |
| 8×224×224 | 1.55x | 1.50x |
| 32×224×224 | 1.36x | 1.45x |
| 8×1080×1920 | 1.50x | 2.00x |

All variants passed numerical, immutable-input, and tail-sentinel checks for one step, 17 steps, and the measured burst/latency repeat counts. Raw JSON contains individual samples, setup/compile costs, reference hashes, topology, and errors.

These are resident-device, warm repeated-buffer measurements with unlocked clocks. They exclude transfers and state resets. SGD uses an established momentum buffer and fixed gradients during each repeated sequence. Image preprocessing measures float RGB normalization only; resize/crop/decode/conversion are excluded. PyTorch normalization reuses a preallocated output, excluding allocation overhead.

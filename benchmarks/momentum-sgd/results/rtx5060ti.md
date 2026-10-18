# momentum-sgd measurements

GPU: NVIDIA GeForce RTX 5060 Ti; PyTorch 2.13.0+cu130; Python 3.12.13.

Times are medians in µs per complete component/step. CKL bursts submit from C++; PyTorch bursts submit from Python. Stream elapsed times include submission gaps and must not be used as a direct cross-framework kernel-speed comparison. Synchronous Python latency uses one submission and synchronization per step for both frameworks.

| Shape | Backend | Nodes (CKL) | Ordinary stream µs | Graph stream µs | Python ordinary µs | Python graph µs |
|---|---|---:|---:|---:|---:|---:|
| 257 | ckl_off | 4 | 5.266 | 4.112 | 8.917 | 7.847 |
| 257 | ckl_on | 4 | 5.094 | 4.118 | 9.107 | 7.574 |
| 257 | ckl_manual | 1 | 1.344 | 2.041 | 5.654 | 5.261 |
| 257 | torch_eager | — | 22.153 | 6.185 | 29.501 | 8.787 |
| 257 | torch_foreach | — | 32.140 | 6.186 | 40.210 | 8.661 |
| 257 | torch_fused | — | 15.648 | 2.102 | 23.646 | 6.077 |
| 257 | torch_compiled | — | 15.406 | 1.869 | 21.244 | 5.558 |
| 4096 | ckl_off | 4 | 5.156 | 4.109 | 9.171 | 7.507 |
| 4096 | ckl_on | 4 | 5.244 | 4.110 | 9.035 | 7.519 |
| 4096 | ckl_manual | 1 | 1.329 | 2.053 | 5.562 | 5.334 |
| 4096 | torch_eager | — | 22.244 | 4.158 | 29.358 | 8.290 |
| 4096 | torch_foreach | — | 31.834 | 6.199 | 40.337 | 9.869 |
| 4096 | torch_fused | — | 15.803 | 2.121 | 23.679 | 6.625 |
| 4096 | torch_compiled | — | 16.970 | 1.769 | 24.103 | 5.583 |
| 65536 | ckl_off | 4 | 5.063 | 6.144 | 10.083 | 8.905 |
| 65536 | ckl_on | 4 | 5.005 | 6.139 | 9.847 | 8.790 |
| 65536 | ckl_manual | 1 | 1.437 | 2.058 | 6.006 | 5.839 |
| 65536 | torch_eager | — | 21.992 | 6.177 | 29.365 | 9.186 |
| 65536 | torch_foreach | — | 31.781 | 28.689 | 52.184 | 31.315 |
| 65536 | torch_fused | — | 15.931 | 10.324 | 32.019 | 14.708 |
| 65536 | torch_compiled | — | 17.404 | 2.096 | 24.869 | 6.033 |
| 1048576 | ckl_off | 4 | 27.786 | 28.680 | 32.282 | 31.825 |
| 1048576 | ckl_on | 4 | 27.726 | 28.678 | 32.230 | 31.642 |
| 1048576 | ckl_manual | 1 | 9.563 | 10.254 | 14.297 | 14.095 |
| 1048576 | torch_eager | — | 22.168 | 22.567 | 35.355 | 25.793 |
| 1048576 | torch_foreach | — | 33.952 | 34.864 | 58.211 | 37.517 |
| 1048576 | torch_fused | — | 15.920 | 16.433 | 37.629 | 20.144 |
| 1048576 | torch_compiled | — | 17.348 | 10.282 | 30.448 | 13.824 |
| 16777216 | ckl_off | 4 | 1859.538 | 1859.477 | 1860.190 | 1861.150 |
| 16777216 | ckl_on | 4 | 1861.128 | 1860.436 | 1862.195 | 1862.452 |
| 16777216 | ckl_manual | 1 | 856.937 | 856.814 | 857.416 | 857.617 |
| 16777216 | torch_eager | — | 1877.332 | 1874.661 | 1891.169 | 1877.419 |
| 16777216 | torch_foreach | — | 1877.789 | 1876.805 | 1906.337 | 1881.656 |
| 16777216 | torch_fused | — | 859.059 | 859.017 | 883.568 | 863.675 |
| 16777216 | torch_compiled | — | 858.343 | 857.136 | 879.154 | 857.302 |

Manual optimization headroom below compares CKL against CKL with identical timing boundaries. Values above 1 favor the manually fused control.

| Shape | Ordinary off/manual | Graph off/manual | Graph on/off | Staged temp bytes |
|---|---:|---:|---:|---:|
| 257 | 3.92x | 2.01x | 1.00x | 1,280 |
| 4096 | 3.88x | 2.00x | 1.00x | 16,384 |
| 65536 | 3.52x | 2.99x | 1.00x | 262,144 |
| 1048576 | 2.91x | 2.80x | 1.00x | 4,194,304 |
| 16777216 | 2.17x | 2.17x | 1.00x | 67,108,864 |

All variants passed numerical, immutable-input, and tail-sentinel checks for one step, 17 steps, and the measured burst/latency repeat counts. Raw JSON contains individual samples, setup/compile costs, reference hashes, topology, and errors.

These are resident-device, warm repeated-buffer measurements with unlocked clocks. They exclude transfers and state resets. SGD uses an established momentum buffer and fixed gradients during each repeated sequence. Image preprocessing measures float RGB normalization only; resize/crop/decode/conversion are excluded. PyTorch normalization reuses a preallocated output, excluding allocation overhead.

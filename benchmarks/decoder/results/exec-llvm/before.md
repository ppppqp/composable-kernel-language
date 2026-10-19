# CKL autoregressive decoder measurements

GPU: NVIDIA GeForce RTX 5060 Ti. Native build: Release. Python 3.12.13, NumPy 2.5.4.

All variants execute identical unfused FP32 device kernels. The model has random weights and is an orchestration experiment, not a trained language model or a competitive attention implementation. Prompt prefill, transfers, resets, compilation, and graph setup are outside steady decode timing.

Native stream/wall times are per complete decode step (one token per request) and include native submission gaps. Each sample averages a growing-context generation sequence, rather than repeatedly decoding the same token/position. Python generation latency covers all generated steps, with one completion wait per submission; batched variants submit all steps at once. That metric measures generation completion, not streamed token latency.

| Case | Batch | Initial context | Variant | Stream µs/step | Native wall µs/step | Python generation ms | Temp heap bytes | Setup ms |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| small | 1 | 8 | sequential | 62.89 | 63.20 | 1.087 | 14,080 | 0.000 |
| small | 1 | 8 | capture | 54.80 | 55.01 | 0.925 | 14,080 | 0.095 |
| small | 1 | 8 | inferred | 46.75 | 47.00 | 0.798 | 14,080 | 0.079 |
| small | 1 | 8 | sequential_reuse | 63.01 | 63.20 | 1.084 | 3,072 | 0.000 |
| small | 1 | 8 | capture_reuse | 55.04 | 55.22 | 0.928 | 3,072 | 0.113 |
| small | 1 | 8 | inferred_reuse | 46.99 | 47.19 | 0.801 | 3,072 | 0.125 |
| small | 1 | 8 | batched_capture | 53.47 | 53.66 | 0.860 | 3,072 | 2.273 |
| small | 1 | 8 | batched_inferred | 45.62 | 45.81 | 0.733 | 3,072 | 2.879 |
| small | 1 | 8 | group2_capture | 54.03 | 54.21 | 0.892 | 3,072 | 0.289 |
| small | 1 | 8 | group2_inferred | 46.08 | 46.26 | 0.762 | 3,072 | 0.317 |
| small | 1 | 8 | group4_capture | 53.77 | 53.95 | 0.873 | 3,072 | 0.703 |
| small | 1 | 8 | group4_inferred | 45.74 | 45.91 | 0.745 | 3,072 | 0.720 |
| small | 1 | 8 | group8_capture | 53.64 | 53.82 | 0.864 | 3,072 | 1.338 |
| small | 1 | 8 | group8_inferred | 45.68 | 45.95 | 0.738 | 3,072 | 1.473 |
| awkward | 3 | 3 | sequential | 34.90 | 35.57 | 0.619 | 17,152 | 0.000 |
| awkward | 3 | 3 | capture | 28.00 | 28.18 | 0.497 | 17,152 | 0.064 |
| awkward | 3 | 3 | inferred | 24.56 | 24.75 | 0.439 | 17,152 | 0.066 |
| awkward | 3 | 3 | sequential_reuse | 35.08 | 35.27 | 0.625 | 6,144 | 0.000 |
| awkward | 3 | 3 | capture_reuse | 28.09 | 28.31 | 0.496 | 6,144 | 0.050 |
| awkward | 3 | 3 | inferred_reuse | 24.59 | 24.78 | 0.440 | 6,144 | 0.056 |
| awkward | 3 | 3 | batched_capture | 26.47 | 26.65 | 0.428 | 6,144 | 0.688 |
| awkward | 3 | 3 | batched_inferred | 23.06 | 23.23 | 0.373 | 6,144 | 0.738 |
| awkward | 3 | 3 | group2_capture | 27.15 | 27.33 | 0.461 | 6,144 | 0.088 |
| awkward | 3 | 3 | group2_inferred | 23.89 | 24.08 | 0.404 | 6,144 | 0.096 |
| awkward | 3 | 3 | group4_capture | 26.78 | 26.96 | 0.441 | 6,144 | 0.161 |
| awkward | 3 | 3 | group4_inferred | 23.40 | 23.59 | 0.386 | 6,144 | 0.177 |
| awkward | 3 | 3 | group8_capture | 26.64 | 26.82 | 0.432 | 6,144 | 0.307 |
| awkward | 3 | 3 | group8_inferred | 23.13 | 23.49 | 0.377 | 6,144 | 0.355 |
| long_context | 1 | 512 | sequential | 176.88 | 177.22 | 2.927 | 45,824 | 0.000 |
| long_context | 1 | 512 | capture | 171.69 | 171.99 | 2.797 | 45,824 | 0.097 |
| long_context | 1 | 512 | inferred | 163.58 | 163.88 | 2.669 | 45,824 | 0.111 |
| long_context | 1 | 512 | sequential_reuse | 177.18 | 177.62 | 2.932 | 18,944 | 0.000 |
| long_context | 1 | 512 | capture_reuse | 171.91 | 172.23 | 2.802 | 18,944 | 0.082 |
| long_context | 1 | 512 | inferred_reuse | 163.82 | 164.13 | 2.673 | 18,944 | 0.097 |
| long_context | 1 | 512 | batched_capture | 170.43 | 170.78 | 2.733 | 18,944 | 1.348 |
| long_context | 1 | 512 | batched_inferred | 162.50 | 162.76 | 2.605 | 18,944 | 1.474 |
| long_context | 1 | 512 | group2_capture | 171.19 | 171.56 | 2.766 | 18,944 | 0.148 |
| long_context | 1 | 512 | group2_inferred | 163.15 | 163.49 | 2.638 | 18,944 | 0.160 |
| long_context | 1 | 512 | group4_capture | 170.72 | 171.06 | 2.748 | 18,944 | 0.282 |
| long_context | 1 | 512 | group4_inferred | 162.67 | 162.99 | 2.619 | 18,944 | 0.302 |
| long_context | 1 | 512 | group8_capture | 170.48 | 170.78 | 2.739 | 18,944 | 0.537 |
| long_context | 1 | 512 | group8_inferred | 162.51 | 162.83 | 2.610 | 18,944 | 0.601 |
| wide | 1 | 8 | sequential | 154.20 | 154.92 | 2.545 | 47,360 | 0.000 |
| wide | 1 | 8 | capture | 148.97 | 149.34 | 2.433 | 47,360 | 0.101 |
| wide | 1 | 8 | inferred | 122.42 | 122.66 | 2.012 | 47,360 | 0.171 |
| wide | 1 | 8 | sequential_reuse | 154.61 | 155.02 | 2.540 | 10,240 | 0.000 |
| wide | 1 | 8 | capture_reuse | 148.98 | 149.28 | 2.437 | 10,240 | 0.102 |
| wide | 1 | 8 | inferred_reuse | 122.30 | 122.57 | 2.010 | 10,240 | 0.121 |
| wide | 1 | 8 | batched_capture | 147.60 | 147.88 | 2.374 | 10,240 | 1.181 |
| wide | 1 | 8 | batched_inferred | 121.28 | 121.54 | 1.950 | 10,240 | 1.237 |
| wide | 1 | 8 | group2_capture | 148.16 | 148.38 | 2.398 | 10,240 | 0.183 |
| wide | 1 | 8 | group2_inferred | 121.54 | 121.78 | 1.976 | 10,240 | 0.180 |
| wide | 1 | 8 | group4_capture | 147.71 | 148.16 | 2.379 | 10,240 | 0.316 |
| wide | 1 | 8 | group4_inferred | 121.24 | 121.63 | 1.958 | 10,240 | 0.328 |
| wide | 1 | 8 | group8_capture | 147.56 | 147.94 | 2.370 | 10,240 | 0.531 |
| wide | 1 | 8 | group8_inferred | 121.16 | 121.45 | 1.955 | 10,240 | 0.611 |
| deep | 1 | 8 | sequential | 231.84 | 232.07 | 3.756 | 54,016 | 0.000 |
| deep | 1 | 8 | capture | 194.32 | 194.53 | 3.160 | 54,016 | 0.283 |
| deep | 1 | 8 | inferred | 162.16 | 162.42 | 2.643 | 54,016 | 0.348 |
| deep | 1 | 8 | sequential_reuse | 231.83 | 232.19 | 3.753 | 3,072 | 0.000 |
| deep | 1 | 8 | capture_reuse | 194.14 | 194.32 | 3.154 | 3,072 | 0.257 |
| deep | 1 | 8 | inferred_reuse | 162.11 | 162.32 | 2.643 | 3,072 | 0.290 |
| deep | 1 | 8 | batched_capture | 192.71 | 192.91 | 3.086 | 3,072 | 5.415 |
| deep | 1 | 8 | batched_inferred | 161.04 | 161.26 | 2.578 | 3,072 | 11.549 |
| deep | 1 | 8 | group2_capture | 193.12 | 193.30 | 3.118 | 3,072 | 1.095 |
| deep | 1 | 8 | group2_inferred | 161.46 | 161.66 | 2.610 | 3,072 | 1.112 |
| deep | 1 | 8 | group4_capture | 192.86 | 193.04 | 3.100 | 3,072 | 2.985 |
| deep | 1 | 8 | group4_inferred | 161.21 | 161.39 | 2.592 | 3,072 | 2.480 |
| deep | 1 | 8 | group8_capture | 192.65 | 192.85 | 3.090 | 3,072 | 4.789 |
| deep | 1 | 8 | group8_inferred | 161.03 | 161.21 | 2.582 | 3,072 | 5.770 |
| large_vocab | 1 | 8 | sequential | 215.51 | 216.24 | 3.577 | 14,080 | 0.000 |
| large_vocab | 1 | 8 | capture | 209.50 | 210.09 | 3.408 | 14,080 | 0.102 |
| large_vocab | 1 | 8 | inferred | 201.45 | 202.05 | 3.280 | 14,080 | 0.110 |
| large_vocab | 1 | 8 | sequential_reuse | 215.35 | 216.03 | 3.567 | 3,072 | 0.000 |
| large_vocab | 1 | 8 | capture_reuse | 209.57 | 210.14 | 3.409 | 3,072 | 0.093 |
| large_vocab | 1 | 8 | inferred_reuse | 201.65 | 202.27 | 3.283 | 3,072 | 0.096 |
| large_vocab | 1 | 8 | batched_capture | 208.05 | 208.70 | 3.341 | 3,072 | 1.119 |
| large_vocab | 1 | 8 | batched_inferred | 200.24 | 200.86 | 3.216 | 3,072 | 1.161 |
| large_vocab | 1 | 8 | group2_capture | 208.71 | 209.33 | 3.373 | 3,072 | 0.161 |
| large_vocab | 1 | 8 | group2_inferred | 200.92 | 201.51 | 3.247 | 3,072 | 0.168 |
| large_vocab | 1 | 8 | group4_capture | 208.43 | 209.04 | 3.355 | 3,072 | 0.301 |
| large_vocab | 1 | 8 | group4_inferred | 200.38 | 201.00 | 3.227 | 3,072 | 0.303 |
| large_vocab | 1 | 8 | group8_capture | 208.19 | 208.83 | 3.346 | 3,072 | 0.552 |
| large_vocab | 1 | 8 | group8_inferred | 200.32 | 200.91 | 3.220 | 3,072 | 0.614 |
| large_batch | 64 | 64 | sequential | 102.01 | 102.76 | 1.713 | 1,017,088 | 0.000 |
| large_batch | 64 | 64 | capture | 95.83 | 96.34 | 1.596 | 1,017,088 | 0.127 |
| large_batch | 64 | 64 | inferred | 88.07 | 88.58 | 1.473 | 1,017,088 | 0.165 |
| large_batch | 64 | 64 | sequential_reuse | 101.60 | 102.44 | 1.703 | 294,912 | 0.000 |
| large_batch | 64 | 64 | capture_reuse | 95.54 | 96.04 | 1.589 | 294,912 | 0.181 |
| large_batch | 64 | 64 | inferred_reuse | 87.62 | 88.13 | 1.464 | 294,912 | 0.158 |
| large_batch | 64 | 64 | batched_capture | 94.22 | 94.72 | 1.523 | 294,912 | 1.311 |
| large_batch | 64 | 64 | batched_inferred | 86.51 | 87.01 | 1.402 | 294,912 | 1.330 |
| large_batch | 64 | 64 | group2_capture | 94.76 | 95.28 | 1.554 | 294,912 | 0.250 |
| large_batch | 64 | 64 | group2_inferred | 87.01 | 87.53 | 1.431 | 294,912 | 0.243 |
| large_batch | 64 | 64 | group4_capture | 94.40 | 94.96 | 1.537 | 294,912 | 0.383 |
| large_batch | 64 | 64 | group4_inferred | 86.51 | 87.03 | 1.413 | 294,912 | 0.394 |
| large_batch | 64 | 64 | group8_capture | 94.20 | 94.75 | 1.530 | 294,912 | 0.647 |
| large_batch | 64 | 64 | group8_inferred | 86.43 | 86.93 | 1.403 | 294,912 | 0.701 |
| batch4 | 4 | 8 | sequential | 63.65 | 63.85 | 1.125 | 50,432 | 0.000 |
| batch4 | 4 | 8 | capture | 55.33 | 55.54 | 0.935 | 50,432 | 0.103 |
| batch4 | 4 | 8 | inferred | 47.37 | 47.64 | 0.809 | 50,432 | 0.092 |
| batch4 | 4 | 8 | sequential_reuse | 63.77 | 63.97 | 1.104 | 11,264 | 0.000 |
| batch4 | 4 | 8 | capture_reuse | 55.44 | 55.62 | 0.934 | 11,264 | 0.084 |
| batch4 | 4 | 8 | inferred_reuse | 47.38 | 47.56 | 0.807 | 11,264 | 0.096 |
| batch4 | 4 | 8 | batched_capture | 53.93 | 54.12 | 0.867 | 11,264 | 1.193 |
| batch4 | 4 | 8 | batched_inferred | 46.01 | 46.19 | 0.739 | 11,264 | 1.208 |
| batch4 | 4 | 8 | group2_capture | 54.79 | 55.01 | 0.899 | 11,264 | 0.151 |
| batch4 | 4 | 8 | group2_inferred | 46.73 | 46.91 | 0.771 | 11,264 | 0.161 |
| batch4 | 4 | 8 | group4_capture | 54.26 | 54.43 | 0.880 | 11,264 | 0.284 |
| batch4 | 4 | 8 | group4_inferred | 46.24 | 46.42 | 0.751 | 11,264 | 0.301 |
| batch4 | 4 | 8 | group8_capture | 54.08 | 54.28 | 0.872 | 11,264 | 0.544 |
| batch4 | 4 | 8 | group8_inferred | 46.09 | 46.37 | 0.745 | 11,264 | 0.590 |
| wide_batch4 | 4 | 8 | sequential | 159.53 | 160.23 | 2.629 | 185,600 | 0.000 |
| wide_batch4 | 4 | 8 | capture | 154.24 | 154.79 | 2.524 | 185,600 | 0.110 |
| wide_batch4 | 4 | 8 | inferred | 128.63 | 129.13 | 2.113 | 185,600 | 0.121 |
| wide_batch4 | 4 | 8 | sequential_reuse | 159.09 | 159.85 | 2.622 | 40,960 | 0.000 |
| wide_batch4 | 4 | 8 | capture_reuse | 153.70 | 154.29 | 2.517 | 40,960 | 0.104 |
| wide_batch4 | 4 | 8 | inferred_reuse | 128.09 | 128.58 | 2.106 | 40,960 | 0.117 |
| wide_batch4 | 4 | 8 | batched_capture | 152.37 | 152.97 | 2.451 | 40,960 | 1.277 |
| wide_batch4 | 4 | 8 | batched_inferred | 126.95 | 127.48 | 2.043 | 40,960 | 1.242 |
| wide_batch4 | 4 | 8 | group2_capture | 153.08 | 153.62 | 2.482 | 40,960 | 0.195 |
| wide_batch4 | 4 | 8 | group2_inferred | 127.36 | 127.91 | 2.070 | 40,960 | 0.192 |
| wide_batch4 | 4 | 8 | group4_capture | 152.60 | 153.12 | 2.463 | 40,960 | 0.312 |
| wide_batch4 | 4 | 8 | group4_inferred | 126.94 | 127.51 | 2.053 | 40,960 | 0.344 |
| wide_batch4 | 4 | 8 | group8_capture | 152.46 | 153.00 | 2.455 | 40,960 | 0.617 |
| wide_batch4 | 4 | 8 | group8_inferred | 126.75 | 127.32 | 2.045 | 40,960 | 0.670 |

| Batch | Context | Sequential/inferred | Capture/inferred | Sequential/batched inferred | Heap without → with reuse |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1.35x | 1.17x | 1.38x | 14,080 → 3,072 |
| 3 | 3 | 1.42x | 1.14x | 1.51x | 17,152 → 6,144 |
| 1 | 512 | 1.08x | 1.05x | 1.09x | 45,824 → 18,944 |
| 1 | 8 | 1.26x | 1.22x | 1.27x | 47,360 → 10,240 |
| 1 | 8 | 1.43x | 1.20x | 1.44x | 54,016 → 3,072 |
| 1 | 8 | 1.07x | 1.04x | 1.08x | 14,080 → 3,072 |
| 64 | 64 | 1.16x | 1.09x | 1.18x | 1,017,088 → 294,912 |
| 4 | 8 | 1.34x | 1.17x | 1.38x | 50,432 → 11,264 |
| 4 | 8 | 1.24x | 1.20x | 1.26x | 185,600 → 40,960 |

Every timed sample and Python generation sequence passed exact token/history/position checks and numerical logits/KV-cache checks against an independent NumPy decoder. Prefilled cache, unused cache suffix, model weights, and allocation tail sentinels were checked for unintended writes. Raw JSON retains samples, topology, generated tokens, setup costs, source hashes, and maximum errors.

Fixed batches, uniform context lengths, contiguous MHA cache, greedy decoding, small random model, and unlocked GPU clocks. No serving arrival scheduler, continuous batching, paged cache, early EOS, or production-serving speedup is claimed. Batched graphs delay host observation until the group completes.

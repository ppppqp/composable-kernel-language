# CKL autoregressive decoder measurements

GPU: NVIDIA GeForce RTX 5060 Ti. Native build: Release. Python 3.12.13, NumPy 2.5.4.

All variants execute identical unfused FP32 device kernels. The model has random weights and is an orchestration experiment, not a trained language model or a competitive attention implementation. Prompt prefill, transfers, resets, compilation, and graph setup are outside steady decode timing.

Native stream/wall times are per complete decode step (one token per request) and include native submission gaps. Each sample averages a growing-context generation sequence, rather than repeatedly decoding the same token/position. Python generation latency covers all generated steps, with one completion wait per submission; batched variants submit all steps at once. That metric measures generation completion, not streamed token latency.

| Case | Batch | Initial context | Variant | Stream µs/step | Native wall µs/step | Python generation ms | Temp heap bytes | Setup ms |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| small | 1 | 8 | sequential | 62.86 | 63.23 | 1.090 | 14,080 | 0.000 |
| small | 1 | 8 | capture | 55.08 | 55.26 | 0.925 | 14,080 | 0.181 |
| small | 1 | 8 | inferred | 46.87 | 47.22 | 0.797 | 14,080 | 0.141 |
| small | 1 | 8 | sequential_reuse | 63.59 | 63.80 | 1.090 | 3,072 | 0.000 |
| small | 1 | 8 | capture_reuse | 55.19 | 55.37 | 0.928 | 3,072 | 0.127 |
| small | 1 | 8 | inferred_reuse | 47.10 | 47.29 | 0.799 | 3,072 | 0.087 |
| small | 1 | 8 | batched_capture | 53.78 | 53.96 | 0.860 | 3,072 | 2.506 |
| small | 1 | 8 | batched_inferred | 45.94 | 46.12 | 0.733 | 3,072 | 3.058 |
| small | 1 | 8 | group2_capture | 54.29 | 54.50 | 0.892 | 3,072 | 0.299 |
| small | 1 | 8 | group2_inferred | 46.13 | 46.32 | 0.764 | 3,072 | 0.359 |
| small | 1 | 8 | group4_capture | 53.98 | 54.17 | 0.874 | 3,072 | 0.766 |
| small | 1 | 8 | group4_inferred | 45.90 | 46.08 | 0.745 | 3,072 | 0.784 |
| small | 1 | 8 | group8_capture | 53.71 | 53.90 | 0.865 | 3,072 | 1.503 |
| small | 1 | 8 | group8_inferred | 45.76 | 46.24 | 0.738 | 3,072 | 1.561 |
| awkward | 3 | 3 | sequential | 35.15 | 35.64 | 0.622 | 17,152 | 0.000 |
| awkward | 3 | 3 | capture | 27.97 | 28.20 | 0.496 | 17,152 | 0.062 |
| awkward | 3 | 3 | inferred | 24.55 | 24.75 | 0.440 | 17,152 | 0.063 |
| awkward | 3 | 3 | sequential_reuse | 35.04 | 35.22 | 0.620 | 6,144 | 0.000 |
| awkward | 3 | 3 | capture_reuse | 28.13 | 28.33 | 0.497 | 6,144 | 0.049 |
| awkward | 3 | 3 | inferred_reuse | 24.55 | 24.74 | 0.441 | 6,144 | 0.059 |
| awkward | 3 | 3 | batched_capture | 26.49 | 26.68 | 0.428 | 6,144 | 0.627 |
| awkward | 3 | 3 | batched_inferred | 23.12 | 23.31 | 0.373 | 6,144 | 0.639 |
| awkward | 3 | 3 | group2_capture | 27.10 | 27.30 | 0.460 | 6,144 | 0.089 |
| awkward | 3 | 3 | group2_inferred | 23.90 | 24.09 | 0.404 | 6,144 | 0.103 |
| awkward | 3 | 3 | group4_capture | 26.81 | 27.04 | 0.441 | 6,144 | 0.151 |
| awkward | 3 | 3 | group4_inferred | 23.42 | 23.60 | 0.386 | 6,144 | 0.167 |
| awkward | 3 | 3 | group8_capture | 26.68 | 26.84 | 0.432 | 6,144 | 0.309 |
| awkward | 3 | 3 | group8_inferred | 23.14 | 23.49 | 0.377 | 6,144 | 0.337 |
| long_context | 1 | 512 | sequential | 177.95 | 178.23 | 2.931 | 45,824 | 0.000 |
| long_context | 1 | 512 | capture | 171.77 | 172.12 | 2.799 | 45,824 | 0.103 |
| long_context | 1 | 512 | inferred | 163.82 | 164.24 | 2.672 | 45,824 | 0.121 |
| long_context | 1 | 512 | sequential_reuse | 177.76 | 178.14 | 2.940 | 18,944 | 0.000 |
| long_context | 1 | 512 | capture_reuse | 172.14 | 172.50 | 2.807 | 18,944 | 0.106 |
| long_context | 1 | 512 | inferred_reuse | 164.03 | 164.41 | 2.678 | 18,944 | 0.110 |
| long_context | 1 | 512 | batched_capture | 170.60 | 171.16 | 2.736 | 18,944 | 1.180 |
| long_context | 1 | 512 | batched_inferred | 162.94 | 163.35 | 2.613 | 18,944 | 1.355 |
| long_context | 1 | 512 | group2_capture | 171.35 | 171.71 | 2.768 | 18,944 | 0.160 |
| long_context | 1 | 512 | group2_inferred | 163.35 | 163.74 | 2.639 | 18,944 | 0.170 |
| long_context | 1 | 512 | group4_capture | 170.90 | 171.25 | 2.749 | 18,944 | 0.301 |
| long_context | 1 | 512 | group4_inferred | 162.89 | 163.31 | 2.621 | 18,944 | 0.305 |
| long_context | 1 | 512 | group8_capture | 170.73 | 171.10 | 2.741 | 18,944 | 0.536 |
| long_context | 1 | 512 | group8_inferred | 162.77 | 163.14 | 2.613 | 18,944 | 0.584 |
| wide | 1 | 8 | sequential | 154.24 | 154.93 | 2.542 | 47,360 | 0.000 |
| wide | 1 | 8 | capture | 149.05 | 149.40 | 2.434 | 47,360 | 0.104 |
| wide | 1 | 8 | inferred | 122.55 | 122.80 | 2.010 | 47,360 | 0.152 |
| wide | 1 | 8 | sequential_reuse | 154.17 | 154.91 | 2.543 | 10,240 | 0.000 |
| wide | 1 | 8 | capture_reuse | 149.09 | 149.37 | 2.436 | 10,240 | 0.093 |
| wide | 1 | 8 | inferred_reuse | 122.50 | 122.84 | 2.011 | 10,240 | 0.103 |
| wide | 1 | 8 | batched_capture | 147.66 | 148.06 | 2.376 | 10,240 | 1.100 |
| wide | 1 | 8 | batched_inferred | 121.28 | 121.64 | 1.952 | 10,240 | 1.187 |
| wide | 1 | 8 | group2_capture | 148.07 | 148.36 | 2.398 | 10,240 | 0.187 |
| wide | 1 | 8 | group2_inferred | 121.43 | 121.72 | 1.976 | 10,240 | 0.178 |
| wide | 1 | 8 | group4_capture | 147.78 | 148.17 | 2.380 | 10,240 | 0.289 |
| wide | 1 | 8 | group4_inferred | 121.18 | 121.56 | 1.955 | 10,240 | 0.318 |
| wide | 1 | 8 | group8_capture | 147.63 | 148.15 | 2.372 | 10,240 | 0.544 |
| wide | 1 | 8 | group8_inferred | 121.18 | 121.63 | 1.956 | 10,240 | 0.596 |
| deep | 1 | 8 | sequential | 230.98 | 231.18 | 3.747 | 54,016 | 0.000 |
| deep | 1 | 8 | capture | 194.53 | 194.71 | 3.160 | 54,016 | 0.276 |
| deep | 1 | 8 | inferred | 162.40 | 162.61 | 2.645 | 54,016 | 0.343 |
| deep | 1 | 8 | sequential_reuse | 229.86 | 230.21 | 3.740 | 3,072 | 0.000 |
| deep | 1 | 8 | capture_reuse | 194.44 | 194.63 | 3.155 | 3,072 | 0.262 |
| deep | 1 | 8 | inferred_reuse | 162.18 | 162.36 | 2.645 | 3,072 | 0.289 |
| deep | 1 | 8 | batched_capture | 193.01 | 193.19 | 3.089 | 3,072 | 5.676 |
| deep | 1 | 8 | batched_inferred | 161.34 | 161.51 | 2.580 | 3,072 | 12.214 |
| deep | 1 | 8 | group2_capture | 193.55 | 193.73 | 3.120 | 3,072 | 1.113 |
| deep | 1 | 8 | group2_inferred | 161.75 | 161.94 | 2.611 | 3,072 | 1.174 |
| deep | 1 | 8 | group4_capture | 193.04 | 193.22 | 3.102 | 3,072 | 3.297 |
| deep | 1 | 8 | group4_inferred | 161.29 | 161.47 | 2.594 | 3,072 | 2.754 |
| deep | 1 | 8 | group8_capture | 192.79 | 192.97 | 3.093 | 3,072 | 5.416 |
| deep | 1 | 8 | group8_inferred | 161.17 | 161.39 | 2.584 | 3,072 | 6.343 |
| large_vocab | 1 | 8 | sequential | 215.90 | 216.53 | 3.572 | 14,080 | 0.000 |
| large_vocab | 1 | 8 | capture | 209.70 | 210.22 | 3.412 | 14,080 | 0.096 |
| large_vocab | 1 | 8 | inferred | 201.66 | 202.26 | 3.286 | 14,080 | 0.101 |
| large_vocab | 1 | 8 | sequential_reuse | 215.77 | 216.49 | 3.578 | 3,072 | 0.000 |
| large_vocab | 1 | 8 | capture_reuse | 209.69 | 210.32 | 3.414 | 3,072 | 0.090 |
| large_vocab | 1 | 8 | inferred_reuse | 201.91 | 202.53 | 3.287 | 3,072 | 0.102 |
| large_vocab | 1 | 8 | batched_capture | 208.31 | 208.94 | 3.345 | 3,072 | 1.065 |
| large_vocab | 1 | 8 | batched_inferred | 200.41 | 201.02 | 3.219 | 3,072 | 1.127 |
| large_vocab | 1 | 8 | group2_capture | 208.98 | 209.57 | 3.379 | 3,072 | 0.148 |
| large_vocab | 1 | 8 | group2_inferred | 201.21 | 201.78 | 3.250 | 3,072 | 0.158 |
| large_vocab | 1 | 8 | group4_capture | 208.58 | 209.21 | 3.359 | 3,072 | 0.271 |
| large_vocab | 1 | 8 | group4_inferred | 200.67 | 201.25 | 3.232 | 3,072 | 0.301 |
| large_vocab | 1 | 8 | group8_capture | 208.41 | 209.05 | 3.350 | 3,072 | 0.532 |
| large_vocab | 1 | 8 | group8_inferred | 200.56 | 201.13 | 3.224 | 3,072 | 0.580 |
| large_batch | 64 | 64 | sequential | 102.02 | 102.86 | 1.711 | 1,017,088 | 0.000 |
| large_batch | 64 | 64 | capture | 95.94 | 96.46 | 1.598 | 1,017,088 | 0.126 |
| large_batch | 64 | 64 | inferred | 88.22 | 88.73 | 1.471 | 1,017,088 | 0.171 |
| large_batch | 64 | 64 | sequential_reuse | 101.77 | 102.65 | 1.708 | 294,912 | 0.000 |
| large_batch | 64 | 64 | capture_reuse | 95.72 | 96.23 | 1.592 | 294,912 | 0.175 |
| large_batch | 64 | 64 | inferred_reuse | 87.66 | 88.18 | 1.465 | 294,912 | 0.154 |
| large_batch | 64 | 64 | batched_capture | 94.33 | 94.83 | 1.525 | 294,912 | 1.216 |
| large_batch | 64 | 64 | batched_inferred | 86.59 | 87.13 | 1.403 | 294,912 | 1.377 |
| large_batch | 64 | 64 | group2_capture | 94.75 | 95.26 | 1.556 | 294,912 | 0.228 |
| large_batch | 64 | 64 | group2_inferred | 87.03 | 87.57 | 1.433 | 294,912 | 0.238 |
| large_batch | 64 | 64 | group4_capture | 94.42 | 94.95 | 1.538 | 294,912 | 0.378 |
| large_batch | 64 | 64 | group4_inferred | 86.63 | 87.15 | 1.414 | 294,912 | 0.381 |
| large_batch | 64 | 64 | group8_capture | 94.29 | 94.80 | 1.529 | 294,912 | 0.628 |
| large_batch | 64 | 64 | group8_inferred | 86.55 | 87.06 | 1.404 | 294,912 | 0.696 |
| batch4 | 4 | 8 | sequential | 63.94 | 64.18 | 1.111 | 50,432 | 0.000 |
| batch4 | 4 | 8 | capture | 55.42 | 55.60 | 0.936 | 50,432 | 0.098 |
| batch4 | 4 | 8 | inferred | 47.38 | 47.61 | 0.809 | 50,432 | 0.092 |
| batch4 | 4 | 8 | sequential_reuse | 63.88 | 64.21 | 1.101 | 11,264 | 0.000 |
| batch4 | 4 | 8 | capture_reuse | 55.46 | 55.65 | 0.935 | 11,264 | 0.081 |
| batch4 | 4 | 8 | inferred_reuse | 47.38 | 47.55 | 0.808 | 11,264 | 0.094 |
| batch4 | 4 | 8 | batched_capture | 54.05 | 54.24 | 0.868 | 11,264 | 1.136 |
| batch4 | 4 | 8 | batched_inferred | 46.12 | 46.30 | 0.740 | 11,264 | 1.132 |
| batch4 | 4 | 8 | group2_capture | 54.87 | 55.05 | 0.899 | 11,264 | 0.145 |
| batch4 | 4 | 8 | group2_inferred | 46.67 | 46.85 | 0.771 | 11,264 | 0.163 |
| batch4 | 4 | 8 | group4_capture | 54.27 | 54.45 | 0.880 | 11,264 | 0.264 |
| batch4 | 4 | 8 | group4_inferred | 46.22 | 46.40 | 0.753 | 11,264 | 0.299 |
| batch4 | 4 | 8 | group8_capture | 54.10 | 54.32 | 0.872 | 11,264 | 0.531 |
| batch4 | 4 | 8 | group8_inferred | 46.16 | 46.41 | 0.745 | 11,264 | 0.566 |
| wide_batch4 | 4 | 8 | sequential | 160.16 | 160.86 | 2.638 | 185,600 | 0.000 |
| wide_batch4 | 4 | 8 | capture | 154.83 | 155.42 | 2.534 | 185,600 | 0.108 |
| wide_batch4 | 4 | 8 | inferred | 129.14 | 129.67 | 2.121 | 185,600 | 0.129 |
| wide_batch4 | 4 | 8 | sequential_reuse | 159.55 | 160.32 | 2.636 | 40,960 | 0.000 |
| wide_batch4 | 4 | 8 | capture_reuse | 154.44 | 155.03 | 2.527 | 40,960 | 0.104 |
| wide_batch4 | 4 | 8 | inferred_reuse | 128.60 | 129.12 | 2.114 | 40,960 | 0.110 |
| wide_batch4 | 4 | 8 | batched_capture | 152.88 | 153.48 | 2.460 | 40,960 | 1.218 |
| wide_batch4 | 4 | 8 | batched_inferred | 127.17 | 127.80 | 2.049 | 40,960 | 1.231 |
| wide_batch4 | 4 | 8 | group2_capture | 153.52 | 154.03 | 2.490 | 40,960 | 0.181 |
| wide_batch4 | 4 | 8 | group2_inferred | 127.57 | 128.18 | 2.078 | 40,960 | 0.184 |
| wide_batch4 | 4 | 8 | group4_capture | 153.10 | 153.66 | 2.472 | 40,960 | 0.306 |
| wide_batch4 | 4 | 8 | group4_inferred | 127.42 | 128.02 | 2.061 | 40,960 | 0.323 |
| wide_batch4 | 4 | 8 | group8_capture | 152.97 | 153.53 | 2.464 | 40,960 | 0.568 |
| wide_batch4 | 4 | 8 | group8_inferred | 127.22 | 127.79 | 2.052 | 40,960 | 0.626 |

| Batch | Context | Sequential/inferred | Capture/inferred | Sequential/batched inferred | Heap without → with reuse |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1.34x | 1.18x | 1.37x | 14,080 → 3,072 |
| 3 | 3 | 1.43x | 1.14x | 1.52x | 17,152 → 6,144 |
| 1 | 512 | 1.09x | 1.05x | 1.09x | 45,824 → 18,944 |
| 1 | 8 | 1.26x | 1.22x | 1.27x | 47,360 → 10,240 |
| 1 | 8 | 1.42x | 1.20x | 1.43x | 54,016 → 3,072 |
| 1 | 8 | 1.07x | 1.04x | 1.08x | 14,080 → 3,072 |
| 64 | 64 | 1.16x | 1.09x | 1.18x | 1,017,088 → 294,912 |
| 4 | 8 | 1.35x | 1.17x | 1.39x | 50,432 → 11,264 |
| 4 | 8 | 1.24x | 1.20x | 1.26x | 185,600 → 40,960 |

Every timed sample and Python generation sequence passed exact token/history/position checks and numerical logits/KV-cache checks against an independent NumPy decoder. Prefilled cache, unused cache suffix, model weights, and allocation tail sentinels were checked for unintended writes. Raw JSON retains samples, topology, generated tokens, setup costs, source hashes, and maximum errors.

Fixed batches, uniform context lengths, contiguous MHA cache, greedy decoding, small random model, and unlocked GPU clocks. No serving arrival scheduler, continuous batching, paged cache, early EOS, or production-serving speedup is claimed. Batched graphs delay host observation until the group completes.

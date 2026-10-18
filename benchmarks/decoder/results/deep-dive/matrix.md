# CKL autoregressive decoder measurements

GPU: NVIDIA GeForce RTX 5060 Ti. Native build: Release. Python 3.12.13, NumPy 2.5.1.

All variants execute identical unfused FP32 device kernels. The model has random weights and is an orchestration experiment, not a trained language model or a competitive attention implementation. Prompt prefill, transfers, resets, compilation, and graph setup are outside steady decode timing.

Native stream/wall times are per complete decode step (one token per request) and include native submission gaps. Each sample averages a growing-context generation sequence, rather than repeatedly decoding the same token/position. Python generation latency covers all generated steps, with one completion wait per submission; batched variants submit all steps at once. That metric measures generation completion, not streamed token latency.

| Case | Batch | Initial context | Variant | Stream µs/step | Native wall µs/step | Python generation ms | Temp heap bytes | Setup ms |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| small | 1 | 8 | sequential | 62.08 | 62.62 | 1.081 | 14,080 | 0.000 |
| small | 1 | 8 | capture | 54.82 | 55.00 | 0.930 | 14,080 | 0.089 |
| small | 1 | 8 | inferred | 46.71 | 46.91 | 0.802 | 14,080 | 0.071 |
| small | 1 | 8 | sequential_reuse | 62.98 | 63.20 | 1.082 | 3,072 | 0.000 |
| small | 1 | 8 | capture_reuse | 54.81 | 55.00 | 0.928 | 3,072 | 0.068 |
| small | 1 | 8 | inferred_reuse | 46.88 | 47.06 | 0.804 | 3,072 | 0.129 |
| small | 1 | 8 | batched_capture | 53.32 | 53.50 | 0.857 | 3,072 | 2.404 |
| small | 1 | 8 | batched_inferred | 45.40 | 45.58 | 0.731 | 3,072 | 2.887 |
| small | 1 | 8 | group2_capture | 53.90 | 54.08 | 0.891 | 3,072 | 0.294 |
| small | 1 | 8 | group2_inferred | 45.81 | 46.01 | 0.763 | 3,072 | 0.327 |
| small | 1 | 8 | group4_capture | 53.60 | 53.78 | 0.871 | 3,072 | 0.697 |
| small | 1 | 8 | group4_inferred | 45.60 | 45.78 | 0.744 | 3,072 | 0.740 |
| small | 1 | 8 | group8_capture | 53.40 | 53.59 | 0.861 | 3,072 | 1.322 |
| small | 1 | 8 | group8_inferred | 45.49 | 45.71 | 0.735 | 3,072 | 1.448 |
| awkward | 3 | 3 | sequential | 34.93 | 35.45 | 0.627 | 17,152 | 0.000 |
| awkward | 3 | 3 | capture | 28.01 | 28.20 | 0.503 | 17,152 | 0.064 |
| awkward | 3 | 3 | inferred | 24.58 | 24.78 | 0.444 | 17,152 | 0.062 |
| awkward | 3 | 3 | sequential_reuse | 35.93 | 36.13 | 0.629 | 6,144 | 0.000 |
| awkward | 3 | 3 | capture_reuse | 28.00 | 28.20 | 0.501 | 6,144 | 0.050 |
| awkward | 3 | 3 | inferred_reuse | 24.46 | 24.64 | 0.445 | 6,144 | 0.056 |
| awkward | 3 | 3 | batched_capture | 26.48 | 26.67 | 0.428 | 6,144 | 0.646 |
| awkward | 3 | 3 | batched_inferred | 23.06 | 23.24 | 0.373 | 6,144 | 0.698 |
| awkward | 3 | 3 | group2_capture | 27.14 | 27.32 | 0.462 | 6,144 | 0.090 |
| awkward | 3 | 3 | group2_inferred | 23.92 | 24.10 | 0.406 | 6,144 | 0.096 |
| awkward | 3 | 3 | group4_capture | 26.74 | 26.92 | 0.443 | 6,144 | 0.153 |
| awkward | 3 | 3 | group4_inferred | 23.42 | 23.61 | 0.387 | 6,144 | 0.173 |
| awkward | 3 | 3 | group8_capture | 26.61 | 26.79 | 0.433 | 6,144 | 0.310 |
| awkward | 3 | 3 | group8_inferred | 23.16 | 23.37 | 0.377 | 6,144 | 0.327 |
| long_context | 1 | 512 | sequential | 177.08 | 177.33 | 2.945 | 45,824 | 0.000 |
| long_context | 1 | 512 | capture | 171.64 | 171.95 | 2.803 | 45,824 | 0.104 |
| long_context | 1 | 512 | inferred | 163.60 | 163.84 | 2.674 | 45,824 | 0.109 |
| long_context | 1 | 512 | sequential_reuse | 176.67 | 176.95 | 2.930 | 18,944 | 0.000 |
| long_context | 1 | 512 | capture_reuse | 171.41 | 171.65 | 2.801 | 18,944 | 0.083 |
| long_context | 1 | 512 | inferred_reuse | 163.41 | 163.65 | 2.675 | 18,944 | 0.094 |
| long_context | 1 | 512 | batched_capture | 169.81 | 170.34 | 2.732 | 18,944 | 1.351 |
| long_context | 1 | 512 | batched_inferred | 161.91 | 162.27 | 2.602 | 18,944 | 1.433 |
| long_context | 1 | 512 | group2_capture | 170.61 | 170.84 | 2.758 | 18,944 | 0.152 |
| long_context | 1 | 512 | group2_inferred | 162.41 | 162.65 | 2.631 | 18,944 | 0.162 |
| long_context | 1 | 512 | group4_capture | 170.13 | 170.37 | 2.739 | 18,944 | 0.255 |
| long_context | 1 | 512 | group4_inferred | 162.16 | 162.44 | 2.613 | 18,944 | 0.319 |
| long_context | 1 | 512 | group8_capture | 169.86 | 170.07 | 2.729 | 18,944 | 0.541 |
| long_context | 1 | 512 | group8_inferred | 162.11 | 162.32 | 2.603 | 18,944 | 0.578 |
| wide | 1 | 8 | sequential | 154.06 | 154.86 | 2.552 | 47,360 | 0.000 |
| wide | 1 | 8 | capture | 149.06 | 149.55 | 2.449 | 47,360 | 0.106 |
| wide | 1 | 8 | inferred | 122.47 | 123.18 | 2.022 | 47,360 | 0.153 |
| wide | 1 | 8 | sequential_reuse | 153.97 | 154.86 | 2.548 | 10,240 | 0.000 |
| wide | 1 | 8 | capture_reuse | 149.03 | 149.57 | 2.446 | 10,240 | 0.096 |
| wide | 1 | 8 | inferred_reuse | 122.46 | 123.07 | 2.018 | 10,240 | 0.105 |
| wide | 1 | 8 | batched_capture | 147.37 | 148.09 | 2.373 | 10,240 | 1.112 |
| wide | 1 | 8 | batched_inferred | 120.85 | 121.48 | 1.949 | 10,240 | 1.188 |
| wide | 1 | 8 | group2_capture | 148.04 | 148.77 | 2.408 | 10,240 | 0.179 |
| wide | 1 | 8 | group2_inferred | 121.44 | 122.14 | 1.982 | 10,240 | 0.184 |
| wide | 1 | 8 | group4_capture | 147.72 | 148.38 | 2.388 | 10,240 | 0.282 |
| wide | 1 | 8 | group4_inferred | 121.09 | 121.79 | 1.963 | 10,240 | 0.322 |
| wide | 1 | 8 | group8_capture | 147.50 | 148.03 | 2.379 | 10,240 | 0.534 |
| wide | 1 | 8 | group8_inferred | 120.97 | 121.64 | 1.955 | 10,240 | 0.585 |
| deep | 1 | 8 | sequential | 230.72 | 230.93 | 3.754 | 54,016 | 0.000 |
| deep | 1 | 8 | capture | 194.67 | 194.85 | 3.171 | 54,016 | 0.276 |
| deep | 1 | 8 | inferred | 162.48 | 162.70 | 2.655 | 54,016 | 0.357 |
| deep | 1 | 8 | sequential_reuse | 229.95 | 230.16 | 3.736 | 3,072 | 0.000 |
| deep | 1 | 8 | capture_reuse | 195.06 | 195.24 | 3.175 | 3,072 | 0.257 |
| deep | 1 | 8 | inferred_reuse | 162.75 | 162.94 | 2.660 | 3,072 | 0.289 |
| deep | 1 | 8 | batched_capture | 193.72 | 193.90 | 3.101 | 3,072 | 5.355 |
| deep | 1 | 8 | batched_inferred | 161.76 | 161.94 | 2.591 | 3,072 | 11.159 |
| deep | 1 | 8 | group2_capture | 194.22 | 194.41 | 3.137 | 3,072 | 1.052 |
| deep | 1 | 8 | group2_inferred | 162.23 | 162.41 | 2.624 | 3,072 | 1.078 |
| deep | 1 | 8 | group4_capture | 193.83 | 194.03 | 3.116 | 3,072 | 2.889 |
| deep | 1 | 8 | group4_inferred | 161.78 | 161.96 | 2.604 | 3,072 | 2.445 |
| deep | 1 | 8 | group8_capture | 193.67 | 193.86 | 3.106 | 3,072 | 4.772 |
| deep | 1 | 8 | group8_inferred | 161.69 | 161.90 | 2.596 | 3,072 | 5.737 |
| large_vocab | 1 | 8 | sequential | 215.23 | 215.89 | 3.577 | 14,080 | 0.000 |
| large_vocab | 1 | 8 | capture | 209.68 | 210.21 | 3.418 | 14,080 | 0.124 |
| large_vocab | 1 | 8 | inferred | 201.58 | 202.12 | 3.287 | 14,080 | 0.117 |
| large_vocab | 1 | 8 | sequential_reuse | 215.22 | 215.91 | 3.577 | 3,072 | 0.000 |
| large_vocab | 1 | 8 | capture_reuse | 209.70 | 210.21 | 3.416 | 3,072 | 0.093 |
| large_vocab | 1 | 8 | inferred_reuse | 201.49 | 202.03 | 3.287 | 3,072 | 0.100 |
| large_vocab | 1 | 8 | batched_capture | 208.10 | 208.67 | 3.343 | 3,072 | 1.091 |
| large_vocab | 1 | 8 | batched_inferred | 200.22 | 200.73 | 3.216 | 3,072 | 1.153 |
| large_vocab | 1 | 8 | group2_capture | 208.77 | 209.31 | 3.377 | 3,072 | 0.154 |
| large_vocab | 1 | 8 | group2_inferred | 200.83 | 201.38 | 3.249 | 3,072 | 0.173 |
| large_vocab | 1 | 8 | group4_capture | 208.53 | 209.06 | 3.358 | 3,072 | 0.281 |
| large_vocab | 1 | 8 | group4_inferred | 200.37 | 200.90 | 3.229 | 3,072 | 0.295 |
| large_vocab | 1 | 8 | group8_capture | 208.28 | 208.80 | 3.348 | 3,072 | 0.545 |
| large_vocab | 1 | 8 | group8_inferred | 200.32 | 200.87 | 3.221 | 3,072 | 0.584 |
| large_batch | 64 | 64 | sequential | 101.79 | 102.76 | 1.712 | 1,017,088 | 0.000 |
| large_batch | 64 | 64 | capture | 95.83 | 96.34 | 1.599 | 1,017,088 | 0.123 |
| large_batch | 64 | 64 | inferred | 88.17 | 88.64 | 1.474 | 1,017,088 | 0.172 |
| large_batch | 64 | 64 | sequential_reuse | 101.63 | 102.46 | 1.707 | 294,912 | 0.000 |
| large_batch | 64 | 64 | capture_reuse | 95.50 | 96.00 | 1.595 | 294,912 | 0.185 |
| large_batch | 64 | 64 | inferred_reuse | 87.65 | 88.17 | 1.466 | 294,912 | 0.168 |
| large_batch | 64 | 64 | batched_capture | 94.12 | 94.64 | 1.523 | 294,912 | 1.282 |
| large_batch | 64 | 64 | batched_inferred | 86.47 | 86.98 | 1.399 | 294,912 | 1.357 |
| large_batch | 64 | 64 | group2_capture | 94.59 | 95.12 | 1.557 | 294,912 | 0.239 |
| large_batch | 64 | 64 | group2_inferred | 86.91 | 87.37 | 1.431 | 294,912 | 0.251 |
| large_batch | 64 | 64 | group4_capture | 94.42 | 94.92 | 1.536 | 294,912 | 0.380 |
| large_batch | 64 | 64 | group4_inferred | 86.42 | 86.96 | 1.412 | 294,912 | 0.396 |
| large_batch | 64 | 64 | group8_capture | 94.22 | 94.73 | 1.528 | 294,912 | 0.650 |
| large_batch | 64 | 64 | group8_inferred | 86.38 | 86.91 | 1.404 | 294,912 | 0.684 |
| batch4 | 4 | 8 | sequential | 63.67 | 64.05 | 1.111 | 50,432 | 0.000 |
| batch4 | 4 | 8 | capture | 55.31 | 55.50 | 0.941 | 50,432 | 0.094 |
| batch4 | 4 | 8 | inferred | 47.47 | 47.65 | 0.814 | 50,432 | 0.094 |
| batch4 | 4 | 8 | sequential_reuse | 63.44 | 63.67 | 1.112 | 11,264 | 0.000 |
| batch4 | 4 | 8 | capture_reuse | 55.57 | 55.76 | 0.942 | 11,264 | 0.081 |
| batch4 | 4 | 8 | inferred_reuse | 47.38 | 47.56 | 0.814 | 11,264 | 0.092 |
| batch4 | 4 | 8 | batched_capture | 54.11 | 54.28 | 0.869 | 11,264 | 1.225 |
| batch4 | 4 | 8 | batched_inferred | 46.14 | 46.32 | 0.741 | 11,264 | 1.226 |
| batch4 | 4 | 8 | group2_capture | 54.83 | 55.01 | 0.903 | 11,264 | 0.150 |
| batch4 | 4 | 8 | group2_inferred | 46.73 | 46.91 | 0.775 | 11,264 | 0.165 |
| batch4 | 4 | 8 | group4_capture | 54.41 | 54.59 | 0.884 | 11,264 | 0.283 |
| batch4 | 4 | 8 | group4_inferred | 46.23 | 46.41 | 0.755 | 11,264 | 0.315 |
| batch4 | 4 | 8 | group8_capture | 54.16 | 54.34 | 0.874 | 11,264 | 0.542 |
| batch4 | 4 | 8 | group8_inferred | 46.26 | 46.45 | 0.747 | 11,264 | 0.594 |
| wide_batch4 | 4 | 8 | sequential | 159.40 | 160.10 | 2.632 | 185,600 | 0.000 |
| wide_batch4 | 4 | 8 | capture | 154.15 | 154.67 | 2.526 | 185,600 | 0.106 |
| wide_batch4 | 4 | 8 | inferred | 128.48 | 128.97 | 2.116 | 185,600 | 0.125 |
| wide_batch4 | 4 | 8 | sequential_reuse | 159.04 | 159.73 | 2.628 | 40,960 | 0.000 |
| wide_batch4 | 4 | 8 | capture_reuse | 153.76 | 154.27 | 2.524 | 40,960 | 0.102 |
| wide_batch4 | 4 | 8 | inferred_reuse | 128.11 | 128.61 | 2.113 | 40,960 | 0.117 |
| wide_batch4 | 4 | 8 | batched_capture | 152.36 | 152.88 | 2.451 | 40,960 | 1.204 |
| wide_batch4 | 4 | 8 | batched_inferred | 126.72 | 127.25 | 2.042 | 40,960 | 1.290 |
| wide_batch4 | 4 | 8 | group2_capture | 153.00 | 153.53 | 2.484 | 40,960 | 0.178 |
| wide_batch4 | 4 | 8 | group2_inferred | 127.46 | 127.95 | 2.072 | 40,960 | 0.193 |
| wide_batch4 | 4 | 8 | group4_capture | 152.65 | 153.15 | 2.465 | 40,960 | 0.298 |
| wide_batch4 | 4 | 8 | group4_inferred | 126.95 | 127.45 | 2.054 | 40,960 | 0.327 |
| wide_batch4 | 4 | 8 | group8_capture | 152.38 | 152.88 | 2.456 | 40,960 | 0.589 |
| wide_batch4 | 4 | 8 | group8_inferred | 126.71 | 127.21 | 2.045 | 40,960 | 0.610 |

| Batch | Context | Sequential/inferred | Capture/inferred | Sequential/batched inferred | Heap without → with reuse |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1.33x | 1.17x | 1.37x | 14,080 → 3,072 |
| 3 | 3 | 1.42x | 1.14x | 1.51x | 17,152 → 6,144 |
| 1 | 512 | 1.08x | 1.05x | 1.09x | 45,824 → 18,944 |
| 1 | 8 | 1.26x | 1.22x | 1.27x | 47,360 → 10,240 |
| 1 | 8 | 1.42x | 1.20x | 1.43x | 54,016 → 3,072 |
| 1 | 8 | 1.07x | 1.04x | 1.07x | 14,080 → 3,072 |
| 64 | 64 | 1.15x | 1.09x | 1.18x | 1,017,088 → 294,912 |
| 4 | 8 | 1.34x | 1.17x | 1.38x | 50,432 → 11,264 |
| 4 | 8 | 1.24x | 1.20x | 1.26x | 185,600 → 40,960 |

Every timed sample and Python generation sequence passed exact token/history/position checks and numerical logits/KV-cache checks against an independent NumPy decoder. Prefilled cache, unused cache suffix, model weights, and allocation tail sentinels were checked for unintended writes. Raw JSON retains samples, topology, generated tokens, setup costs, source hashes, and maximum errors.

Fixed batches, uniform context lengths, contiguous MHA cache, greedy decoding, small random model, and unlocked GPU clocks. No serving arrival scheduler, continuous batching, paged cache, early EOS, or production-serving speedup is claimed. Batched graphs delay host observation until the group completes.

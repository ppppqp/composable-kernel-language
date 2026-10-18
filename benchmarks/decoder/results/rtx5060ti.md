# CKL autoregressive decoder measurements

GPU: NVIDIA GeForce RTX 5060 Ti. Native build: Release. Python 3.12.13, NumPy 2.5.1.

All variants execute identical unfused FP32 device kernels. The model has random weights and is an orchestration experiment, not a trained language model or a competitive attention implementation. Prompt prefill, transfers, resets, compilation, and graph setup are outside steady decode timing.

Native stream/wall times are per complete decode step (one token per request) and include native submission gaps. Each sample averages a growing-context generation sequence, rather than repeatedly decoding the same token/position. Python generation latency covers all generated steps, with one completion wait per submission; batched variants submit all steps at once. That metric measures generation completion, not streamed token latency.

| Batch | Initial context | Variant | Stream µs/step | Native wall µs/step | Python generation ms | Temp heap bytes | Setup ms |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 1 | sequential | 62.14 | 62.32 | 1.078 | 14,080 | 0.000 |
| 1 | 1 | capture | 53.24 | 53.43 | 0.905 | 14,080 | 0.085 |
| 1 | 1 | inferred | 45.20 | 45.40 | 0.777 | 14,080 | 0.103 |
| 1 | 1 | inferred_reuse | 45.34 | 45.55 | 0.779 | 3,072 | 0.141 |
| 1 | 1 | batched_capture | 51.90 | 52.08 | 0.834 | 3,072 | 2.333 |
| 1 | 1 | batched_inferred | 43.94 | 44.13 | 0.707 | 3,072 | 2.808 |
| 1 | 8 | sequential | 62.82 | 63.06 | 1.087 | 14,080 | 0.000 |
| 1 | 8 | capture | 54.87 | 55.05 | 0.929 | 14,080 | 0.093 |
| 1 | 8 | inferred | 46.75 | 46.93 | 0.803 | 14,080 | 0.093 |
| 1 | 8 | inferred_reuse | 46.69 | 46.92 | 0.803 | 3,072 | 0.094 |
| 1 | 8 | batched_capture | 53.25 | 53.44 | 0.856 | 3,072 | 1.086 |
| 1 | 8 | batched_inferred | 45.36 | 45.65 | 0.730 | 3,072 | 1.293 |
| 8 | 8 | sequential | 63.30 | 63.74 | 1.099 | 99,584 | 0.000 |
| 8 | 8 | capture | 56.04 | 56.26 | 0.951 | 99,584 | 0.097 |
| 8 | 8 | inferred | 48.28 | 48.46 | 0.826 | 99,584 | 0.092 |
| 8 | 8 | inferred_reuse | 48.25 | 48.45 | 0.826 | 22,528 | 0.093 |
| 8 | 8 | batched_capture | 54.58 | 54.83 | 0.878 | 22,528 | 1.097 |
| 8 | 8 | batched_inferred | 46.75 | 46.97 | 0.753 | 22,528 | 1.446 |
| 1 | 64 | sequential | 73.25 | 73.51 | 1.274 | 17,152 | 0.000 |
| 1 | 64 | capture | 67.83 | 68.01 | 1.140 | 17,152 | 0.092 |
| 1 | 64 | inferred | 59.81 | 60.01 | 1.013 | 17,152 | 0.090 |
| 1 | 64 | inferred_reuse | 59.55 | 59.76 | 1.006 | 4,608 | 0.096 |
| 1 | 64 | batched_capture | 66.11 | 66.29 | 1.062 | 4,608 | 1.066 |
| 1 | 64 | batched_inferred | 58.26 | 58.47 | 0.936 | 4,608 | 1.195 |
| 8 | 64 | sequential | 76.48 | 77.54 | 1.327 | 128,256 | 0.000 |
| 8 | 64 | capture | 71.39 | 72.12 | 1.206 | 128,256 | 0.095 |
| 8 | 64 | inferred | 63.49 | 64.09 | 1.080 | 128,256 | 0.109 |
| 8 | 64 | inferred_reuse | 63.50 | 64.03 | 1.081 | 36,864 | 0.097 |
| 8 | 64 | batched_capture | 69.90 | 70.49 | 1.134 | 36,864 | 1.101 |
| 8 | 64 | batched_inferred | 62.09 | 62.66 | 1.005 | 36,864 | 1.176 |
| 32 | 64 | sequential | 96.08 | 96.82 | 1.622 | 509,184 | 0.000 |
| 32 | 64 | capture | 90.84 | 91.38 | 1.513 | 509,184 | 0.109 |
| 32 | 64 | inferred | 82.67 | 83.23 | 1.386 | 509,184 | 0.110 |
| 32 | 64 | inferred_reuse | 83.06 | 83.55 | 1.390 | 147,456 | 0.120 |
| 32 | 64 | batched_capture | 89.34 | 89.87 | 1.443 | 147,456 | 1.106 |
| 32 | 64 | batched_inferred | 81.55 | 82.07 | 1.318 | 147,456 | 1.171 |

| Batch | Context | Sequential/inferred | Capture/inferred | Sequential/batched inferred | Heap without → with reuse |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.37x | 1.18x | 1.41x | 14,080 → 3,072 |
| 1 | 8 | 1.34x | 1.17x | 1.38x | 14,080 → 3,072 |
| 8 | 8 | 1.31x | 1.16x | 1.35x | 99,584 → 22,528 |
| 1 | 64 | 1.22x | 1.13x | 1.26x | 17,152 → 4,608 |
| 8 | 64 | 1.20x | 1.12x | 1.23x | 128,256 → 36,864 |
| 32 | 64 | 1.16x | 1.10x | 1.18x | 509,184 → 147,456 |

Every timed sample and Python generation sequence passed exact token/history/position checks and numerical logits/KV-cache checks against an independent NumPy decoder. Prefilled cache, unused cache suffix, model weights, and allocation tail sentinels were checked for unintended writes. Raw JSON retains samples, topology, generated tokens, setup costs, source hashes, and maximum errors.

Fixed batches, uniform context lengths, contiguous MHA cache, greedy decoding, small random model, and unlocked GPU clocks. No serving arrival scheduler, continuous batching, paged cache, early EOS, or production-serving speedup is claimed. Batched graphs delay host observation until the group completes.

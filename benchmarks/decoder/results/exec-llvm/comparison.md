# Decoder host compilation migration comparison

Positive changes mean slower execution. Comparisons use per-variant medians; they are descriptive measurements, not statistical equivalence tests.

All 126 case/variant pairs passed both runners' reference checks. Prompts, generated tokens, dependency topologies, dispatch counts, temporary counts, and temporary heap bytes match exactly.

| Metric | Minimum change | Median change | Maximum change |
|---|---:|---:|---:|
| stream_us | -0.85% | +0.13% | +0.92% |
| wall_us | -0.85% | +0.13% | +0.95% |
| generation_ms | -1.20% | +0.09% | +0.57% |

## Frontend lowering setup

Lowering is excluded from steady decode timing.

| Case | Before ms | After ms |
|---|---:|---:|
| small | 21.95 | 20.46 |
| awkward | 12.49 | 13.18 |
| long_context | 17.80 | 17.50 |
| wide | 24.40 | 47.77 |
| deep | 40.40 | 41.17 |
| large_vocab | 22.59 | 22.29 |
| large_batch | 17.38 | 17.37 |
| batch4 | 16.72 | 16.93 |
| wide_batch4 | 17.04 | 21.59 |

## Per-variant medians

| Case | Variant | Before stream µs/step | After stream µs/step | Stream change | Generation change |
|---|---|---:|---:|---:|---:|
| small | sequential | 62.89 | 62.86 | -0.05% | +0.32% |
| small | capture | 54.80 | 55.08 | +0.50% | -0.00% |
| small | inferred | 46.75 | 46.87 | +0.26% | -0.04% |
| small | sequential_reuse | 63.01 | 63.59 | +0.92% | +0.57% |
| small | capture_reuse | 55.04 | 55.19 | +0.28% | +0.01% |
| small | inferred_reuse | 46.99 | 47.10 | +0.24% | -0.16% |
| small | batched_capture | 53.47 | 53.78 | +0.58% | +0.09% |
| small | batched_inferred | 45.62 | 45.94 | +0.72% | +0.08% |
| small | group2_capture | 54.03 | 54.29 | +0.48% | +0.03% |
| small | group2_inferred | 46.08 | 46.13 | +0.11% | +0.15% |
| small | group4_capture | 53.77 | 53.98 | +0.39% | +0.08% |
| small | group4_inferred | 45.74 | 45.90 | +0.35% | +0.08% |
| small | group8_capture | 53.64 | 53.71 | +0.13% | +0.10% |
| small | group8_inferred | 45.68 | 45.76 | +0.18% | +0.03% |
| awkward | sequential | 34.90 | 35.15 | +0.71% | +0.47% |
| awkward | capture | 28.00 | 27.97 | -0.11% | -0.13% |
| awkward | inferred | 24.56 | 24.55 | -0.06% | +0.13% |
| awkward | sequential_reuse | 35.08 | 35.04 | -0.11% | -0.88% |
| awkward | capture_reuse | 28.09 | 28.13 | +0.14% | +0.18% |
| awkward | inferred_reuse | 24.59 | 24.55 | -0.15% | +0.30% |
| awkward | batched_capture | 26.47 | 26.49 | +0.09% | +0.09% |
| awkward | batched_inferred | 23.06 | 23.12 | +0.29% | -0.02% |
| awkward | group2_capture | 27.15 | 27.10 | -0.17% | -0.34% |
| awkward | group2_inferred | 23.89 | 23.90 | +0.04% | +0.08% |
| awkward | group4_capture | 26.78 | 26.81 | +0.10% | +0.06% |
| awkward | group4_inferred | 23.40 | 23.42 | +0.08% | +0.05% |
| awkward | group8_capture | 26.64 | 26.68 | +0.14% | +0.01% |
| awkward | group8_inferred | 23.13 | 23.14 | +0.08% | +0.07% |
| long_context | sequential | 176.88 | 177.95 | +0.60% | +0.14% |
| long_context | capture | 171.69 | 171.77 | +0.05% | +0.08% |
| long_context | inferred | 163.58 | 163.82 | +0.15% | +0.11% |
| long_context | sequential_reuse | 177.18 | 177.76 | +0.33% | +0.27% |
| long_context | capture_reuse | 171.91 | 172.14 | +0.13% | +0.15% |
| long_context | inferred_reuse | 163.82 | 164.03 | +0.13% | +0.18% |
| long_context | batched_capture | 170.43 | 170.60 | +0.10% | +0.11% |
| long_context | batched_inferred | 162.50 | 162.94 | +0.27% | +0.31% |
| long_context | group2_capture | 171.19 | 171.35 | +0.09% | +0.08% |
| long_context | group2_inferred | 163.15 | 163.35 | +0.12% | +0.04% |
| long_context | group4_capture | 170.72 | 170.90 | +0.10% | +0.06% |
| long_context | group4_inferred | 162.67 | 162.89 | +0.14% | +0.07% |
| long_context | group8_capture | 170.48 | 170.73 | +0.14% | +0.07% |
| long_context | group8_inferred | 162.51 | 162.77 | +0.16% | +0.09% |
| wide | sequential | 154.20 | 154.24 | +0.02% | -0.11% |
| wide | capture | 148.97 | 149.05 | +0.06% | +0.04% |
| wide | inferred | 122.42 | 122.55 | +0.11% | -0.12% |
| wide | sequential_reuse | 154.61 | 154.17 | -0.28% | +0.13% |
| wide | capture_reuse | 148.98 | 149.09 | +0.07% | -0.04% |
| wide | inferred_reuse | 122.30 | 122.50 | +0.17% | +0.03% |
| wide | batched_capture | 147.60 | 147.66 | +0.04% | +0.07% |
| wide | batched_inferred | 121.28 | 121.28 | +0.00% | +0.10% |
| wide | group2_capture | 148.16 | 148.07 | -0.06% | +0.01% |
| wide | group2_inferred | 121.54 | 121.43 | -0.09% | +0.04% |
| wide | group4_capture | 147.71 | 147.78 | +0.04% | +0.06% |
| wide | group4_inferred | 121.24 | 121.18 | -0.05% | -0.16% |
| wide | group8_capture | 147.56 | 147.63 | +0.05% | +0.05% |
| wide | group8_inferred | 121.16 | 121.18 | +0.02% | +0.08% |
| deep | sequential | 231.84 | 230.98 | -0.37% | -0.25% |
| deep | capture | 194.32 | 194.53 | +0.10% | -0.01% |
| deep | inferred | 162.16 | 162.40 | +0.15% | +0.09% |
| deep | sequential_reuse | 231.83 | 229.86 | -0.85% | -0.35% |
| deep | capture_reuse | 194.14 | 194.44 | +0.16% | +0.04% |
| deep | inferred_reuse | 162.11 | 162.18 | +0.04% | +0.09% |
| deep | batched_capture | 192.71 | 193.01 | +0.15% | +0.12% |
| deep | batched_inferred | 161.04 | 161.34 | +0.18% | +0.06% |
| deep | group2_capture | 193.12 | 193.55 | +0.22% | +0.06% |
| deep | group2_inferred | 161.46 | 161.75 | +0.18% | +0.02% |
| deep | group4_capture | 192.86 | 193.04 | +0.09% | +0.08% |
| deep | group4_inferred | 161.21 | 161.29 | +0.05% | +0.07% |
| deep | group8_capture | 192.65 | 192.79 | +0.07% | +0.08% |
| deep | group8_inferred | 161.03 | 161.17 | +0.09% | +0.10% |
| large_vocab | sequential | 215.51 | 215.90 | +0.18% | -0.14% |
| large_vocab | capture | 209.50 | 209.70 | +0.10% | +0.12% |
| large_vocab | inferred | 201.45 | 201.66 | +0.10% | +0.19% |
| large_vocab | sequential_reuse | 215.35 | 215.77 | +0.20% | +0.31% |
| large_vocab | capture_reuse | 209.57 | 209.69 | +0.05% | +0.12% |
| large_vocab | inferred_reuse | 201.65 | 201.91 | +0.13% | +0.11% |
| large_vocab | batched_capture | 208.05 | 208.31 | +0.13% | +0.11% |
| large_vocab | batched_inferred | 200.24 | 200.41 | +0.08% | +0.10% |
| large_vocab | group2_capture | 208.71 | 208.98 | +0.13% | +0.19% |
| large_vocab | group2_inferred | 200.92 | 201.21 | +0.14% | +0.11% |
| large_vocab | group4_capture | 208.43 | 208.58 | +0.07% | +0.10% |
| large_vocab | group4_inferred | 200.38 | 200.67 | +0.15% | +0.14% |
| large_vocab | group8_capture | 208.19 | 208.41 | +0.10% | +0.13% |
| large_vocab | group8_inferred | 200.32 | 200.56 | +0.12% | +0.12% |
| large_batch | sequential | 102.01 | 102.02 | +0.01% | -0.14% |
| large_batch | capture | 95.83 | 95.94 | +0.11% | +0.13% |
| large_batch | inferred | 88.07 | 88.22 | +0.17% | -0.12% |
| large_batch | sequential_reuse | 101.60 | 101.77 | +0.17% | +0.24% |
| large_batch | capture_reuse | 95.54 | 95.72 | +0.19% | +0.18% |
| large_batch | inferred_reuse | 87.62 | 87.66 | +0.04% | +0.07% |
| large_batch | batched_capture | 94.22 | 94.33 | +0.11% | +0.10% |
| large_batch | batched_inferred | 86.51 | 86.59 | +0.09% | +0.07% |
| large_batch | group2_capture | 94.76 | 94.75 | -0.01% | +0.14% |
| large_batch | group2_inferred | 87.01 | 87.03 | +0.02% | +0.12% |
| large_batch | group4_capture | 94.40 | 94.42 | +0.03% | +0.09% |
| large_batch | group4_inferred | 86.51 | 86.63 | +0.14% | +0.06% |
| large_batch | group8_capture | 94.20 | 94.29 | +0.09% | -0.03% |
| large_batch | group8_inferred | 86.43 | 86.55 | +0.14% | +0.08% |
| batch4 | sequential | 63.65 | 63.94 | +0.45% | -1.20% |
| batch4 | capture | 55.33 | 55.42 | +0.15% | +0.10% |
| batch4 | inferred | 47.37 | 47.38 | +0.03% | -0.04% |
| batch4 | sequential_reuse | 63.77 | 63.88 | +0.17% | -0.32% |
| batch4 | capture_reuse | 55.44 | 55.46 | +0.05% | +0.10% |
| batch4 | inferred_reuse | 47.38 | 47.38 | -0.02% | +0.14% |
| batch4 | batched_capture | 53.93 | 54.05 | +0.24% | +0.16% |
| batch4 | batched_inferred | 46.01 | 46.12 | +0.23% | +0.11% |
| batch4 | group2_capture | 54.79 | 54.87 | +0.14% | +0.05% |
| batch4 | group2_inferred | 46.73 | 46.67 | -0.12% | +0.11% |
| batch4 | group4_capture | 54.26 | 54.27 | +0.03% | +0.02% |
| batch4 | group4_inferred | 46.24 | 46.22 | -0.04% | +0.18% |
| batch4 | group8_capture | 54.08 | 54.10 | +0.05% | +0.09% |
| batch4 | group8_inferred | 46.09 | 46.16 | +0.16% | +0.07% |
| wide_batch4 | sequential | 159.53 | 160.16 | +0.39% | +0.34% |
| wide_batch4 | capture | 154.24 | 154.83 | +0.38% | +0.38% |
| wide_batch4 | inferred | 128.63 | 129.14 | +0.39% | +0.37% |
| wide_batch4 | sequential_reuse | 159.09 | 159.55 | +0.29% | +0.53% |
| wide_batch4 | capture_reuse | 153.70 | 154.44 | +0.48% | +0.39% |
| wide_batch4 | inferred_reuse | 128.09 | 128.60 | +0.40% | +0.40% |
| wide_batch4 | batched_capture | 152.37 | 152.88 | +0.34% | +0.39% |
| wide_batch4 | batched_inferred | 126.95 | 127.17 | +0.18% | +0.31% |
| wide_batch4 | group2_capture | 153.08 | 153.52 | +0.29% | +0.32% |
| wide_batch4 | group2_inferred | 127.36 | 127.57 | +0.17% | +0.38% |
| wide_batch4 | group4_capture | 152.60 | 153.10 | +0.33% | +0.37% |
| wide_batch4 | group4_inferred | 126.94 | 127.42 | +0.38% | +0.41% |
| wide_batch4 | group8_capture | 152.46 | 152.97 | +0.34% | +0.35% |
| wide_batch4 | group8_inferred | 126.75 | 127.22 | +0.37% | +0.37% |

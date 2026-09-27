# Milestone 0 CUDA feasibility benchmarks

This directory tests whether CKL's proposed orchestration layer has measurable work to do before
the compiler and runtime are built. It contains two deliberately different repeated programs:

- `multi_field`: two independent, deliberately underfilled compute chains joined by diagnostics.
  It models narrow per-field physics or solver updates and exposes branch concurrency that
  single-stream capture hides. The default 2,048 elements are part of this use case; scaling it
  until one branch fills the GPU intentionally removes the opportunity.
- `linear_stencil`: a serial stencil/transform chain. It measures launch amortization but offers no
  dependency optimization and acts as a negative control.

Each workload runs through four modes:

1. ordinary single-stream launches;
2. single-stream CUDA Graph capture;
3. manually coordinated streams and events; and
4. an explicit CUDA Graph with minimal dependencies.

Every mode starts from identical data. The executable compares final checksums with the sequential
mode and emits one CSV row per workload/mode.

## Build and run

```bash
cmake -S . -B build-cuda -G Ninja \
  -DCKL_BUILD_CUDA_BENCHMARKS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build build-cuda --target ckl-milestone0

python3 benchmarks/milestone0/run.py \
  --executable build-cuda/benchmarks/milestone0/ckl-milestone0 \
  --output milestone0.csv

python3 benchmarks/milestone0/analyze.py milestone0.csv
```

Use a release build and an architecture matching the test GPU for reported results. The runner
records device and configuration metadata alongside timing rows.

## Selection rule

The analyzer selects a primary workload only when all modes pass correctness and the explicit graph
beats single-stream capture by at least 10%. If several workloads qualify, it selects the largest
speedup. Otherwise it reports that no primary workload has been established. This prevents the
project from claiming orchestration value from launch amortization alone.

The threshold can be changed with `analyze.py --minimum-speedup`, but any published result must
state the chosen threshold and include the raw CSV.

## Recorded result

The checked-in [RTX 5060 Ti result](results/rtx5060ti.csv) used CUDA 12.8, 2,048 elements, 200
cycles, 20 warm-up cycles, and nine samples. All modes passed checksum validation.

| workload | capture median | explicit median | capture / explicit | decision |
| --- | ---: | ---: | ---: | --- |
| `multi_field` | 2.871 ms | 1.836 ms | 1.564x | selected |
| `linear_stencil` | 0.823 ms | 0.823 ms | 1.000x | negative control |

This establishes a focused use case rather than a universal scheduling win: repeated narrow field
updates whose branches underfill the GPU individually. Full-width stencil branches were also
tested and rejected because each branch already saturated the device, leaving no overlap headroom.

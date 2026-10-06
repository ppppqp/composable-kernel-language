# HIP runtime-overhead benchmark (Stage 0)

This benchmark isolates CKL's steady-state cost from compilation and integration work. It launches
the same `add_bias` HSACO on one imported HIP stream and buffer as linear chains of 2, 4, 8, and 16
nodes through four modes:

1. direct `hipModuleLaunchKernel` calls;
2. CKL ordinary submission from a retained `HipPlan`;
3. a manually constructed and retained HIP Graph; and
4. a retained CKL `HipGraphExecutable`.

Every mode is checked numerically. Raw GPU-event and CPU-submission samples are retained in CSV.
Module load, CKL resolution, and graph instantiation are emitted as separate setup metrics and never
occur in timed loops.

## RX 9060 XT build

The development toolchain and runtime library must come from the same ROCm installation. On this
machine, configure explicitly for the ROCm 10 development build and `gfx1200`:

```bash
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCKL_BUILD_TESTS=ON \
  -DCKL_ENABLE_HIP_RUNTIME=ON \
  -DCKL_BUILD_HIP_BENCHMARKS=ON \
  -DCKL_HIP_INCLUDE_DIR=/opt/rocm/core-10.0/include \
  -DCKL_HIP_LIBRARY=/opt/rocm/core-10.0/lib/libamdhip64.so \
  -DCKL_HIPCC_EXECUTABLE=/opt/rocm/core-10.0/bin/hipcc \
  -DCKL_HIP_RUNTIME_TEST_ARCH=gfx1200 \
  -DCKL_HIP_BENCHMARK_ARCH=gfx1200
cmake --build build-hip --target ckl-hip-runtime-validation ckl-hip-runtime-overhead
```

Run from a shell that can access `/dev/kfd` and `/dev/dri`:

```bash
python3 benchmarks/hip-runtime-overhead/run.py \
  --executable build-hip/benchmarks/hip-runtime-overhead/ckl-hip-runtime-overhead \
  --hsaco build-hip/benchmarks/hip-runtime-overhead/overhead-kernel.hsaco \
  --output benchmarks/hip-runtime-overhead/results/rx9060xt.csv

python3 benchmarks/hip-runtime-overhead/analyze.py \
  benchmarks/hip-runtime-overhead/results/rx9060xt.csv
```

`--iterations` controls launches per raw sample; it does not alter graph topology. Setup cost is
reported separately so a future integration can calculate the break-even replay count without
mistaking file loading or graph construction for steady-state overhead.

## Recorded RX 9060 XT result

The checked-in [raw samples](results/rx9060xt.csv) were collected on October 6, 2026 with 256
elements, 100 warmups, 1,000 replays per sample, and 11 interleaved samples. Ratios below are CKL
median divided by its corresponding native median; values near 1.0 indicate parity.

| nodes | ordinary GPU | ordinary CPU | graph GPU | graph CPU |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.024x | 1.028x | 0.997x | 1.014x |
| 4 | 1.000x | 1.021x | 0.995x | 1.029x |
| 8 | 1.007x | 1.003x | 1.000x | 1.025x |
| 16 | 1.000x | 1.001x | 1.000x | 1.009x |

The retained CKL graph meets the Stage 0 5% parity criterion at every topology. Its absolute CPU
submission difference from manual HIP Graph is 0.02–0.05 microseconds per replay. Pre-resolved CKL
ordinary execution also meets the 5% criterion; by 16 nodes its median CPU and GPU costs are within
0.2% and 0.01% of direct HIP respectively. This supports the narrow claim that CKL's retained C++
execution objects do not add material steady-state overhead. It does not yet measure a Python FFI
boundary or prove an end-to-end FlyDSL speedup.

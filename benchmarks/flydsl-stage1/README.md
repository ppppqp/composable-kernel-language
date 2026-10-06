# FlyDSL two-stage benchmark (Stage 1)

This benchmark runs FlyDSL's two-stage RMSNorm backward on the same PyTorch tensors and HIP stream
through four steady-state paths:

1. the native FlyDSL callable;
2. PyTorch `CUDAGraph` capture of that callable;
3. a retained, pre-resolved CKL `HipPlan`; and
4. a retained CKL `HipGraphExecutable`.

The workload launches a partial input/weight-gradient kernel followed by a weight-gradient
reduction kernel. Every path is checked against the native outputs (`dx`, `dweight`, and the
intermediate partial buffer) before timing. GPU execution and CPU submission are sampled
separately; compilation, module loading, plan resolution, and graph capture remain setup metrics.

## Prerequisites

The FlyDSL checkout must expose `CompiledFunction.artifact.export_for_orchestration()` and preserve
direct host memrefs in its launch plan. The CKL adapter expands each logical FlyDSL memref into the
physical ROCm kernarg ABI: a bare device pointer followed by the packed dynamic shape/stride
descriptor. The initial benchmark supports only a statically specialized contiguous layout and
rejects missing descriptor bytes.

CKL and PyTorch must use the same HIP runtime in one process. The commands below use the HIP 7.2
headers bundled with Triton and PyTorch's `libamdhip64.so`; substituting a different system ROCm
runtime can produce symbol/version conflicts.

## Reproduce

From the CKL repository, with FlyDSL at the path shown:

```bash
FLYDSL=/home/qiping-pan/Documents/workspace/FlyDSL
cmake --build build --target ckl-import-manifest ckl-hostgen

source "$FLYDSL/.venv/bin/activate"
export PYTHONPATH="$PWD/python:$FLYDSL:$FLYDSL/build-fly/python_packages:$FLYDSL/python"

python benchmarks/flydsl-stage1/prepare.py \
  --output /tmp/ckl-flydsl-stage1-bundle \
  --ckl-import-manifest build/tools/ckl-import-manifest/ckl-import-manifest \
  --ckl-hostgen build/tools/ckl-hostgen/ckl-hostgen

cmake -S . -B build-flydsl-stage1 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCKL_ENABLE_HIP_RUNTIME=ON \
  -DCKL_BUILD_HIP_BENCHMARKS=ON \
  -DCKL_HIP_INCLUDE_DIR="$FLYDSL/.venv/lib/python3.12/site-packages/triton/backends/amd/include" \
  -DCKL_HIP_LIBRARY="$FLYDSL/.venv/lib/python3.12/site-packages/torch/lib/libamdhip64.so" \
  -DCKL_FLYDSL_STAGE1_BUNDLE_DIR=/tmp/ckl-flydsl-stage1-bundle

cmake --build build-flydsl-stage1 --target ckl-flydsl-stage1-bridge

python benchmarks/flydsl-stage1/run.py \
  --bundle /tmp/ckl-flydsl-stage1-bundle \
  --bridge build-flydsl-stage1/benchmarks/flydsl-stage1/libckl-flydsl-stage1-bridge.so \
  --output benchmarks/flydsl-stage1/results/rx9060xt-512x256.csv

python benchmarks/flydsl-stage1/analyze.py \
  benchmarks/flydsl-stage1/results/rx9060xt-512x256.csv
```

`prepare.py` specializes descriptor bytes for the requested dimensions. To test another row count,
pass the same `--rows` value to both `prepare.py` and `run.py`, then rebuild the bridge after
regenerating the bundle. File loading and graph creation happen once in `Bridge.create`; neither is
inside a timed replay loop.

## Recorded RX 9060 XT result

The checked-in [raw samples](results/rx9060xt-512x256.csv) use 512 rows, 256 columns, 32 partial
programs, 100 warmups, 1,000 replays per sample, and 11 interleaved samples. All modes passed the
three-output numerical comparison.

| mode | median GPU (us) | vs eager GPU | median CPU submit (us) | vs eager CPU |
| --- | ---: | ---: | ---: | ---: |
| FlyDSL eager | 14.881 | 1.000x | 4.534 | 1.000x |
| PyTorch graph | 21.502 | 1.445x | 3.230 | 0.712x |
| CKL ordinary | 14.876 | 1.000x | 1.842 | 0.406x |
| CKL graph | 21.543 | 1.448x | 2.970 | 0.655x |

The focused Stage 1 conclusion is parity, not a graph speedup. CKL ordinary matches native FlyDSL
device time, and CKL graph matches PyTorch graph device time within 0.2%. Both graph paths are
about 44% slower on the GPU for this small two-launch specialization, even though they reduce CPU
submission time. That negative result makes larger launch-count workloads—particularly statically
specialized MoE sorting—the next useful benchmark rather than further tuning this RMSNorm case.

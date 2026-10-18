# Image inference preprocessing

**Status:** float RGB normalization component implemented and measured. Resize, crop, byte-input
conversion, decode, and full inference preprocessing remain deferred. See
[the report](results/rtx5060ti.md) and [raw samples](results/rtx5060ti.json).

## Source and scope

Use [TorchVision's classification presets](https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py)
and [tensor transforms](https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py)
as references. Pin the revision and specify input format, resize geometry, interpolation,
antialiasing, crop, dtype conversion, and channel statistics before implementing the full pipeline.
The normalization component follows the clone/subtract/divide semantics in
[TorchVision v0.24.0](https://github.com/pytorch/vision/blob/v0.24.0/torchvision/transforms/_functional_tensor.py).

```text
resize → center crop → convert/rescale → subtract channel mean → divide by channel std
```

The current timed unit is normalization of one complete image batch resident on the GPU, with
float input in [0, 1]. The future full-pipeline benchmark must record its actual kernel sequence:
a crop may be a view, and library operations need not each launch exactly one kernel. Do not
turn views into copies solely to create fusion opportunities.

## Implemented controls

The [CKL source](workloads.py) preserves contiguous NCHW layout, flattening only H × W into a
pixel dimension. Separate launch axes select pixels, RGB channel, and batch; per-channel mean
and standard deviation remain buffer inputs. The existing frontend needed no extension.

- `ckl_off` / `ckl_on`: copy input, subtract mean in place, divide by std in place (three kernels).
- `ckl_copy_elided`: subtract directly into output, then divide (two kernels). No arithmetic
  fusion is needed to remove the copy.
- `ckl_manual`: authored normalization in one kernel. This is a headroom control, not an added
  compiler transformation.
- PyTorch eager, copy-elided, and compiled versions of the same component. The eager baseline
  uses a preallocated output copy instead of allocating a clone on each call; all variants
  exclude output allocation and preserve the source. No TorchVision runtime dependency is needed.

Each is measured with ordinary launches and graph execution. The automatic pass does not fuse
this rank-three, channel-dependent computation; the runner checks the actual CKL node counts.

## Staged port

1. Float-input RGB normalization is implemented, including channel statistics and batch layout.
2. Add faithful resize/crop and conversion. The full byte-input pipeline needs unsigned-byte
   buffer support and indexing facilities beyond the current frontend subset. Avoid silently
   substituting float input or a cheaper resize algorithm in the full-pipeline comparison.
3. Extend the current controls to the full pipeline and quantify how much normalization matters
   to total preprocessing. Component speedups alone cannot establish that result.

## Cases and baselines

- Start with RGB batches of 1, 8, and 32 images, resizing the shorter edge to 256 and taking a
  224 × 224 center crop. Include non-square and odd input dimensions; define exact sizes in
  the eventual runner. Add larger frames and batches to exceed last-level cache where feasible.
- Fix and report the physical layout; compare an additional layout only as a separate case.
- Measure CKL fusion off/on, ordinary/graph execution, and the manual CKL implementation.
- Compare TorchVision's tensor GPU path. Include a compiled PyTorch implementation where
  supported, with identical interpolation/antialiasing and numerical semantics.

The measured component cases are B × H × W = 1 × 257 × 259, 1 × 224 × 224,
8 × 224 × 224, 32 × 224 × 224, and 8 × 1080 × 1920, always with three channels.
The large batch has a roughly 380 MiB input/output working set, exceeding the measured GPU's
32 MiB L2 cache. No resize operation is included in these shapes.

## Reproduce

Use the same Release CKL build and CUDA PyTorch + NumPy environment as the
[SGD benchmark](../momentum-sgd/README.md#reproduce):

```bash
PYTHONPATH=python \
CKL_NATIVE_LIBRARY=/tmp/ckl-standalone-bench/python/ckl/libCKLCompiler.so \
TORCHINDUCTOR_CACHE_DIR=/tmp/ckl-app-inductor \
TRITON_CACHE_DIR=/tmp/ckl-app-triton \
CUDA_CACHE_PATH=/tmp/ckl-app-cuda \
XDG_CACHE_HOME=/tmp/ckl-app-cache \
/path/to/cuda-pytorch-env/bin/python benchmarks/image-preprocessing/run.py \
  --toolkit /path/to/cuda --chip sm_120 \
  --image-shapes 1x257x259 1x224x224 8x224x224 32x224x224 8x1080x1920 \
  --iterations 100 --samples 7 --warmup 10 --latency-iterations 10 \
  --output benchmarks/image-preprocessing/results/local.json
```

## Correctness and decisions

Validate shape, crop location, channel order, conversion endpoints, normalization, edge pixels,
and every output element. Specify tolerances appropriate to interpolation and dtype conversion.
Keep decode and transfers outside the resident-device measurement; report a separate
transfer-inclusive case if application latency is being claimed.

For the implemented normalization component, the independent NumPy checker validates every
output, unchanged source/statistics, and tail sentinels, including incomplete pixel blocks.
Each backend/mode is checked at repeat counts 1, 17, and the measured counts with
`atol=3e-6, rtol=3e-5`. Future crop/conversion checks above do not imply those stages are implemented.
State/output resets are outside timing. See the shared protocol and report for native-versus-Python
submission differences; direct cross-framework event timings are not isolated kernel comparisons.

Follow the [shared protocol](../README.md#shared-evaluation-protocol). Report images/s and complete
batch latency alongside the normalization component. This workload should tell us whether to
prioritize shape/index-aware fusion, conversion/broadcast support, or faster resize kernels.
A fast normalization tail with little full-pipeline benefit is a valid result.

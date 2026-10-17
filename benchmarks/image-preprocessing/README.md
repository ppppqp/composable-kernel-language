# Image inference preprocessing

**Status:** proposed; no CKL implementation or results yet.

## Source and scope

Use [TorchVision's classification presets](https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py)
and [tensor transforms](https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py)
as references. Pin the revision and specify input format, resize geometry, interpolation,
antialiasing, crop, dtype conversion, and channel statistics before implementing.

```text
resize → center crop → convert/rescale → subtract channel mean → divide by channel std
```

The primary timed unit is preprocessing one complete image batch resident on the GPU. Record
the actual kernel sequence: a crop may be a view, and library operations need not each launch
exactly one kernel. Do not turn views into copies solely to create fusion opportunities.

## Staged port

1. Implement a float-input normalization component first, retaining channel semantics. A
   contiguous single-channel case can exercise today's unary fusion; it is not the full RGB
   workload. Channel broadcasting/layout may require additional frontend support.
2. Add faithful resize/crop and conversion. The full byte-input pipeline needs unsigned-byte
   buffer support and indexing facilities beyond the current frontend subset. Avoid silently
   substituting float input or a cheaper resize algorithm in the full-pipeline comparison.
3. Add a manually optimized CKL version that removes legal materializations. Compare component
   and whole-pipeline results to quantify how much normalization actually matters.

## Cases and baselines

- Start with RGB batches of 1, 8, and 32 images, resizing the shorter edge to 256 and taking a
  224 × 224 center crop. Include non-square and odd input dimensions; define exact sizes in
  the eventual runner. Add larger frames and batches to exceed last-level cache where feasible.
- Fix and report the physical layout; compare an additional layout only as a separate case.
- Measure CKL fusion off/on, ordinary/graph execution, and the manual CKL implementation.
- Compare TorchVision's tensor GPU path. Include a compiled PyTorch implementation where
  supported, with identical interpolation/antialiasing and numerical semantics.

## Correctness and decisions

Validate shape, crop location, channel order, conversion endpoints, normalization, edge pixels,
and every output element. Specify tolerances appropriate to interpolation and dtype conversion.
Keep decode and transfers outside the resident-device measurement; report a separate
transfer-inclusive case if application latency is being claimed.

Follow the [shared protocol](../README.md#shared-evaluation-protocol). Report images/s and complete
batch latency alongside the normalization component. This workload should tell us whether to
prioritize shape/index-aware fusion, conversion/broadcast support, or faster resize kernels.
A fast normalization tail with little full-pipeline benefit is a valid result.

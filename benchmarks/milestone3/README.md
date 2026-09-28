# Milestone 3 NVIDIA runtime validation

This validation runs the selected multi-field topology through three paths over the same external
CUBIN, arguments, buffers, and six kernel nodes:

1. ordinary source-ordered Driver API launches;
2. a cached `CKLNvidiaRuntime` explicit graph; and
3. an independently constructed raw CUDA Driver graph.

It checks output equivalence, graph-cache reuse, and executable kernel-parameter update. When MLIR
is enabled, a separate test compiles `tests/Runtime/generated-kernel.mlir` through `ckl-opt`,
extracts its CUBIN, loads it through the runtime, and validates the lowered memref ABI.

Build and run on SM 120 with CUDA 12.8:

```bash
source ~/.bashrc
activate-cuda

cmake -S . -B build-runtime -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCKL_ENABLE_MLIR=ON \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DCKL_ENABLE_CUDA_RUNTIME=ON \
  -DCKL_RUNTIME_TEST_ARCH=120 \
  -DCKL_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build build-runtime

build-runtime/tests/Runtime/ckl-nvidia-runtime-validation \
  build-runtime/tests/Runtime/runtime-kernels.cubin \
  --elements 2048 --cycles 200 --warmup 20 --repeats 9
```

The checked-in RTX 5060 Ti result used CUDA 12.8. All modes produced the same checksum. CKL's graph
measured within 1% of the raw manual graph in this run; this validates runtime overhead parity, not
a general performance claim.

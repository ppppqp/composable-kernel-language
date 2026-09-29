# Milestone 4 measured optimization

This benchmark validates three plan-level optimizations on the selected small-kernel workload:

- reachability-proven reuse of two non-overlapping CUDA temporaries;
- graph-boundary batching by repeating the five-node topology 1, 2, 4, or 8 times; and
- 1, 2, or 4 independent graph instances in flight on separate streams and buffer sets.

The memory planner consumes graph edges and the complete set of users for each resource. It does
not trust manually supplied start/end intervals and does not add dependencies to make reuse legal.
The validation first runs an unplanned, ordinary-launch reference, then checks every measured
candidate against its checksum. The selected configuration is the one with the lowest median time;
setup time and every rejected candidate remain in the output as decision provenance.

Build and run on SM 120 with CUDA 12.8:

```bash
source ~/.bashrc
activate-cuda

cmake -S . -B build-runtime -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCKL_ENABLE_CUDA_RUNTIME=ON \
  -DCKL_RUNTIME_TEST_ARCH=120 \
  -DCKL_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build build-runtime

python benchmarks/milestone4/run.py \
  --executable build-runtime/tests/Runtime/ckl-milestone4-validation \
  --cubin build-runtime/tests/Runtime/runtime-kernels.cubin \
  --output benchmarks/milestone4/results/local.csv
```

The checked-in RTX 5060 Ti run reduced per-instance temporary storage from 16,384 to 8,192 bytes.
For 192 logical iterations, the measured search selected batch 8 with four instances in flight at
0.622 ms, versus 2.755 ms for batch 1 with one instance. This is a focused underfilled-kernel
result, not a claim that more streams or larger graphs universally improve performance.

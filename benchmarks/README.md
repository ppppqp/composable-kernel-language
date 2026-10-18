# CKL benchmarks

The [autoregressive decoder](decoder/README.md) is the benchmark suite. It compares identical,
unfused CKL kernels under ordinary launches, captured and inferred graphs, storage reuse,
step grouping, and independent request streams with matched batching controls.

See the [GPU trace report](decoder/results/deep-dive/REPORT.md),
[full measurement matrix](decoder/results/deep-dive/matrix.md), and
[interactive trace viewer](decoder/results/deep-dive/traces/viewer.html).
The decoder README provides build, measurement, correctness, and profiling instructions.

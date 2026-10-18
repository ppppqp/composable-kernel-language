# In-process CUDA compilation and execution

CKL compiles specialized kernel bodies and their planned host program in the Python process.
Following FlyDSL's approach, device bytes are embedded in a host LLVM module, and MLIR's
`ExecutionEngine` JIT-compiles its entry point. There is no generated C++ file, exported CUBIN,
or external CKL compilation tool in this path.

```python
from array import array
from pathlib import Path
import ckl
from vector_pipeline import lower  # examples/vector_pipeline.py

program = lower().compile(
    target=ckl.NVIDIATarget("sm_120", "+ptx87", Path("/path/to/cuda")),
)
device = ckl.Device()
spec = ckl.BufferSpec((257,), ckl.f32)
src, dst = device.empty(spec), device.empty(spec)
src.copy_from_host(array("f", range(257)))

program.run(src, dst, 0.125)
result = array("f", [0.] * 257)
dst.copy_to_host(result)  # result[i] == i + 0.25

# Bind once to reuse temporary storage and a lazily instantiated CUDA graph.
invocation = program.bind(src, dst, 0.125)
invocation.run(graph=True)
invocation.run(graph=True)
```

## Build and ownership

Build with `CKL_ENABLE_MLIR=ON` against LLVM/MLIR with the native execution engine, NVPTX, and
NVIDIA binary serialization enabled. This builds `libCKLCompiler.so` under `build/python/ckl`.
Python finds it there or alongside the installed package; `CKL_NATIVE_LIBRARY` selects another
build. `CKL_ENABLE_CUDA_RUNTIME=ON` enables execution. Compilation needs a compatible CUDA toolkit
but no GPU and no MLIR Python bindings. Choose chip/PTX features supported by your toolkit and GPU.
The toolkit path is passed directly to the serializer without changing process environment variables.
Upstream serialization may invoke CUDA toolkit programs such as `ptxas` internally.

`CompiledProgram` owns its MLIR context, intermediate modules, embedded binary, host entry point,
and lazily loaded CUDA module. Creating `Device` initializes CUDA device 0. A `DeviceBuffer` owns
its allocation, or borrows an address with `device.import_buffer(address, spec, owner=owner)`.
The owner is retained; the caller must supply a valid device-0 allocation of sufficient size.
Host copies accept contiguous Python buffer objects, including `array` and NumPy arrays.
Host-to-device copies wait for device transfer completion before returning, including pageable
host buffers; subsequent nonblocking execution streams can safely consume those buffers.

A bound `Invocation` retains the compiled program, device, external buffers and their owners,
temporary heaps, stream, and optional graph executable. Execution is synchronous. This makes
`program.run(...)` safe as a convenience call; `bind(...).run(...)` supports repeated execution
without reallocating heaps or recreating the graph. Scalar arguments are captured at binding;
rebind to change them. Automatic tensor/framework conversion and borrowed streams are deferred.

## Execution modes, repetition, and storage ablations

Bound invocations support explicit `mode="ordinary"`, `mode="capture"`, and `mode="graph"`.
Ordinary mode submits kernels in topological order to one stream. Capture records those same
launches with CUDA stream capture. Graph mode builds an explicit graph using inferred dependencies.
`graph=True` remains a shorthand for graph mode. Specify either `mode` or `graph=True`.

```python
invocation.run(mode="capture")
invocation.run(mode="graph", batch_size=16)
timing = invocation.benchmark(iterations=1, mode="graph", batch_size=16)
```

`batch_size` repeats the complete bound program, not its tensor batch dimension. Captured and
inferred graphs are cached by mode/repeat count within the invocation. Every next iteration's
roots depend on all previous leaves, preserving mutable state and temporary lifetimes. A grouped
plan is limited to one million kernel nodes. Benchmark times are per original program execution;
the example executes 16 programs and divides stream/wall elapsed time by 16. Graph setup is
reported separately. Scalar bindings remain fixed; evolving state must live in buffers, as in
the [decoder benchmark](../benchmarks/decoder/README.md).

All calls still synchronize before returning. Grouping delays host observation until the group
finishes; it does not provide an asynchronous completion/ownership API. The caller must ensure
the program's input bounds/capacity contract remains valid across every repetition. Capture has
unspecified native node indices and does not support indexed kernel-parameter updates.

`.compile(target=..., reuse_storage=False)` assigns dedicated temporary slots instead of recycling
ordered lifetimes. It preserves device bodies and dependency inference, enabling a storage-only
ablation. The graph pass exposes the same choice as `--ckl-build-graph=reuse-storage=false`.

## Pointwise-chain fusion

`.compile(target=...)` enables conservative pointwise fusion by default. Use
`.compile(target=..., fuse_pointwise=False)` to retain original launches for comparison.
The `ckl-fuse-pointwise` pass runs before CUDA preparation and artifact hashing. It combines
adjacent unary kernels connected by a single-use compiler-owned temporary, substitutes the
producer's scalar result for consumer loads, and removes the intermediate allocation and free.
Effects, dependencies, storage, and binary identity are then derived from the transformed bodies.

The initial recognizer requires two same-type, static, contiguous rank-one buffers; `global_id(0)`;
an exact `i < N` guard; same-index loads; and one final output store. Bodies contain pure `arith`
operations, with no relaxed floating-point flags. Both launches must have matching one-dimensional
geometry/device, zero shared memory, no explicit token edges, and no intervening effects.
Views, escaping or multiply consumed temporaries, nested control flow, stencils, and binary
joins remain unchanged. Operations are cloned without reassociation or added fast-math flags.

External buffers can overlap even when they have different argument names. When fusion succeeds,
the native compiler retains a separately compiled original program. At binding, it compares the
logical byte ranges of all external buffers; any overlap selects the original topology. Adjacent
disjoint slices can use fusion. Exact aliases conservatively use the original program too.
`invocation.node_count` reports the actual selected topology; the compiled object's IR properties
describe the optimized candidate. This guard preserves shifted-overlap semantics without requiring
a new no-alias promise from source authors. Both paths retain their usual ordinary/graph support.

The fallback roughly doubles compilation time for eligible programs in the initial implementation.
It is compiled once per compiled object, with device loading deferred until that path is bound.
Guarded pass output carries `ckl.fusion.requires_disjoint`; it is inspection IR with a runtime
contract, not a standalone replacement for original source. Native compilation rejects pre-fused
marked input because it needs original source to construct the fallback.

Fusion regression tests retain automatic, disabled, and manually fused controls. CUDA tests cover
partial blocks, million-element buffers, shifted overlaps, adjacent slices, and rounding-sensitive
multiply/add operations.

## Compiler stages

The native library runs CKL's pass managers directly. CUDA preparation checks the supported
kernel subset, maps dispatches to logical device 0, and assigns `cuda.direct` ABI metadata.
A versioned artifact identity covers specialized kernel IR, chip/features, and the lowering policy:
64-bit indices, bare buffer pointers, and optimization level 3. Analysis summaries and source
locations do not participate; runtime scalar values and buffer addresses do not participate.
This identity is not a complete persistent compiler/toolkit cache key. There is no disk cache.

Effect inference, graph construction, storage planning, and executable lowering operate on the
prepared bodies. The device branch clones those same bodies into `gpu.func`, lowers indexed
accesses to `memref.load/store`, and expands device coordinates to GPU operations. Upstream NVVM
lowering serializes a CUBIN. The compiler validates the single NVIDIA binary object and reads its
bytes directly from the attribute in memory.

The host branch lowers the verified `ckl_exec.plan` into LLVM calls that allocate planned heaps,
bind external buffers and temporary slices, pack physical kernel arguments, and build the native
launch topology. The binary is an LLVM global in that same module. `ExecutionEngine` compiles the
host entry point and resolves its callbacks to CKL's CUDA runtime. Binding calls that entry point;
ordinary execution and CUDA graph execution then use the existing native executor.

For debugging, `program.prepared_ir`, `.executable_ir`, `.device_ir`, and `.host_ir` expose text
without writing files. `ckl-opt` remains an optional verifier/inspection tool. Host-generation and
binary-extraction executables and their C++ source emitter have been removed.

The initial ABI accepts static identity-layout default-space memrefs and f32/f64/i32/i64 scalars
and elements. Each buffer occupies one pointer slot; shape and stride are compile-time values.
Kernel IR and the Python frontend support structured kernel loops with scalar loop-carried values.
Constant program loops unroll during specialization; runtime host/program control flow remains deferred.
Dynamic shapes, views, nontrivial layouts, device calls, atomics, synchronization, and legacy
whole-buffer operations are unsupported. Host composition is straight-line with static launches.
Kernel authors guard partial blocks; the compiler does not insert bounds checks automatically.
The HIP runtime remains available for infrastructure tests; the in-process source compiler is CUDA-only.

## Acceptance test

Enable MLIR and the CUDA runtime, configure `CUDAToolkit_ROOT` and `CKL_RUNTIME_TEST_ARCH`, then:

```bash
cmake --build build
ctest --test-dir build -R '^ckl-cuda-frontend-runtime$' --output-on-failure
```

The Python test compiles `examples/vector_pipeline.py` in process, checks the host LLVM and
inferred two-node topology, then checks every output element for two runtime biases, sequential
launches, and repeated graph execution. A length of 257 exercises a partial block; tail sentinels
and unchanged input are checked. It also checks argument rejection and that a bound invocation
keeps its program and buffer owners alive after other Python references are deleted.
Additional numerical checks cover negative i64 constants and runtime f64 scalars.
Compilation is checked even without an accessible GPU; numerical execution then returns skip code 77.

The [decoder benchmark](../benchmarks/decoder/README.md) measures complete autoregressive steps
and independent request groups. `Invocation.benchmark(iterations=..., graph=...)`
measures a native submission burst with one final synchronization, returning CUDA stream and host
wall microseconds per program plus graph-instantiation time. This is measurement instrumentation;
the application-facing `.run()` remains synchronous. Stream timing includes host submission gaps.

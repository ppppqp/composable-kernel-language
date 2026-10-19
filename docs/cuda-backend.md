# In-process CUDA compilation and execution

CKL compiles specialized kernel bodies and their planned host program in the Python process.
Following FlyDSL's approach, a compiler pass lowers the executable plan to the MLIR LLVM dialect,
with device bytes embedded as an LLVM dialect global. Upstream LLVM translation and MLIR's
`ExecutionEngine` JIT-compile its entry point. There is no generated C++ file, exported CUBIN,
or external CKL compilation tool in this path.

```python
from array import array
from pathlib import Path
import ckl
from vector_pipeline import lower  # examples/vector_pipeline.py

with lower().compile(
    target=ckl.NVIDIATarget("sm_120", "+ptx87", Path("/path/to/cuda")),
) as program, ckl.Device() as device:
    spec = ckl.BufferSpec((257,), ckl.f32)
    with device.empty(spec) as src, device.empty(spec) as dst:
        src.copy_from_host(array("f", range(257)))
        with program.bind(src, dst, 0.125) as invocation:
            invocation.run(graph=True)
            invocation.run(graph=True)
        result = array("f", [0.] * 257)
        dst.copy_to_host(result)  # result[i] == i + 0.25
```

## Host lowering

The host compilation path is:

```text
verified ckl_exec plan + source signature + embedded CUBIN
    → ckl-lower-exec-to-llvm
    → MLIR LLVM dialect
    → upstream translateModuleToLLVMIR
    → ExecutionEngine → ckl_entry(frame, arguments)
```

`lib/Transforms/ExecToLLVM.cpp` implements the lowering. `python/ckl/_jit.py` selects and runs
the pipelines using `mlir.passmanager.PassManager`, stages the host module, and creates
`mlir.execution_engine.ExecutionEngine`. `NativeBindings.cpp` registers CKL dialects/passes
and exposes LLVM IR inspection. `lib/Runtime/NvidiaProgram.cpp` owns the host runtime callbacks,
argument packing, binding/alias fallback, graph caching, execution, and timing. Its C++ interface
in `include/ckl/Runtime/NvidiaProgram.h` has no Python or MLIR dependency.
`RuntimeBindings.cpp` only converts Python arguments, handles the Python buffer protocol, and
exposes runtime objects with their keep-alive relationships. There is no native compile function
or direct LLVM IR builder. Inspect
`program.host_mlir` for the lowered LLVM dialect,
`program.host_ir` for translated LLVM IR, and `program.executable_ir` for the original plan.

The pass also runs independently through `ckl-opt --ckl-lower-exec-to-llvm`. Its input is a
dedicated host module containing exactly one `ckl_exec.plan`, the referenced `func.func`
declaration describing the original argument types, and a non-empty `ckl_exec.binary` string
attribute on the module. It consumes that staging module and emits LLVM operations only.
The current ABI supports CUDA device 0, global resources, and `cuda.direct` kernels. Unsupported
backends, unresolved or mismatched argument bindings, and unrelated module operations fail with
diagnostics. The pass preserves physical slot order, scalar bits, temporary heap offsets, launch
geometry, and dependency IDs. See `tests/MLIR/test_exec_to_llvm.py` for a complete standalone input.
The Python driver uses JIT optimization level 0: the host entry constructs a plan once per bind,
while repeated execution remains in the native runtime and cached graphs.

The generated entry runs once during `bind()`: it loads/caches the device module, allocates
temporary heaps, and constructs the bound runtime plan. Subsequent execution uses that plan or
its cached command graph. This lowering change preserves that lifecycle; it does not add dynamic
host control flow or move launches into Python.

The [host compilation migration report](../benchmarks/decoder/results/exec-llvm/README.md)
contains Debug/Release regression results, matched decode measurements, and repeated setup costs.

## Build and ownership

Build with `CKL_ENABLE_MLIR=ON` against LLVM/MLIR with the native execution engine, NVPTX, and
NVIDIA binary serialization enabled. Install `nanobind>=2.9,<3` in the selected Python
3.12 environment, then configure CMake with `-DPython_EXECUTABLE=/path/to/python`.
This builds `_ckl_native.<python-abi>.so` under `build/python/ckl`.
Python imports that nanobind extension from there or alongside the installed package;
`CKL_NATIVE_LIBRARY` optionally selects the extension file from another build. It no longer
points to a C ABI library. Matching upstream MLIR Python bindings are required for the frontend;
set `CKL_MLIR_PYTHON_PATH` for CMake tests and include it in `PYTHONPATH` for direct Python use. `CKL_ENABLE_CUDA_RUNTIME=ON` enables execution. Compilation needs a compatible CUDA toolkit
but no GPU. Choose chip/PTX features supported by your toolkit and GPU.
The extension links against the matching Python package's shared `libMLIRPythonCAPI`, so
Python's pass manager and CKL use one MLIR implementation and pass registry. CMake locates
that library under `CKL_MLIR_PYTHON_PATH/mlir/_mlir_libs`; `CKL_MLIR_PYTHON_CAPI` can override it.
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
without reallocating heaps or recreating the graph.

The four resource classes support `with`, idempotent `close()`, and `closed`. Closing a
wrapper prevents further operations through that wrapper. Native shared ownership keeps resources
alive for existing buffers and bound invocations, including imported CKL allocations whose owner
wrapper has closed. Native destructors also provide cleanup when wrappers become unreachable;
there is no Python `__del__` handle management. Native bind/JIT entry, execution, and benchmark
bursts release the GIL. Compilation is orchestrated through the MLIR Python bindings.
Native program wrappers retain their Python execution engine, and bound invocation wrappers
retain those program wrappers. The engine's runtime-symbol references keep its host module and
MLIR context alive until after JIT destruction. Host copies use the Python buffer protocol.
Invalid native arguments map to Python `ValueError`; compilation and driver failures map to `RuntimeError`.

Scalar arguments are captured at binding; rebind to change them. Automatic tensor/framework conversion and borrowed streams are deferred.

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

## Subview packing

Static contiguous program `ckl.subview` operations retain identity-layout kernel argument types.
Executable descriptors carry `byte_offset` and `byte_length`, checked against the base resource
and element alignment before host lowering. `ckl_argument_resource` packs the base address plus
that offset; kernel indexing starts at zero within the logical view. Unmodeled `memref.subview`
and other layouts remain rejected. Runtime `DeviceBuffer.subview` wraps a bounded native slice.

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
the Python compiler retains a separately compiled original program. At binding, the native runtime compares the
logical byte ranges of all external buffers; any overlap selects the original topology. Adjacent
disjoint slices can use fusion. Exact aliases conservatively use the original program too.
`invocation.node_count` reports the actual selected topology; the compiled object's IR properties
describe the optimized candidate. This guard preserves shifted-overlap semantics without requiring
a new no-alias promise from source authors. Both paths retain their usual ordinary/graph support.

The fallback roughly doubles compilation time for eligible programs in the initial implementation.
It is compiled once per compiled object, with device loading deferred until that path is bound.
Guarded pass output carries `ckl.fusion.requires_disjoint`; it is inspection IR with a runtime
contract, not a standalone replacement for original source. Compilation rejects pre-fused
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
Dynamic shapes, dynamic/strided views, nontrivial layouts, device calls, atomics, synchronization, and legacy
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

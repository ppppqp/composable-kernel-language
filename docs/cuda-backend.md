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
Kernel IR supports structured conditionals and loops, although Python loops remain deferred.
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

The [standalone benchmark suite](../benchmarks/standalone/README.md) exercises pointwise chains,
fork/join composition, and neighbor-dependent stencils. `Invocation.benchmark(iterations=..., graph=...)`
measures a native submission burst with one final synchronization, returning CUDA stream and host
wall microseconds per program plus graph-instantiation time. This is measurement instrumentation;
the application-facing `.run()` remains synchronous. Stream timing includes host submission gaps.

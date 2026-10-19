# Initial CKL kernel IR

The first standalone-language step is implemented at the IR level. These operations describe
indexed computation and have verified types and effect semantics. The Python frontend emits
these bodies, and the initial [CUDA backend](cuda-backend.md) compiles them to device binaries.

## Kernel bodies and types

A kernel body is a defined `func.func` with the unit attribute `ckl.kernel`. It has no results;
output values are written to buffer arguments. Arguments are builtin integer, floating-point, or
`index` scalars, or ranked memrefs with those element types. Unranked buffers and nonnumeric types
are rejected. Static and dynamic dimensions are valid for kernel analysis; the current executable
program subset still requires statically planned resources.

Arithmetic, comparisons, and structured kernel control flow use upstream `arith` and `scf`.
Ranked memref shape, element type, layout, and memory space remain explicit in IR. Executable
lowering continues to reject resource layouts or views it cannot pack correctly.

## Device coordinates

The following operations take an `axis` i32 attribute in the range 0 through 2 (x, y, z) and
produce an `index` value. They require an enclosing marked kernel and have no memory effects.

| Operation | Meaning on the selected axis |
| --- | --- |
| `ckl.thread_id` | Zero-based thread coordinate within the current block |
| `ckl.block_id` | Zero-based block coordinate within the current grid |
| `ckl.block_dim` | Number of threads per block |
| `ckl.grid_dim` | Number of blocks per grid |
| `ckl.global_id` | `block_id * block_dim + thread_id` |

Coordinates are per-axis, not a flattened multidimensional index. Their values come from the
current invocation's launch configuration. Static launch dimensions use the existing three-value
`grid` and `block` attributes on `ckl.dispatch`.

## Indexed memory

`ckl.load_element(buffer, indices...)` returns the buffer element type.
`ckl.store_element(value, buffer, indices...)` requires that same type. Both require an enclosing
marked kernel and exactly one `index` operand per buffer dimension. Rank-zero buffers have no
index operands.

Indices are zero-based. Every executed access must satisfy `0 <= index < size` on every axis.
Verification rejects negative constant indices and constant indices past a static dimension.
Dynamic bounds are the kernel author's responsibility. These operations do not implicitly insert
clamping, masking, or bounds traps; an executed out-of-bounds access violates the kernel contract.
A guarded access in `scf.if` is the initial way to handle partial blocks.

Loads expose a read effect and stores a write effect on the buffer. Effects under conditionals
and loops are combined conservatively. Constant indices can describe one element; dynamic indices
cover the operand's complete region. Subviews preserve the operand region relative to its base.

The legacy `ckl.load`/`ckl.store` operations remain analysis-only whole-buffer fixtures. New device
kernel bodies use `load_element`/`store_element`; a later device lowering must reject legacy
operations rather than invent an index for them. The CUDA backend now enforces this rejection.

## Static contiguous subviews and access regions

`ckl.subview` has a source memref, an `offset` i64 attribute in source elements, and a result
memref. Both types have static identity layouts, matching element types and memory spaces.
The nonempty result span must fit within the source; nested offsets are relative to the immediate
source. The result shape can differ, but describes a contiguous row-major span. The Python
frontend exposes this in programs as `ckl.subview(buffer, offset=..., shape=...)`.

`AccessRegionOpInterface.getAccessRegion(resource)` returns an `Attribute`: `"whole"`, `"unknown"`,
or `{offset = ... : i64, length = ... : i64}` in bytes relative to the accessed operand. Constant
indexed loads/stores report one element. Other accesses conservatively cover their operand.
Analysis composes subview offsets, substitutes regions through calls and dispatches, and merges
multiple accesses to an enclosing interval. Memory-conflict edges can be removed for disjoint
intervals of the same proven base. May-alias external arguments and unknown views remain
conservative. Allocation/free effects retain whole-base lifetime ordering, and storage planning
continues to operate on complete base allocations.

Static contiguous subviews are supported in CUDA program argument packing. Dynamic and strided
views, upstream views without an implemented offset model, and kernel-local subviews remain
outside the executable subset.

## Example and launch model

```mlir
func.func @update(%input: memref<257xf32>, %output: memref<257xf32>)
    attributes {ckl.kernel} {
  %i = "ckl.global_id"() {axis = 0 : i32} : () -> index
  %count = arith.constant 257 : index
  %inside = arith.cmpi ult, %i, %count : index
  scf.if %inside {
    %value = "ckl.load_element"(%input, %i) : (memref<257xf32>, index) -> f32
    %one = arith.constant 1.0 : f32
    %updated = arith.addf %value, %one : f32
    "ckl.store_element"(%updated, %output, %i) : (f32, memref<257xf32>, index) -> ()
  }
  func.return
}
```

The program invokes this kernel using `ckl.dispatch` with grid `[2, 1, 1]` and block `[256, 1, 1]`.
The second block's excess threads do no memory access. Kernel launches occur in host/program
functions; dispatch inside a marked device kernel is rejected. A dispatch's referenced function
must have no results and its actual argument types must match the kernel signature.

The initial program execution subset is straight-line host dispatches on one device with static
geometry. Kernel-local `scf.if` and `scf.for` are analyzed recursively; they do not make the host
launch conditional. Host-level conditional/repeated launches still fail executable lowering.

Legacy analysis fixtures may dispatch unmarked functions. The new frontend will mark every device
kernel; CUDA preparation generates ABI, implementation, and artifact metadata. Handwritten IR
can still supply that metadata to exercise plan construction independently.

[The complete fixture](../tests/MLIR/indexed-kernel.mlir) composes two calls through a temporary
buffer and includes a multidimensional kernel loop. Inspect its effects and graph with:

```bash
build/tools/ckl-opt/ckl-opt tests/MLIR/indexed-kernel.mlir --ckl-summarize-effects
build/tools/ckl-opt/ckl-opt tests/MLIR/indexed-kernel.mlir --ckl-build-graph
```

The compiler derives the temporary's write/read dependency and can lower the graph to an
executable plan. This verifies analysis and plan construction only: the fixture's artifact name
does not refer to a generated device binary yet.

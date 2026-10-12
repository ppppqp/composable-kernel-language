# Python frontend proposal

**Status:** initial AST lowering implemented, 2026-10-11. The broader control-flow and runtime
APIs remain subsequent design stages.

## Implemented subset

The decorators, type declarations, `BufferSpec`, and `.lower(signature=..., constants=...)` are
implemented. The authoring example below lowers without a GPU or MLIR Python bindings. Optional
`.verify(ckl_opt=...)` and `.analyze(*passes, ckl_opt=...)` invoke the native optimizer; analysis
returns the existing `CompiledModule` with its `.mlir` output.

The initial implementation includes numeric scalar/buffer signatures, dimension unification,
constexpr bindings, indexed accesses, device coordinates, arithmetic/comparisons/casts,
compile-time conditionals, runtime kernel guards, temporary allocations, and explicit launches.
Runtime branch locals are scoped to their branch; updating an outer local or using a branch value
outside its branch is rejected until SSA branch results are implemented. Kernel `for` loops and
program unrolling are also rejected for now. `.compile` and runtime invocation are not available.

The runnable lowering example is [examples/vector_pipeline.py](../examples/vector_pipeline.py).
The remaining sections define the approved direction, including staged features; they do not
promise that deferred control-flow or execution APIs exist.

The first frontend compiles a restricted Python AST into verified CKL/MLIR. It does not execute
kernel or program bodies to discover operations. It preserves Python source locations, derives
memory effects from emitted kernel bodies, and leaves device binaries and physical ABI decisions
to CKL backend lowering.

## Authoring model

Use `@ckl.kernel` for device computation and `@ckl.program` for host composition. Both decorated
functions are source definitions, not ordinary callable Python functions. Kernel outputs are
buffer writes; program outputs are caller-provided buffers in the first subset.

```python
import ckl

N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]

@ckl.kernel
def update(src: Vector, dst: Vector, bias: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < src.shape[0]:
        dst[i] = src[i] + bias

@ckl.program
def step(src: Vector, dst: Vector, bias: ckl.f32) -> None:
    temporary = ckl.empty_like(src)
    grid = (ckl.cdiv(src.shape[0], 256),)
    block = (256,)
    ckl.launch(update, src, temporary, bias, grid=grid, block=block)
    ckl.launch(update, temporary, dst, bias, grid=grid, block=block)
```

Matching occurrences of `N` require equal extents across arguments of each specialization. The
program writes `dst = src + 2 * bias` and the compiler derives the temporary write/read dependency.
For a length of 257, the launch uses two blocks; excess threads are explicitly guarded.

Python outside decorated bodies can generate configurations and choose which program to compile.
Inside a body, only the documented language subset is accepted. An unsupported construct raises a
source-located error; it never falls back to executing arbitrary Python.

## Why AST compilation

Executing Python with symbolic values can intercept arithmetic and indexing, but ordinary Python
conditionals need a concrete truth value. AST compilation can lower a runtime `if` to `scf.if` and a
kernel `for` to `scf.for` while checking both paths. It also detects unsupported syntax before any
GPU work and retains exact source locations.

Decorators capture retrievable source, the function's lexical names, and its signature. The frontend
parses the source and resolves only recognized type declarations, dimension symbols, CKL
intrinsics, CKL definitions, and immutable scalar/tuple constants. It does not `eval` annotation
strings or call arbitrary user functions during compilation.

Ordinary argument annotations may already have been evaluated by Python when the function was
defined. With postponed annotations, the frontend resolves the supported annotation AST itself.
Mutable containers, arbitrary objects, and module state are not compile-time captures. Referenced
immutable bindings are snapshotted when the decorated definition is created; later rebinding does
not silently change its meaning. Captured CKL definitions and aliases have stable identities.

The initial implementation requires functions in source files with retrievable source. Notebook
cells, dynamically created functions, and lambdas are deferred; an explicit source API can be
added later. Diagnostics identify the definition and explain missing source.

## Types and shapes

| Source declaration | Meaning |
| --- | --- |
| `ckl.f32`, `ckl.f64` | Floating-point scalar |
| `ckl.i32`, `ckl.i64` | Signed scalar arithmetic, lowered to signless MLIR integers |
| `ckl.index` | Device index or shape/index expression |
| `ckl.bool` | Boolean predicate, lowered to i1 |
| `ckl.Buffer[element, shape]` | Ranked device buffer with that element type and extents |
| `ckl.dim("N")` | Named compile-time shape variable, inferred from buffer specifications |
| `ckl.constexpr[type]` | Explicit specialization parameter of the given scalar type |

Annotations are required for kernel and program arguments. Scalar types and ranks are not inferred
from device contents. Buffer shapes are tuples of nonnegative integer extents or dimension symbols.
The initial executable frontend requires positive extents; zero-size execution and genuinely
runtime dimensions are deferred. Kernel IR can still represent dynamic dimensions independently.

A compile signature supplies concrete shapes, element types, and layout information. Repeated
symbols must unify, and fixed extents must match. The first executable subset accepts only
contiguous, identity-layout buffers on one device. Specifications describe buffers without reading
their contents or allocating storage. The same symbolic name denotes the same extent in a
specialization; diagnostics explain any conflicting bindings.

Different functions are specialized independently. At each launch, actual buffer types bind the
callee's shape variables; these are not global mutable dimension assignments. `buffer.shape[k]`
is a compile-time integer for a specialized buffer, with a constant valid axis. `buffer.dtype` is a
compile-time type usable by allocation intrinsics. Neither is a device load.

Ordinary scalar arguments remain runtime values even when a caller happens to pass a Python
literal. A parameter explicitly marked `constexpr` must be bound during lowering and is removed
from the physical runtime signature. Shape symbols are inferred rather than duplicated as
`constexpr` arguments. Missing or incompatible bindings fail before IR emission.

For the first runnable path, buffer elements and runtime scalar parameters are limited to f32,
f64, i32, and i64, matching existing host argument packing. Boolean and index values initially
serve local predicates/index expressions and specialization constants. Broader scalar/buffer
signatures may be added to analysis lowering separately, but must not be advertised as runnable
until backend ABI and host packing support them.

## Expressions and numeric rules

The first kernel subset supports local assignment, buffer subscripts, scalar arithmetic,
comparisons, and explicit casts. Buffer indices are `index` values and there is one per dimension.
Rank-zero buffers use `buffer[()]`; negative indexing and slices are unsupported.

Scalar literals take the type of a typed operand or assignment context when unambiguous. Two
untyped integer literals form a compile-time integer expression. A literal must fit the selected
integer type; floating literals are rounded to the selected floating type. A runtime integer and a
runtime float are not implicitly combined. Use `ckl.cast(value, type)` when conversion is intended.
Index arithmetic and integer arithmetic are distinct; narrowing index values requires an explicit
cast. Unsigned integer types and implicit promotion ladders are deferred.

Initially support `+`, `-`, `*`, unary `-`, and comparisons on compatible types; `/` is floating
division. General integer division/modulo, bitwise operations, overflow policy changes, math
libraries, and vector values are staged additions. `ckl.cdiv(a, b)` is a compile-time intrinsic for
nonnegative extents and a positive divisor, primarily for launch geometry.

Integer runtime addition/subtraction/multiplication use fixed-width wraparound semantics without
signed-overflow assumptions. Index expressions must remain representable in the target index
width. Floating arithmetic initially preserves operation order without implicit fast-math flags.
Boolean expressions are predicates; Python `and`/`or` and chained comparisons are deferred until
short-circuit evaluation can be lowered correctly. Bitwise substitutes must not silently turn a
guarded buffer access into an unconditional one.

## Kernel control flow and memory

`if` with a compile-time predicate selects a specialization branch. `if` with a runtime predicate
lowers to `scf.if`; both paths are checked and their effects conservatively combined. Locals that
survive the branch become SSA results, with matching types on both paths. A local must be defined
on every path before use; an existing outer binding supplies the unchanged value on a path that
does not assign it.

`for i in range(start, stop, step)` lowers to `scf.for`. Bounds must be index-compatible and the
initial step must be a positive compile-time integer. The one- and two-argument forms use the
usual defaults. Loop-carried locals become `iter_args` with invariant types. A variable first
created inside a possibly empty loop cannot be used after it. The loop induction variable is
local to the loop; other loop-carried bindings retain their final values. `break`, `continue`,
`while`, and early returns are deferred. A final bare `return` or implicit fallthrough is allowed.

Device-coordinate intrinsics use constant axes 0 through 2 and map to the existing CKL index ops.
Indexed reads/writes map to `ckl.load_element`/`ckl.store_element`. Kernel bodies cannot allocate
program temporaries, launch kernels, use host I/O, or call Python functions. Device helper calls
are deferred; the first implementation compiles one complete body per kernel.

Bounds follow [the kernel IR contract](kernel-ir.md). No automatic masking, clamping, or traps are
inserted. Constant out-of-bounds accesses are rejected; runtime accesses must be guarded by the
kernel. Conditional stores and loads stay inside their source control-flow regions. An empty
`if` branch does not justify dropping an unknown effect.

## Program composition

The first program body has allocation, compile-time configuration, and explicit kernel launches.
`ckl.launch(kernel, *arguments, grid=..., block=..., shared_memory=0)` is the only launch spelling.
Calling a kernel as an ordinary Python function is an error with a suggested `ckl.launch` form.
Launch geometry belongs to the call site, allowing the same body to be launched differently.

Grid/block accept one, two, or three compile-time positive integers; missing trailing axes become
one. Shared-memory bytes are a nonnegative compile-time integer. Device and backend come from the
compilation target; source programs do not choose artifact paths, ABI identifiers, or native
handles. Target-specific limits are checked by backend compilation before execution.

`ckl.empty_like(buffer)` and `ckl.empty(shape, dtype=...)` create uninitialized, program-owned
temporaries. They are local to one program invocation and released after all their users complete.
The frontend emits allocations and end-of-program frees; the existing memory planner chooses
legal physical reuse. Uninitialized reads remain a source error/contract violation; `empty` does
not promise zeroing. Definite initialization analysis is deferred and must not be advertised as
implemented. Temporaries cannot escape or be returned in the first subset.

Program order is the conservative reference behavior. The compiler derives SSA, alias, memory,
and lifetime dependencies without a source-order token chain that would hide independence.
Caller-owned buffers may alias; no implicit `noalias` promise follows from different names.
Explicit completion ordering can be added when a concrete workload needs it.

Compile-time program conditionals and loops can specialize or unroll to straight-line IR with a
configurable expansion limit. Runtime program control flow is rejected. Ordinary program scalar
arguments can be forwarded unchanged to kernels; computed runtime host scalars are rejected
because current executable lowering cannot pack them. Kernel-local arithmetic is unaffected.

## Compilation API and staging

The first public API is explicit lowering without device execution:

```python
signature = {
    "src": ckl.BufferSpec(shape=(257,), dtype=ckl.f32),
    "dst": ckl.BufferSpec(shape=(257,), dtype=ckl.f32),
    "bias": ckl.f32,
}
module = step.lower(signature=signature, constants={})
print(module.mlir)
```

`lower` returns a `LoweredProgram` containing source-associated typed MLIR, the resolved signature,
specializations, a deterministic cache key, and diagnostics. Its signature exposes immutable
parameter and dimension descriptors; ordinary runtime parameter values are never part of the key. It is a new frontend result type, separate from the existing
raw-MLIR `CompiledModule`. Frontend lowering does not need a GPU, binary extraction, or a target
compiler. It can invoke `ckl-opt` for verification and analysis when available.

A subsequent backend API will accept this result and a target:

```python
executable = step.compile(signature=signature, constants={}, target=target)
execution = executable.run(src_buffer, dst_buffer, bias, mode="graph")
execution.wait()
```

This execution API is a direction for later work, not part of the first frontend implementation.
It needs complete kernel device lowering and a Python/runtime invocation bridge, neither of which
exists yet. `run` uses CKL buffers and optionally borrowed buffer/stream bindings, validates its
specialized signature, retains resources through completion, and leaves repeated execution and
synchronization explicit. Automatic compilation from arbitrary framework tensors is outside the
initial design.

`lower` creates an analyzed kernel specialization identity but does not invent a compiled artifact
or claim an ABI. Backend compilation attaches device, ABI, binary/artifact identity, and final
launch metadata before executable lowering. Only the complete backend pipeline may emit a
runnable program. Until then, frontend output supports verification/effect/graph analysis; an
unresolved executable plan correctly fails closed.

The existing dispatch operation requires a device string and implementation identity even for
analysis. Target-free lowering uses a logical `unassigned:0` device, the specialization hash as
implementation identity, and no ABI capabilities. Its artifact accessor currently falls back to
that implementation identity; this is an analysis reference, not a promise of an available binary.
Dependency analysis can run on it, while the current CUDA-specific storage-planning device parser
reports an unavailable plan. Backend preparation must map the logical device to CUDA/HIP and rerun
planning before executable lowering. Supporting logical devices in the memory planner is a small
follow-up if target-independent storage reports are needed; fabricating a CUDA target is not part
of frontend lowering.

Compilation identities include normalized source AST, referenced type/dimension/constant
bindings, callee identities, concrete buffer shapes/types/layouts, explicit constexpr values,
frontend version, and relevant options. Runtime scalar values, buffer addresses, and contents are
excluded. Backend cache keys additionally include target, ABI, compiler/toolchain, and device
artifact identity. Launch geometry participates in program identity. Source locations are retained
for diagnostics without making absolute paths the semantic identity.

## Implementation structure

Keep source capture, type/shape resolution, AST checking, and IR emission separate:

1. Capture definitions and source locations in immutable kernel/program objects.
2. Resolve annotations, compile signatures, dimension constraints, and constexpr bindings.
3. Check the supported AST subset and build a typed representation with definite assignment and
   static/runtime value distinctions.
4. Lower typed kernel control flow and memory operations, then program allocations and dispatches.
5. Verify emitted IR and reuse the existing effect/graph passes.

Start with deterministic textual MLIR emission to keep frontend authoring independent of MLIR
Python bindings. A scoped SSA/name builder handles blocks, values, types, and source locations;
use centralized escaping for symbols and strings. Generated identifiers use compiler-controlled
names, not arbitrary Python strings. Verification through `ckl-opt` is authoritative before
backend compilation. Direct MLIR Python construction can be evaluated later without changing the
source model.

Proposed modules under `python/ckl/` are `types.py`, `frontend/source.py`, `frontend/checker.py`, and
`frontend/emitter.py`. The current `compiler.py` stays responsible for optimizer invocation and
raw compiler artifacts. No runtime, framework adapter, or foreign compiler dependency belongs in
frontend source capture.

## Initial milestones and acceptance

Implement the authoring example first: typed buffers, shape specialization, a runtime f32 scalar,
a device index, a guarded indexed update, one temporary, and two explicit launches. This now
produces the same analysis semantics as the existing indexed-kernel fixture and infer the temporary
write/read edge without manual effect metadata.

Then add runtime kernel `if` results and `range` loops with loop-carried locals. Compile-time program
branching/unrolling and constexpr parameters follow once the straight-line path is verified. The
full proposal is a design boundary, not a requirement to implement every feature in one patch.

Required checks include deterministic output, shape/type mismatches, source-located unsupported
syntax, literal conversions, invalid launch geometry, branch/loop definite assignment, bounds
checks remaining inside control flow, and emitted IR verification. Existing analysis tests must
confirm read/write effects and dependency topology. Numerical source-to-device execution becomes
an acceptance gate after backend lowering and the runtime bridge are implemented.

The key proposed decisions are AST compilation, explicit typed signatures, shape specialization,
call-site launches, separate kernel/program contexts, and a lowering-first implementation. These
should be reviewed before the frontend syntax becomes a compatibility commitment.

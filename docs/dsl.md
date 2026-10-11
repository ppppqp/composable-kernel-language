# CKL language direction

**Status:** frontend design target; syntax is provisional.

CKL is a self-contained GPU DSL. A program authors both device computation and the sequence of
kernel invocations that forms a repeated GPU workload. The compiler owns kernel semantics,
specialization, launch configuration, device code generation, physical ABI packing, dependency
analysis, and execution planning.

## Source model

The initial frontend should be a small Python-embedded language with explicit kernel and program
entry points. Python is the authoring surface; typed MLIR is the semantic source of truth. The
current Python package contains compilation utilities only. The former prototype tracer is not
present, so the frontend must be built rather than assumed to exist.

The following sketches the intended authoring model; none of these decorators is implemented:

```python
@ckl.kernel
def update(source, destination, count):
    i = ckl.global_id(0)
    if i < count:
        destination[i] = source[i] + 1.0

@ckl.program
def step(a, b, count):
    temporary = ckl.empty_like(a)
    update(a, temporary, count)
    update(temporary, b, count)
```

Program order defines the observable reference behavior. The compiler derives memory effects and
launch dependencies from bodies and actual arguments. Authors do not maintain read/write lists,
ports, artifact registries, or physical ABI descriptors in source programs. Explicit completion
ordering is available when required by semantics beyond ordinary memory conflicts.

## First language subset

The first implementation needs:

- typed scalar values, ranked device buffers, arithmetic, and comparisons;
- indexed loads and stores with explicit bounds behavior;
- thread/block/global indices and a static launch configuration;
- structured conditionals and loops inside kernels;
- kernel calls from a straight-line program body;
- compiler-owned temporary allocations and caller-owned input/output buffers; and
- shape/target specialization with deterministic compilation identities.

The initial executable program subset has static geometry and sizes, one device, base resources,
and direct pointer/scalar arguments. Views and dynamic shapes may remain analyzable while
executable lowering rejects unsupported packing. Atomics, synchronization, reductions, shared
memory, and richer layouts are staged additions with explicit semantics and numerical tests.

The current `ckl.load` and `ckl.store` operations model whole-buffer effects for analysis tests.
They do not yet express indexed device computation. Extending kernel IR with indexing and device
execution semantics is required before those bodies can be compiled as real kernels.

## Compiler ownership

The frontend emits CKL kernel/program IR and ordinary upstream arithmetic and control-flow ops.
Effects and aliases are inferred before device lowering makes memory accesses opaque. CKL then
constructs the dependency graph, plans temporary storage, lowers kernel computation through
upstream GPU/NVVM or GPU/ROCDL infrastructure, and generates a verified executable plan.

Backend lowering defines the kernel ABI and expands logical operands to physical slots. The same
specialized body must supply both the inferred summary and executable binary. Cache keys include
source, specialization, target, ABI, and compiler options. Compiled objects and plan topology stay
separate internally to support loading and graph reuse, but callers invoke one compiled CKL
program.

## Implementation sequence

1. Define indexed kernel operations, buffer/scalar types, and the initial launch model.
2. Implement the frontend for a vector update and a two-kernel program.
3. Lower the same CKL kernel bodies to device binaries; connect their identities and ABI to
   `ckl_exec` and host generation.
4. Execute the two-kernel program and compare sequential and graph results element by element.
5. Port the branched multi-field workload entirely to CKL source, then measure temporary reuse,
   batching, and instances in flight.
6. Extend language features only when workloads and measured limitations justify them.

CUDA is the first complete source-to-device target. The retained HIP executor is a second backend
foundation; a complete CKL-to-ROCDL pipeline is separate work. Existing runtime fixtures with
hand-written device binaries remain infrastructure checks until CKL-generated bodies replace them.

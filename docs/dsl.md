# CKL language direction

**Status:** kernel IR and initial Python AST lowering implemented.

CKL is a self-contained GPU DSL. A program authors both device computation and the sequence of
kernel invocations that forms a repeated GPU workload. The compiler owns kernel semantics,
specialization, launch configuration, device code generation, physical ABI packing, dependency
analysis, and execution planning.

## Source model

The [Python frontend proposal](python-frontend.md) uses AST compilation of a restricted Python
subset with explicit kernel and program entry points. Python is the authoring surface; typed MLIR
is the semantic source of truth. The Python package contains the initial frontend and compilation
utilities. Richer control flow and runtime execution remain later work.

The following authoring example is supported by the initial frontend:

```python
N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]

@ckl.kernel
def update(source: Vector, destination: Vector) -> None:
    i = ckl.global_id(0)
    if i < source.shape[0]:
        destination[i] = source[i] + 1.0

@ckl.program
def step(a: Vector, b: Vector) -> None:
    temporary = ckl.empty_like(a)
    grid = (ckl.cdiv(a.shape[0], 256),)
    ckl.launch(update, a, temporary, grid=grid, block=(256,))
    ckl.launch(update, temporary, b, grid=grid, block=(256,))
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

The initial [kernel IR](kernel-ir.md) now provides `ckl.load_element`, `ckl.store_element`,
thread/block/global coordinates, and a verified `ckl.kernel` function marker. Ranked memrefs and
builtin numeric scalars supply the type model. The legacy `ckl.load` and `ckl.store` operations
remain analysis fixtures. Device lowering of the new indexed bodies is still to be implemented.

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

1. **Implemented at the IR level:** indexed kernel operations, buffer/scalar types, and the initial
   launch model. See [the verified contract](kernel-ir.md).
2. **Implemented for the initial subset:** AST frontend for a guarded vector update and a
   two-kernel program, including typed buffers, specialization, temporaries, and explicit launches.
3. Lower the same CKL kernel bodies to device binaries; connect their identities and ABI to
   `ckl_exec` and host generation.
4. Execute the two-kernel program and compare sequential and graph results element by element.
5. Port the branched multi-field workload entirely to CKL source, then measure temporary reuse,
   batching, and instances in flight.
6. Extend language features only when workloads and measured limitations justify them.

CUDA is the first complete source-to-device target. The retained HIP executor is a second backend
foundation; a complete CKL-to-ROCDL pipeline is separate work. Existing runtime fixtures with
hand-written device binaries remain infrastructure checks until CKL-generated bodies replace them.

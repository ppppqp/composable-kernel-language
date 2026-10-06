# CKL executable manifest v1

The executable manifest is the producer-neutral interchange format for a backend-ready CKL plan.
It lets a kernel DSL emit physical launch information without linking to CKL's MLIR libraries or
handing control to the DSL's native host runtime. The manifest is imported into verified
`ckl_exec.plan` IR; all host generation and execution after that point is shared with CKL-authored
programs.

```bash
build/tools/ckl-import-manifest/ckl-import-manifest plan.json -o plan.mlir
build/tools/ckl-hostgen/ckl-hostgen plan.mlir -o plan.cpp
```

See [`tests/MLIR/external-plan.json`](../tests/MLIR/external-plan.json) for a complete example.

## Contract

The root contains `schema_version: 1` and one `plan`. A plan contains:

- `source`, an MLIR symbol reference used for provenance;
- `name` and `backend` identities;
- `heaps`, the device allocations owned by the generated plan;
- `resources`, either caller-owned external buffers or slices of declared heaps; and
- `kernels`, in topological order, with explicit dependencies and physical ABI slots.

Each kernel provides its producer implementation identity, artifact registry key, entry-point
symbol, device, launch dimensions, dynamic shared-memory size, and ordered argument descriptors.
Argument `slot` values are physical ABI positions and must be contiguous. `logical_index` records
which producer-level operand generated a slot, so one logical operand may expand into multiple
physical slots in a future ABI adapter.

The v1 NVIDIA host path supports:

- `resource` arguments with `packing: "direct_pointer"`;
- named runtime `scalar` arguments of type `i32`, `i64`, `f32`, or `f64`; and
- embedded `constant` arguments of those scalar types.

Artifact bytes are intentionally not embedded in the manifest. `artifact` is a stable key into
the runtime `ArtifactRegistry`; the producer can register an in-memory CUBIN while retaining its
own compiler and cache. External resource names and scalar names become arguments of the generated
C++ plan builder.

## Trust boundary

This is a post-analysis interface. Importing a manifest does not infer memory effects, prove
dependencies, plan lifetimes, or reinterpret a producer ABI. A producer should either use CKL's
semantic graph construction before executable lowering or perform those tasks itself and export
the resulting plan. The importer validates the structural and backend invariants represented by
`ckl_exec`; it never fills in omitted packing rules or dependencies.

FlyDSL now exposes `CompiledFunction.artifact.export_for_orchestration()`. CKL's
`import_flydsl_artifact` consumes that public object, normalizes its copied `gpu.binary` objects,
and assigns a deterministic artifact identity. Device payloads cross the boundary as plain bytes,
so CKL does not load another MLIR Python runtime beside FlyDSL's bundled runtime. The adapter
deliberately does not read private `CompiledArtifact` or `CallState` fields. FlyDSL's exported
`rocm.bare_ptr` device ABI is distinct
from the host-wrapper ABI represented by `CallState`; completing executable integration requires
the HIP executor and automatic conversion of supported launch regions into this manifest.

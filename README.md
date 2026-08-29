# CKL

CKL is an experimental compiler for composing layout-aware GPU tile tasks. The project is
currently validating its target-independent core model before defining an MLIR dialect.

The repository follows the broad organization of FlyDSL while keeping the current build small:

```text
include/ckl/Core/   public layout and distribution APIs
lib/Core/           implementations
tests/Core/         semantic and property validation
specs/              design specifications
```

Build and validate the core with:

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

MLIR integration is intentionally disabled until the acceptance gates in
[`specs/core-design-tech-spec.md`](specs/core-design-tech-spec.md) are met.
The current implementation status and remaining gates are tracked in
[`docs/core-validation.md`](docs/core-validation.md).

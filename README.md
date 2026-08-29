# Composable Kernel Language

Composable Kernel Language(CKL) is an experimental compiler for composing layout-aware GPU tile tasks. It strives to solve the following issues that no existing compiler is good enough as far as I know:
1. Clean separation between executor space, local space, tile space and storage abstraction.
2. Visibility into the layout inference process.
3. Decoupling of compiler/DSL from the core layout infrastructure.


The project is organized into two parts:
- **Core**, which contains the core abstraction for layout, composition, planning and provenance. 
- **DSL**, which contains the frontend parser, MLIR dialect, and lowering pipeline.

The two parts are intentionally kept separate so that the core can be reused independent of the DSL.

## Core

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

## DSL
Not implemented yet.


## Acknowledgments
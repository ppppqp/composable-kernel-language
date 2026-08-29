# Composable Kernel Language

Composable Kernel Language(CKL) is an experimental compiler for composing layout-aware GPU tile tasks.

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

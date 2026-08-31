import os
import runpy
from pathlib import Path


namespace = runpy.run_path(os.environ["CKL_NVIDIA_EXAMPLE"])
compiled = namespace["compiled"]
assert len(compiled.gpu_objects) == 1
assert compiled.gpu_objects[0].data.startswith(b"\x7fELF")
assert "scalar-fallback" not in compiled.mlir
Path(os.environ["CKL_NVIDIA_OUTPUT"]).write_text(compiled.mlir)
print(f"frontend artifact {len(compiled.gpu_objects[0].data)} bytes")

import os
import runpy
from pathlib import Path


namespace = runpy.run_path(os.environ["CKL_NVIDIA_EXAMPLE"])
target = namespace["NVIDIATarget"](
    chip=os.environ["CKL_CUDA_ARCH"],
    features=os.environ["CKL_CUDA_PTX_FEATURE"],
    toolkit_root=Path(os.environ["CUDA_HOME"]),
)
options = namespace["CompilerOptions"](target=target)
compiled = namespace["kernel"].compile(options)
assert len(compiled.gpu_objects) == 1
assert compiled.gpu_objects[0].data.startswith(b"\x7fELF")
Path(os.environ["CKL_NVIDIA_OUTPUT"]).write_text(compiled.mlir)
print(f"frontend artifact {len(compiled.gpu_objects[0].data)} bytes")

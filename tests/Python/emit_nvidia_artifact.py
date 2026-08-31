import os
from pathlib import Path

from ckl import CompilerOptions, NVIDIATarget
from ckl.compiler import compile_module


source = Path(os.environ["CKL_NVIDIA_INPUT"]).read_text()
target = NVIDIATarget(
    chip=os.environ["CKL_CUDA_ARCH"],
    features=os.environ["CKL_CUDA_PTX_FEATURE"],
    toolkit_root=Path(os.environ["CKL_CUDA_TOOLKIT_ROOT"]),
)
compiled = compile_module(
    source,
    CompilerOptions(
        passes=(
            "--ckl-nvidia-materialize-mma-sync",
            "--ckl-nvidia-lower-mma-sync",
            "--ckl-nvidia-lower-fragment-io",
        ),
        target=target,
    ),
)
assert len(compiled.gpu_objects) == 1
gpu_object = compiled.gpu_objects[0]
assert gpu_object.format == 3
assert gpu_object.data.startswith(b"\x7fELF")
assert target.pipeline() in compiled.command
print(f"artifact {len(gpu_object.data)} bytes {gpu_object.target}")

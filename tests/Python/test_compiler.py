import unittest
from dataclasses import dataclass
from pathlib import Path

from ckl import CompilerOptions, NVIDIATarget, import_flydsl_artifact
from ckl.compiler import compilation_key, extract_gpu_objects


class CompilerUtilitiesTest(unittest.TestCase):
    def test_compilation_key_covers_source_and_command(self) -> None:
        baseline = compilation_key("module {}", ("ckl-opt", "--canonicalize"))
        self.assertEqual(baseline, compilation_key("module {}", ("ckl-opt", "--canonicalize")))
        self.assertNotEqual(baseline, compilation_key("module { }", ("ckl-opt", "--canonicalize")))
        self.assertNotEqual(baseline, compilation_key("module {}", ("ckl-opt", "--cse")))

    def test_nvidia_target_describes_pipeline_and_cache_identity(self) -> None:
        target = NVIDIATarget(
            chip="sm_90",
            features="+ptx80",
            toolkit_root=Path("/opt/cuda"),
        )
        self.assertIn("cubin-chip=sm_90", target.pipeline())
        self.assertIn("cubin-features=+ptx80", target.pipeline())
        self.assertIn("nvidia:sm_90:+ptx80", target.cache_identity())

    def test_explicit_optimizer_is_resolved(self) -> None:
        optimizer = Path(__file__)
        options = CompilerOptions(ckl_opt=optimizer)
        self.assertEqual(options.resolve_ckl_opt(), optimizer.resolve())

    def test_source_without_gpu_binary_has_no_objects(self) -> None:
        self.assertEqual(extract_gpu_objects("module {}"), ())

    def test_extracts_flydsl_rocm_gpu_object(self) -> None:
        source = '''module {
          gpu.binary @kernels [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "\\7FELF">]
        }'''
        try:
            objects = extract_gpu_objects(source)
        except ModuleNotFoundError:
            self.skipTest("MLIR Python bindings are not installed")
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].data, b"\x7fELF")
        self.assertIn("gfx942", objects[0].target)

    def test_flydsl_public_artifact_import(self) -> None:
        @dataclass(frozen=True)
        class DeviceObject:
            data: bytes = b"\x7fELF"
            format: int = 0
            target: str = '#rocdl.target<chip = "gfx942">'

        @dataclass(frozen=True)
        class Export:
            compiled_ir: str = "module {}"
            source_ir: str = "module { func.func @launch() { return } }"
            host_entry: str = "launch"
            backend: str = "rocm"
            target: str = "gfx942"
            kernel_abi: str = "rocm.bare_ptr"
            device_objects: tuple = (DeviceObject(),)

        first = import_flydsl_artifact(Export())
        second = import_flydsl_artifact(Export())
        self.assertEqual(first.identity, second.identity)
        self.assertTrue(first.identity.startswith("flydsl.rocm."))
        self.assertEqual(first.kernel_abi, "rocm.bare_ptr")
        self.assertEqual(first.gpu_objects[0].data, b"\x7fELF")
        self.assertIn("gfx942", first.gpu_objects[0].target)

    def test_flydsl_import_rejects_incomplete_export(self) -> None:
        @dataclass(frozen=True)
        class Export:
            compiled_ir: str = "module {}"
            source_ir: None = None
            host_entry: str = "launch"
            backend: str = "rocm"
            target: str = "gfx942"
            kernel_abi: str = "rocm.bare_ptr"
            device_objects: tuple = ()

        with self.assertRaisesRegex(ValueError, "source_ir"):
            import_flydsl_artifact(Export())


if __name__ == "__main__":
    unittest.main()

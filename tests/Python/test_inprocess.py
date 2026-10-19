"""Exercise the native compiler without a GPU."""
from dataclasses import replace
from contextlib import chdir
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ckl
from ckl.cuda import _native
from test_frontend import step


class InProcessTest(unittest.TestCase):
    def setUp(self):
        try:
            self.native = _native()
        except RuntimeError as error:
            self.skipTest(str(error))
        self.lowered = step.lower(signature={
            "src": ckl.BufferSpec((257,), ckl.f32),
            "dst": ckl.BufferSpec((257,), ckl.f32), "bias": ckl.f32})
        self.target = ckl.NVIDIATarget("sm_120", "+ptx87")

    def test_invalid_ir_reports_diagnostics_in_process(self):
        source = replace(self.lowered, mlir="module { invalid syntax }")
        with patch("subprocess.run", side_effect=AssertionError("external CKL tool")):
            with self.assertRaisesRegex(RuntimeError, "invalid CKL source module"):
                source.compile(target=self.target)

    def test_unsupported_kernel_rejected_before_device_compilation(self):
        source = replace(self.lowered, mlir=self.lowered.mlir.replace(
            "memref<257xf32>", "memref<?xf32>"))
        with self.assertRaisesRegex(RuntimeError, "static identity-layout"):
            source.compile(target=self.target)
        # A failed compile must not poison global registrations or later contexts.
        with self.assertRaisesRegex(RuntimeError, "invalid CKL source module"):
            replace(self.lowered, mlir="broken").compile(target=self.target)

    def test_compile_rejects_invalid_target(self):
        for target in (ckl.NVIDIATarget("bad target", "+ptx87"),
                       ckl.NVIDIATarget("sm_120", "+ptx87", binary_format="assembly")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.lowered.compile(target=target)

    def test_guarded_ir_requires_original_source(self):
        fused = self.lowered.analyze("--ckl-fuse-pointwise")
        with self.assertRaisesRegex(RuntimeError, "not pre-fused guarded IR"):
            replace(self.lowered, mlir=fused.mlir).compile(target=self.target)

    def test_embedded_binary_and_host_entry_point(self):
        toolkit = os.environ.get("CKL_CUDA_TOOLKIT")
        if not toolkit:
            self.skipTest("CUDA toolkit is not configured")
        target = ckl.NVIDIATarget("sm_120", "+ptx87", Path(toolkit))
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            before = set(Path(directory).iterdir())
            environment = {key: os.environ.get(key) for key in ("CUDA_HOME", "PATH")}
            with patch("subprocess.run", side_effect=AssertionError("external CKL tool")):
                compiled = self.lowered.compile(target=target, fuse_pointwise=False)
            self.assertEqual(before, set(Path(directory).iterdir()))
            self.assertEqual(environment, {key: os.environ.get(key) for key in environment})
        self.assertIn("@ckl_entry", compiled.host_ir)
        self.assertIn("@ckl.binary", compiled.host_ir)
        self.assertIn("call void @ckl_argument_value", compiled.host_ir)
        self.assertIn('dependencies = array<i64: 1>', compiled.executable_ir)
        self.assertIn('"ckl_exec.heap"', compiled.executable_ir)
        self.assertIn("gpu.binary", compiled.device_ir)


if __name__ == "__main__":
    unittest.main()

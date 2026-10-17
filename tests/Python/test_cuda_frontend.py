"""CUDA preparation/extraction checks that need neither a toolkit nor a GPU."""
import os
from pathlib import Path
import re
import subprocess
import unittest

from test_frontend import step
import ckl


class CudaFrontendTest(unittest.TestCase):
    def setUp(self):
        self.optimizer = Path(os.environ.get("CKL_OPT", str(
            Path(__file__).resolve().parents[2] / "build/tools/ckl-opt/ckl-opt")))
        if not self.optimizer.is_file():
            self.skipTest("CKL MLIR tools are not built")
        self.source = step.lower(signature={
            "src": ckl.BufferSpec((257,), ckl.f32),
            "dst": ckl.BufferSpec((257,), ckl.f32), "bias": ckl.f32,
        }).mlir

    def run_pass(self, source, *passes, error=None):
        result = subprocess.run([str(self.optimizer), *passes], input=source,
                                text=True, capture_output=True)
        if error:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_same_body_drives_host_and_device(self):
        prepared = self.run_pass(self.source, "--ckl-prepare-cuda")
        identity = re.search(r'ckl.cuda.artifact = "([^"]+)"', prepared).group(1)
        self.assertEqual(prepared.count(f'artifact = "{identity}"'), 3)
        executable = self.run_pass(prepared, "--ckl-build-graph", "--ckl-lower-graph-to-exec")
        self.assertIn(identity, executable)
        self.assertIn('"cuda.direct"', executable)
        self.assertIn('dependencies = array<i64: 1>', executable)
        device = self.run_pass(prepared, "--ckl-lower-kernels-to-gpu")
        for operation in ("gpu.func", "gpu.block_id", "gpu.block_dim", "gpu.thread_id",
                          "memref.load", "memref.store", "scf.if", "arith.addf"):
            self.assertIn(operation, device)
        self.assertNotIn("func.func", device)
        self.assertNotIn("ckl.dispatch", device)
        self.assertIn(identity, device)

    def test_identity_changes_with_body_or_target(self):
        def identity(source, *passes):
            text = self.run_pass(source, "--ckl-prepare-cuda", *passes)
            return re.search(r'ckl.cuda.artifact = "([^"]+)"', text).group(1)
        original = identity(self.source)
        self.assertEqual(original, identity(self.source))
        self.assertNotEqual(original, identity(self.source.replace("arith.addf", "arith.subf")))
        target = self.run_pass(self.source, "--ckl-prepare-cuda=chip=sm_90 features=+ptx80")
        self.assertNotIn(original, target)
        # Analysis metadata does not change the binary identity.
        self.assertEqual(original, identity(self.run_pass(self.source, "--ckl-summarize-effects")))

    def test_reject_stale_identity_and_missing_preparation(self):
        prepared = self.run_pass(self.source, "--ckl-prepare-cuda")
        self.run_pass(prepared.replace("arith.addf", "arith.subf"),
                      "--ckl-lower-kernels-to-gpu", error="identity is stale")
        self.run_pass(self.source, "--ckl-lower-kernels-to-gpu", error="requires a ckl-prepare-cuda")

    def test_reject_unsupported_abi_and_kernel_operations(self):
        self.run_pass(self.source.replace("memref<257xf32>", "memref<?xf32>"),
                      "--ckl-prepare-cuda", error="requires static identity-layout")
        # Construct a valid analysis-only operation rather than relying on frontend names.
        legacy = re.sub(r'"ckl.load_element"\(([^,]+), [^)]+\)([^\n]*?)\(memref<257xf32>, index\)',
                        r'"ckl.load"(\1)\2(memref<257xf32>)', self.source)
        self.run_pass(legacy, "--ckl-prepare-cuda", error="unsupported operation")

    def test_reject_incompatible_geometry_and_device(self):
        self.run_pass(self.source.replace("block = array<i64: 256, 1, 1>",
                                           "block = array<i64: 1025, 1, 1>"),
                      "--ckl-prepare-cuda", error="exceeds 1024 threads")
        self.run_pass(self.source.replace('"unassigned:0"', '"hip:0"'),
                      "--ckl-prepare-cuda", error="supports only logical device 0")

    def test_coordinates_multidimensional_access_and_loop(self):
        source = (Path(__file__).resolve().parents[1] / "MLIR/indexed-kernel.mlir").read_text()
        device = self.run_pass(source, "--ckl-prepare-cuda", "--ckl-lower-kernels-to-gpu")
        self.assertEqual(device.count("gpu.func"), 2)
        self.assertIn("gpu.grid_dim x", device)
        self.assertIn("gpu.block_id y", device)
        self.assertIn("gpu.block_dim z", device)
        self.assertIn("scf.for", device)
        self.assertIn("memref<2x3xi32>", device)


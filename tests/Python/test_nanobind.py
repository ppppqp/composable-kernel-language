"""Native binding ownership and context lifetime checks."""

import gc
import os
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

import ckl
from ckl.cuda import _native
from test_frontend import step


class NanobindTest(unittest.TestCase):
    def setUp(self):
        try:
            self.native = _native()
        except RuntimeError as error:
            self.skipTest(str(error))

    def test_builder_does_not_parse_operation_text(self):
        from mlir import ir

        with patch.object(ir.Module, "parse", side_effect=AssertionError("text parser")):
            lowered = step.lower(
                signature={
                    "src": ckl.BufferSpec((4,), ckl.f32),
                    "dst": ckl.BufferSpec((4,), ckl.f32),
                    "bias": ckl.f32,
                }
            )
        self.assertIn('"ckl.dispatch"', lowered.mlir)

    def test_program_context_closes_on_exception(self):
        toolkit = os.environ.get("CKL_CUDA_TOOLKIT")
        if not toolkit:
            self.skipTest("CUDA toolkit not configured")
        lowered = step.lower(
            signature={
                "src": ckl.BufferSpec((4,), ckl.f32),
                "dst": ckl.BufferSpec((4,), ckl.f32),
                "bias": ckl.f32,
            }
        )
        program = lowered.compile(target=ckl.NVIDIATarget("sm_120", "+ptx87", Path(toolkit)))
        with self.assertRaisesRegex(ValueError, "body failure"), program as opened:
            self.assertIs(opened, program)
            self.assertFalse(program.closed)
            raise ValueError("body failure")
        self.assertTrue(program.closed)
        program.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = program.host_ir
        with self.assertRaisesRegex(RuntimeError, "closed"), program:
            pass

    def test_bound_invocation_retains_closed_parents_and_imported_storage(self):
        if not self.native.cuda_available():
            self.skipTest("CUDA device unavailable")
        toolkit = os.environ.get("CKL_CUDA_TOOLKIT")
        if not toolkit:
            self.skipTest("CUDA toolkit not configured")
        spec = ckl.BufferSpec((4,), ckl.f32)
        lowered = step.lower(signature={"src": spec, "dst": spec, "bias": ckl.f32})
        with ckl.Device() as device:
            owner = device.empty(spec)
            owner.copy_from_host(array("f", [1, 2, 3, 4]))
            source = device.import_buffer(owner.address, spec, owner=owner)
            output = device.empty(spec)
            destination = device.import_buffer(output.address, spec, owner=output)
            with lowered.compile(
                target=ckl.NVIDIATarget("sm_120", "+ptx87", Path(toolkit)), fuse_pointwise=False
            ) as program:
                invocation = program.bind(source, destination, 0.5)
            owner.close()
            source.close()
            destination.close()
        gc.collect()
        with invocation:
            for mode in ("ordinary", "capture", "graph"):
                invocation.run(mode=mode)
                result = array("f", [0] * 4)
                output.copy_to_host(result)
                self.assertEqual(list(result), [2, 3, 4, 5])
            self.assertEqual(invocation.node_count, 2)
            invocation.benchmark(iterations=2, graph=True)
        self.assertTrue(invocation.closed)
        invocation.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            invocation.run()
        output.close()


if __name__ == "__main__":
    unittest.main()

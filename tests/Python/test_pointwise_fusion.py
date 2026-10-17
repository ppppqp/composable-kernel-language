"""Structural fusion checks, including cases where boundaries must survive."""
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

import ckl
from test_frontend import step

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/standalone"))
from workloads import CASES


class PointwiseFusionTest(unittest.TestCase):
    def setUp(self):
        self.optimizer = Path(os.environ.get("CKL_OPT", str(
            Path(__file__).resolve().parents[2] / "build/tools/ckl-opt/ckl-opt")))
        if not self.optimizer.is_file():
            self.skipTest("CKL MLIR tools are not built")
        self.source = step.lower(signature={"src": ckl.BufferSpec((257,), ckl.f32),
            "dst": ckl.BufferSpec((257,), ckl.f32), "bias": ckl.f32}).mlir

    def run_pass(self, source, *passes, error=None):
        result = subprocess.run([str(self.optimizer), *passes], input=source,
                                text=True, capture_output=True)
        if error:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def fused(self, source):
        return self.run_pass(source, "--ckl-fuse-pointwise")

    def test_two_stage_scalarization_and_determinism(self):
        fused = self.fused(self.source)
        self.assertEqual(fused, self.fused(self.source))
        self.assertEqual(fused.count('"ckl.dispatch"'), 1)
        self.assertEqual(fused.count('"ckl.load_element"'), 1)
        self.assertEqual(fused.count('"ckl.store_element"'), 1)
        self.assertEqual(fused.count("arith.addf"), 2)
        self.assertIn("ckl.fusion.requires_disjoint", fused)
        self.assertIn("ckl.fused_steps = 2", fused)
        self.assertNotIn('"ckl.alloc"', fused)
        self.assertNotIn('"ckl.free"', fused)
        self.assertEqual(self.fused(fused), fused)
        executable = self.run_pass(fused, "--ckl-prepare-cuda", "--ckl-build-graph",
                                   "--ckl-lower-graph-to-exec")
        self.assertEqual(executable.count('"ckl_exec.kernel"'), 1)
        self.assertNotIn('"ckl_exec.heap"', executable)

    def test_complete_chain_but_not_branches_or_stencils(self):
        for name, definition in CASES.items():
            signature = {"src": ckl.BufferSpec((257,), ckl.f32),
                         "dst": ckl.BufferSpec((257,), ckl.f32)}
            if name == "fork_join": signature["other"] = signature["src"]
            text = self.fused(definition.lower(signature=signature).mlir)
            expected = {"single_affine": 1, "chain_six": 1, "fused_chain_six": 1,
                        "fork_join": 6, "stencil_two": 2}[name]
            with self.subTest(case=name):
                self.assertEqual(text.count('"ckl.dispatch"'), expected)
                if name == "chain_six":
                    self.assertIn("ckl.fused_steps = 6", text)
                    self.assertNotIn('"ckl.alloc"', text)

    def test_launch_and_guard_mismatches_are_not_fused(self):
        mismatches = [
            self.source.replace("block = array<i64: 256, 1, 1>",
                                "block = array<i64: 128, 1, 1>", 1),
            self.source.replace("grid = array<i64: 2, 1, 1>",
                                "grid = array<i64: 2, 2, 1>"),
            self.source.replace("shared_memory = 0 : i64", "shared_memory = 16 : i64"),
            self.source.replace("arith.cmpi slt", "arith.cmpi sle"),
            self.source.replace("arith.constant 257 : index", "arith.constant 256 : index"),
        ]
        for source in mismatches:
            with self.subTest(source=source):
                self.assertEqual(self.fused(source).count('"ckl.dispatch"'), 2)

    def test_escaping_temporary_and_effect_barrier(self):
        free = re.search(r'    "ckl.free"\(([^)]+)\)[^\n]*', self.source).group(0)
        escaped = self.source.replace(free, free.replace('"ckl.free"', '"ckl.view"')
            .replace('-> ()', '-> memref<257xf32>') + "\n" + free)
        self.assertEqual(self.fused(escaped).count('"ckl.dispatch"'), 2)
        dispatches = [m.start() for m in re.finditer(r'    %[^\n]*"ckl.dispatch"', self.source)]
        barrier = '    "ckl.sync"() {scope = "device"} : () -> ()\n'
        source = self.source[:dispatches[1]] + barrier + self.source[dispatches[1]:]
        self.assertEqual(self.fused(source).count('"ckl.dispatch"'), 2)

    def test_fast_math_and_prepared_modules(self):
        # Restrict the attribute to addf operations while preserving MLIR syntax.
        fast = re.sub(r'(arith.addf [^\n]*?)( : f32)', r'\1 fastmath<contract>\2', self.source)
        self.assertEqual(self.fused(fast).count('"ckl.dispatch"'), 2)
        prepared = self.run_pass(self.source, "--ckl-prepare-cuda")
        self.run_pass(prepared, "--ckl-fuse-pointwise", error="before CUDA preparation")

    def test_multiple_consumers_and_explicit_tokens(self):
        dispatches = re.findall(r'    %[^\n]*"ckl.dispatch"[^\n]*', self.source)
        extra = re.sub(r'    %[^ ]+ =', '    %extra =', dispatches[1])
        free = self.source.index('    "ckl.free"')
        source = self.source[:free] + extra + "\n" + self.source[free:]
        self.assertEqual(self.fused(source).count('"ckl.dispatch"'), 3)
        token = re.match(r'    (%[^ ]+) =', dispatches[0]).group(1)
        dependent = re.sub(r'("ckl.dispatch"\([^)]*)\)', r'\1, ' + token + ')', dispatches[1])
        dependent = dependent.replace('array<i32: 3, 0>', 'array<i32: 3, 1>')
        dependent = dependent.replace('f32) -> !ckl.token', 'f32, !ckl.token) -> !ckl.token')
        self.assertEqual(self.fused(self.source.replace(dispatches[1], dependent)).count('"ckl.dispatch"'), 2)


if __name__ == "__main__":
    unittest.main()

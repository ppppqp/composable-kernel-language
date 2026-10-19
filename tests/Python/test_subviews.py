"""Subview regions must agree between effects, dependency edges, and pointer packing."""
import os
from pathlib import Path
import re
import subprocess
import unittest

import ckl

N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]


@ckl.kernel
def fill(dst: Vector, value: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < N:
        dst[i] = value


@ckl.kernel
def copy(src: Vector, dst: Vector) -> None:
    i = ckl.global_id(0)
    if i < N:
        dst[i] = src[i]


@ckl.program
def partitions(storage: Vector, output: Vector) -> None:
    middle = ckl.subview(storage, offset=2, shape=(12,))
    left = ckl.subview(middle, offset=0, shape=(4,))
    right = ckl.subview(middle, offset=4, shape=(4,))
    overlap = ckl.subview(middle, offset=3, shape=(4,))
    ckl.launch(fill, left, 1.0, grid=(1,), block=(32,))
    ckl.launch(fill, right, 2.0, grid=(1,), block=(32,))
    ckl.launch(fill, overlap, 3.0, grid=(1,), block=(32,))
    ckl.launch(copy, storage, output, grid=(1,), block=(32,))


@ckl.program
def temporary_partitions(output: Vector) -> None:
    scratch = ckl.empty((16,), dtype=ckl.f32)
    left = ckl.subview(scratch, offset=0, shape=(8,))
    right = ckl.subview(scratch, offset=8, shape=(8,))
    ckl.launch(fill, left, 1.0, grid=(1,), block=(32,))
    ckl.launch(fill, right, 2.0, grid=(1,), block=(32,))
    ckl.launch(copy, scratch, output, grid=(1,), block=(32,))


class SubviewTest(unittest.TestCase):
    def setUp(self):
        self.optimizer = Path(os.environ.get("CKL_OPT", "build/tools/ckl-opt/ckl-opt"))
        if not self.optimizer.is_file():
            self.skipTest("CKL optimizer not built")
        spec = ckl.BufferSpec((16,), ckl.f32)
        self.lowered = partitions.lower(signature=dict(storage=spec, output=spec))

    def analyze(self, *passes):
        return self.lowered.analyze(*passes, ckl_opt=self.optimizer).mlir

    def test_nested_offsets_disjoint_and_overlap_dependencies(self):
        graph = self.analyze("--ckl-build-graph")
        edges = {(int(a), int(b)) for a, b in re.findall(
            r'"ckl.graph_edge".*?from = (\d+).*?to = (\d+)', graph)}
        self.assertNotIn((0, 1), edges)
        self.assertTrue({(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)} <= edges)
        self.assertIn('length = 16 : i64, offset = 8 : i64', graph)
        self.assertIn('length = 16 : i64, offset = 24 : i64', graph)
        executable = self.analyze("--ckl-prepare-cuda", "--ckl-build-graph", "--ckl-lower-graph-to-exec")
        for offset in (8, 20, 24):
            self.assertIn(f'byte_offset = {offset} : i64', executable)
        self.assertIn('byte_length = 16 : i64', executable)
        self.assertNotIn('unsupported_view', executable)

    def test_temporary_subviews_preserve_base_lifetimes(self):
        result = temporary_partitions.lower(signature=dict(output=ckl.BufferSpec((16,), ckl.f32)))
        executable = result.analyze('--ckl-prepare-cuda', '--ckl-build-graph',
                                    '--ckl-lower-graph-to-exec', ckl_opt=self.optimizer).mlir
        self.assertEqual(executable.count('"ckl_exec.heap"'), 1)
        self.assertEqual(executable.count('kind = "temporary"'), 1)
        self.assertIn('byte_offset = 32 : i64', executable)
        self.assertIn('dependencies = array<i64: 1, 2>', executable)

    def test_pointer_ranges_verified_before_host_lowering(self):
        executable = self.analyze("--ckl-prepare-cuda", "--ckl-build-graph", "--ckl-lower-graph-to-exec")
        for replacement in ('byte_offset = -1 : i64', 'byte_offset = 64 : i64'):
            source = executable.replace('byte_offset = 8 : i64', replacement)
            result = subprocess.run([str(self.optimizer), '-'], input=source, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('resource byte range', result.stderr)

    def test_invalid_source_bounds_are_rejected(self):
        @ckl.program
        def bad(storage: Vector) -> None:
            view = ckl.subview(storage, offset=15, shape=(2,))
            ckl.launch(fill, view, 1.0, grid=(1,), block=(32,))
        with self.assertRaisesRegex(ckl.FrontendError, 'within the source'):
            bad.lower(signature=dict(storage=ckl.BufferSpec((16,), ckl.f32)))
        source = self.lowered.mlir.replace('offset = 2 : i64', 'offset = 12 : i64')
        result = subprocess.run([str(self.optimizer), '-'], input=source, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('out of bounds', result.stderr)

    def test_interprocedural_region_and_external_alias_conservatism(self):
        source = '''module {
          func.func @write(%a: memref<4xf32>) attributes {ckl.kernel} {
            %i = arith.constant 1 : index
            %v = arith.constant 1.0 : f32
            "ckl.store_element"(%v, %a, %i) : (f32, memref<4xf32>, index) -> ()
            return
          }
          func.func @wrapper(%a: memref<16xf32>) {
            %v = "ckl.subview"(%a) {offset = 3 : i64} : (memref<16xf32>) -> memref<4xf32>
            func.call @write(%v) : (memref<4xf32>) -> ()
            return
          }
        }'''
        result = subprocess.run([str(self.optimizer), '-', '--ckl-summarize-effects'],
                                input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r'@wrapper.*?length = 4 : i64, offset = 16 : i64')
        # Separate formal arguments can be overlapping views at binding time.
        @ckl.program
        def aliases(a: Vector, b: Vector) -> None:
            x = ckl.subview(a, offset=0, shape=(4,))
            y = ckl.subview(b, offset=8, shape=(4,))
            ckl.launch(fill, x, 1.0, grid=(1,), block=(32,))
            ckl.launch(fill, y, 2.0, grid=(1,), block=(32,))
        spec = ckl.BufferSpec((16,), ckl.f32)
        graph = aliases.lower(signature=dict(a=spec, b=spec)).analyze(
            '--ckl-build-graph', ckl_opt=self.optimizer).mlir
        self.assertIn('"ckl.graph_edge"', graph)


if __name__ == '__main__':
    unittest.main()

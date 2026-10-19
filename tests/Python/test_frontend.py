from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path

import ckl
from ckl import FrontendError

N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]
BLOCK = 256
SIDE_EFFECTS = []


def unsafe():
    SIDE_EFFECTS.append("executed")
    return 1


@ckl.kernel
def update(src: Vector, dst: Vector, bias: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < src.shape[0]:
        value = src[i] + bias
        dst[i] = value


@ckl.program
def step(src: Vector, dst: Vector, bias: ckl.f32) -> None:
    temporary = ckl.empty_like(src)
    grid = (ckl.cdiv(src.shape[0], BLOCK),)
    ckl.launch(update, src, temporary, bias, grid=grid, block=(BLOCK,))
    ckl.launch(update, temporary, dst, bias, grid=grid, block=(BLOCK,))


@ckl.kernel
def constant_update(src: Vector, dst: Vector, bias: ckl.constexpr[ckl.f32]) -> None:
    i = ckl.global_id(0)
    if i < N:
        dst[i] = src[i] + bias


@ckl.program
def constant_step(src: Vector, dst: Vector, bias: ckl.constexpr[ckl.f32]) -> None:
    ckl.launch(constant_update, src, dst, bias,
               grid=(ckl.cdiv(N, 256),), block=(256,))


@ckl.kernel
def affine(src: ckl.Buffer[ckl.i32, (4,)], dst: ckl.Buffer[ckl.i32, (4,)]) -> None:
    i = ckl.thread_id(0)
    if i < 4:
        value = src[i] * 2 + 1
        dst[i] = value


@ckl.program
def affine_step(src: ckl.Buffer[ckl.i32, (4,)], dst: ckl.Buffer[ckl.i32, (4,)]) -> None:
    ckl.launch(affine, src, dst, grid=(1,), block=(4,))


class FrontendTest(unittest.TestCase):
    def signature(self, size=257):
        return {"src": ckl.BufferSpec((size,), ckl.f32),
                "dst": ckl.BufferSpec((size,), ckl.f32), "bias": ckl.f32}

    def optimizer(self):
        path = Path(os.environ.get("CKL_OPT", str(
            Path(__file__).resolve().parents[2] / "build/tools/ckl-opt/ckl-opt")))
        if not path.is_file():
            self.skipTest("CKL MLIR tools are not built")
        return path

    def test_pipeline_lowering_and_specialization(self):
        result = step.lower(signature=self.signature())
        self.assertEqual(result.mlir, step.lower(signature=self.signature()).mlir)
        self.assertEqual(result.cache_key, step.lower(signature=self.signature()).cache_key)
        self.assertNotEqual(result.cache_key, step.lower(signature=self.signature(513)).cache_key)
        self.assertEqual(len(result.specializations), 1)
        self.assertEqual(result.mlir.count('"ckl.dispatch"'), 2)
        self.assertEqual(result.mlir.count('"ckl.alloc"'), 1)
        self.assertEqual(result.mlir.count('"ckl.free"'), 1)
        self.assertIn("memref<257xf32>", result.mlir)
        self.assertIn("grid = array<i64: 2, 1, 1>", result.mlir)
        self.assertIn("operandSegmentSizes = array<i32: 3, 0>", result.mlir)
        self.assertIn("scf.if", result.mlir)
        self.assertIn('device = "unassigned:0"', result.mlir)
        self.assertIn("capabilities = []", result.mlir)
        self.assertIn("test_frontend.py", result.mlir)
        self.assertEqual(result.diagnostics, ())

    def test_generated_ir_effects_graph_and_fail_closed_execution(self):
        result = step.lower(signature=self.signature(), verify=True, ckl_opt=self.optimizer())
        analysis = result.analyze("--ckl-build-graph", ckl_opt=self.optimizer()).mlir
        self.assertIn('effects = ["read"]', analysis)
        self.assertIn('effects = ["write"]', analysis)
        self.assertIn('kind = "memory"', analysis)
        self.assertRegex(analysis, r'from = 1 : i64, reasons = .*?to = 2 : i64')
        self.assertIn('status = "unavailable"', analysis)
        self.assertIn('only cuda:<ordinal> devices are supported', analysis)
        from ckl.compiler import CompilationError
        with self.assertRaises(CompilationError):
            result.analyze("--ckl-build-graph", "--ckl-lower-graph-to-exec", ckl_opt=self.optimizer())

    def test_constexpr_removed_from_signature_and_changes_identity(self):
        signature = self.signature()
        del signature["bias"]
        result = constant_step.lower(signature=signature, constants={"bias": 1.5})
        self.assertEqual(len(result.signature.runtime), 2)
        self.assertEqual(len(result.specializations[0].signature.runtime), 2)
        self.assertIn("operandSegmentSizes = array<i32: 2, 0>", result.mlir)
        self.assertNotEqual(result.cache_key,
            constant_step.lower(signature=signature, constants={"bias": 2.0}).cache_key)
        result.verify(ckl_opt=self.optimizer())

    def test_integer_literals_have_contextual_types(self):
        result = affine_step.lower(signature={
            "src": ckl.BufferSpec((4,), ckl.i32), "dst": ckl.BufferSpec((4,), ckl.i32)})
        self.assertIn("arith.muli", result.mlir)
        self.assertIn("arith.addi", result.mlir)
        result.verify(ckl_opt=self.optimizer())

    def test_source_locations_do_not_change_semantic_cache_identity(self):
        moved = replace(step, filename='relocated/quoted"source.py', first_line=1000)
        original = step.lower(signature=self.signature())
        result = moved.lower(signature=self.signature())
        self.assertNotEqual(original.mlir, result.mlir)
        self.assertEqual(original.cache_key, result.cache_key)
        result.verify(ckl_opt=self.optimizer())

    def test_contextual_literal_range_and_runtime_branch_updates(self):
        @ckl.kernel
        def overflow(src: ckl.Buffer[ckl.i32, (4,)], dst: ckl.Buffer[ckl.i32, (4,)]) -> None:
            dst[0] = src[0] + 2147483648

        @ckl.kernel
        def change(src: ckl.Buffer[ckl.i32, (4,)], dst: ckl.Buffer[ckl.i32, (4,)]) -> None:
            i = ckl.global_id(0)
            value = src[0]
            if i < 4:
                value = src[i]
            dst[0] = value

        for kernel, message in ((overflow, "does not fit i32"), (change, "runtime branch results")):
            @ckl.program
            def entry(src: ckl.Buffer[ckl.i32, (4,)], dst: ckl.Buffer[ckl.i32, (4,)]) -> None:
                ckl.launch(kernel, src, dst, grid=(1,), block=(4,))

            with self.subTest(kernel=kernel.name), self.assertRaisesRegex(FrontendError, message):
                entry.lower(signature={"src": ckl.BufferSpec((4,), ckl.i32),
                                       "dst": ckl.BufferSpec((4,), ckl.i32)})

    def test_casts_else_and_rank_zero_access_verify(self):
        @ckl.kernel
        def scalar(src: ckl.Buffer[ckl.i32, ()], dst: ckl.Buffer[ckl.f64, ()]) -> None:
            i = ckl.cast(ckl.global_id(0), ckl.i64)
            if i == 0:
                dst[()] = ckl.cast(src[()], ckl.f64) / 2.0
            else:
                dst[()] = -1.0

        @ckl.program
        def entry(src: ckl.Buffer[ckl.i32, ()], dst: ckl.Buffer[ckl.f64, ()]) -> None:
            ckl.launch(scalar, src, dst, grid=(1,), block=(1,))

        result = entry.lower(signature={"src": ckl.BufferSpec((), ckl.i32),
                                       "dst": ckl.BufferSpec((), ckl.f64)})
        self.assertIn("arith.index_cast", result.mlir)
        self.assertIn("arith.sitofp", result.mlir)
        self.assertIn("arith.divf", result.mlir)
        self.assertIn("} else {", result.mlir)
        result.verify(ckl_opt=self.optimizer())

    def test_dimension_rank_and_scalar_mismatches(self):
        cases = [
            ({**self.signature(), "dst": ckl.BufferSpec((258,), ckl.f32)}, "dimension N conflicts"),
            ({**self.signature(), "dst": ckl.BufferSpec((257, 1), ckl.f32)}, "buffer type/rank mismatch"),
            ({**self.signature(), "bias": ckl.i32}, "scalar type mismatch"),
            ({"src": self.signature()["src"], "dst": self.signature()["dst"]}, "missing or misplaced"),
            ({**self.signature(), "surprise": ckl.i32}, "unknown specialization arguments"),
        ]
        for signature, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(FrontendError, message):
                step.lower(signature=signature)

    def test_definition_bodies_and_python_callbacks_do_not_execute(self):
        @ckl.kernel
        def bad(src: Vector, dst: Vector) -> None:
            value = unsafe()
            dst[0] = value

        @ckl.program
        def entry(src: Vector, dst: Vector) -> None:
            ckl.launch(bad, src, dst, grid=(1,), block=(1,))

        with self.assertRaisesRegex(FrontendError, "unsupported captured binding"):
            entry.lower(signature={"src": ckl.BufferSpec((4,), ckl.f32),
                                   "dst": ckl.BufferSpec((4,), ckl.f32)})
        self.assertEqual(SIDE_EFFECTS, [])
        with self.assertRaisesRegex(FrontendError, "not Python callables"):
            update(None, None, 1.0)
        with self.assertRaisesRegex(FrontendError, "only valid inside"):
            ckl.global_id(0)

    def test_launch_validation_and_program_arithmetic(self):
        @ckl.program
        def bad_grid(src: Vector, dst: Vector, bias: ckl.f32) -> None:
            ckl.launch(update, src, dst, bias, grid=(0,), block=(256,))

        @ckl.program
        def computed(src: Vector, dst: Vector, bias: ckl.f32) -> None:
            ckl.launch(update, src, dst, bias + 1.0, grid=(1,), block=(256,))

        @ckl.program
        def direct(src: Vector, dst: Vector, bias: ckl.f32) -> None:
            update(src, dst, bias)

        for definition, message in ((bad_grid, "positive compile-time"),
                                    (computed, "computed runtime host scalars"),
                                    (direct, "use ckl.launch")):
            with self.subTest(definition=definition.name), self.assertRaisesRegex(FrontendError, message):
                definition.lower(signature=self.signature())

    def test_captured_constants_are_snapshotted(self):
        width = 32

        @ckl.program
        def local(src: Vector, dst: Vector, bias: ckl.f32) -> None:
            ckl.launch(update, src, dst, bias, grid=(1,), block=(width,))

        width = 64
        self.assertIn("block = array<i64: 32, 1, 1>", local.lower(signature=self.signature()).mlir)

    def test_annotation_calls_are_never_evaluated(self):
        @ckl.program
        def bad(src: unsafe()) -> None:
            pass

        with self.assertRaisesRegex(FrontendError, "unsupported annotation"):
            bad.lower(signature={"src": ckl.f32})
        self.assertEqual(SIDE_EFFECTS, [])

    def test_branch_definite_assignment_and_unsupported_statements(self):
        @ckl.kernel
        def branch(src: Vector, dst: Vector) -> None:
            i = ckl.global_id(0)
            if i < src.shape[0]:
                value = src[i]
            dst[i] = value

        @ckl.kernel
        def loop(src: Vector, dst: Vector) -> None:
            while True:
                dst[0] = src[0]

        @ckl.kernel
        def negative(src: Vector, dst: Vector) -> None:
            dst[0] = src[-1]

        for kernel, message in ((branch, "not defined on this path"),
                                (loop, "unsupported statement: While"),
                                (negative, "out of bounds")):
            @ckl.program
            def entry(src: Vector, dst: Vector) -> None:
                ckl.launch(kernel, src, dst, grid=(1,), block=(1,))

            with self.subTest(kernel=kernel.name), self.assertRaisesRegex(FrontendError, message) as error:
                entry.lower(signature={"src": ckl.BufferSpec((4,), ckl.f32),
                                       "dst": ckl.BufferSpec((4,), ckl.f32)})
            self.assertRegex(str(error.exception), r"test_frontend.py:\d+:\d+")
            self.assertIn("^", str(error.exception))

    def test_static_program_branch_and_empty_shape(self):
        @ckl.program
        def local(src: Vector, dst: Vector, bias: ckl.f32) -> None:
            if src.shape[0] > 256:
                temporary = ckl.empty(src.shape, dtype=src.dtype)
                ckl.launch(update, src, temporary, bias, grid=(2,), block=(256,))
                ckl.launch(update, temporary, dst, bias, grid=(2,), block=(256,))
            else:
                ckl.launch(update, src, dst, bias, grid=(1,), block=(256,))

        self.assertEqual(local.lower(signature=self.signature()).mlir.count('"ckl.dispatch"'), 2)
        self.assertEqual(local.lower(signature=self.signature(128)).mlir.count('"ckl.dispatch"'), 1)
        with self.assertRaisesRegex(ValueError, "positive extents"):
            ckl.BufferSpec((0,), ckl.f32)
        with self.assertRaisesRegex(ValueError, "identity-layout"):
            ckl.BufferSpec((4,), ckl.f32, layout="strided")


if __name__ == "__main__":
    unittest.main()

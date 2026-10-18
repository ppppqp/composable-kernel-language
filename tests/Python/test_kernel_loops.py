"""Kernel loop/state semantics and explicit program specialization loops."""
import os
from pathlib import Path
import unittest
import ckl

N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]


@ckl.kernel
def reduce(src: Vector, dst: Vector, stop: ckl.i32) -> None:
    end = ckl.cast(stop, ckl.index)
    total = 0.0
    best = src[0]
    for i in range(end):
        value = src[i]
        total = total + value
        best = ckl.select(value > best, value, best)
    dst[0] = total
    dst[1] = ckl.exp(best) * ckl.rsqrt(best * best + 1.0)


@ckl.program
def reduction(src: Vector, dst: Vector, stop: ckl.i32) -> None:
    ckl.launch(reduce, src, dst, stop, grid=(1,), block=(1,))


class KernelLoopsTest(unittest.TestCase):
    def signature(self):
        return dict(src=ckl.BufferSpec((4,), ckl.f32), dst=ckl.BufferSpec((4,), ckl.f32))

    def verify(self, program):
        optimizer = Path(os.environ.get("CKL_OPT", "build/tools/ckl-opt/ckl-opt"))
        if optimizer.is_file():
            program.verify(ckl_opt=optimizer)

    def test_dynamic_multi_carried_loop_and_math_verify(self):
        result = reduction.lower(signature=dict(self.signature(), stop=ckl.i32))
        self.assertIn("iter_args", result.mlir)
        self.assertIn("-> (f32, f32)", result.mlir)
        self.assertIn("math.exp", result.mlir)
        self.assertIn("math.rsqrt", result.mlir)
        self.verify(result)

    def test_program_unrolls_without_running_python(self):
        @ckl.program
        def stages(src: Vector, dst: Vector) -> None:
            for stage in range(2):
                ckl.launch(reduce, src, dst, stage, grid=(1,), block=(1,))

        result = stages.lower(signature=self.signature())
        self.assertEqual(result.mlir.count('"ckl.dispatch"'), 2)
        # The only device loop is in the specialized kernel body.
        self.assertEqual(result.mlir.count("scf.for"), 1)
        self.verify(result)

    def test_loop_local_does_not_escape_zero_trip_loop(self):
        @ckl.kernel
        def bad(src: Vector, dst: Vector) -> None:
            for i in range(0):
                value = src[i]
            dst[0] = value

        @ckl.program
        def entry(src: Vector, dst: Vector) -> None:
            ckl.launch(bad, src, dst, grid=(1,), block=(1,))

        with self.assertRaisesRegex(ckl.FrontendError, "not defined on this path"):
            entry.lower(signature=self.signature())

    def test_invalid_step_and_induction_mutation(self):
        @ckl.kernel
        def bad_step(src: Vector, dst: Vector) -> None:
            for i in range(0, 4, 0):
                dst[i] = src[i]

        @ckl.kernel
        def bad_induction(src: Vector, dst: Vector) -> None:
            for i in range(4):
                i = 1
                dst[i] = src[i]

        for kernel, error in ((bad_step, "positive compile-time"),
                              (bad_induction, "induction variable")):
            @ckl.program
            def entry(src: Vector, dst: Vector) -> None:
                ckl.launch(kernel, src, dst, grid=(1,), block=(1,))
            with self.assertRaisesRegex(ckl.FrontendError, error):
                entry.lower(signature=self.signature())

    def test_runtime_program_loop_rejected(self):
        @ckl.program
        def bad(src: Vector, dst: Vector, stop: ckl.i32) -> None:
            for i in range(stop):
                ckl.launch(reduce, src, dst, stop, grid=(1,), block=(1,))

        with self.assertRaisesRegex(ckl.FrontendError, "compile-time integers"):
            bad.lower(signature=dict(self.signature(), stop=ckl.i32))


if __name__ == "__main__":
    unittest.main()

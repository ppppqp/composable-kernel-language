"""GPU semantic checks for zero-trip/multiple-carried loops, math, and execution modes."""
from array import array
import argparse
import math
from pathlib import Path
import ckl

Vector = ckl.Buffer[ckl.f64, (4,)]


@ckl.kernel
def statistics(src: Vector, dst: Vector, count: ckl.i32) -> None:
    if ckl.global_id(0) == 0:
        total = src[0] * 0.0
        maximum = src[0]
        winner = count * 0
        for i in range(ckl.cast(count, ckl.index)):
            value = src[i]
            wins = value > maximum
            winner = ckl.select(wins, ckl.cast(i, ckl.i32), winner)
            maximum = ckl.select(wins, value, maximum)
            total = total + value
        dst[0] = total
        dst[1] = ckl.exp(maximum) * ckl.rsqrt(maximum * maximum + 1.0)
        dst[2] = ckl.cast(winner, ckl.f64)


@ckl.program
def entry(src: Vector, dst: Vector, count: ckl.i32) -> None:
    ckl.launch(statistics, src, dst, count, grid=(1,), block=(1,))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--chip", default="sm_120")
    args = parser.parse_args()
    device = ckl.Device()
    spec = ckl.BufferSpec((4,), ckl.f64)
    src, dst = device.empty(spec), device.empty(spec)
    src.copy_from_host(array("d", [1., 0.25, 2., -0.5]))
    program = entry.lower(signature=dict(src=spec, dst=spec, count=ckl.i32)).compile(
        target=ckl.NVIDIATarget(args.chip, "+ptx87", args.toolkit), fuse_pointwise=False)
    for count, total, maximum, winner in ((0, 0., 1., 0), (4, 2.75, 2., 2)):
        bound = program.bind(src, dst, count)
        for mode in ("capture", "ordinary", "graph"):
            dst.copy_from_host(array("d", [-12345.] * 4))
            bound.run(mode=mode, batch_size=3)
            actual = array("d", [0.] * 4)
            dst.copy_to_host(actual)
            expected = [total, math.exp(maximum) / math.sqrt(maximum * maximum + 1), winner, -12345.]
            assert all(math.isclose(x, y, abs_tol=1e-12, rel_tol=1e-12)
                       for x, y in zip(actual, expected)), (count, mode, actual, expected)
        try:
            bound.run(mode="graph", batch_size=1000001)
        except ValueError as error:
            assert "one million" in str(error)
        else:
            raise AssertionError("oversized repeated plan accepted")
    print("zero-trip and multi-carried f64 loops/math, capture/inferred repetition passed")


if __name__ == "__main__":
    main()

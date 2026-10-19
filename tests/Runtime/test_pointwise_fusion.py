"""Numerical fusion, overlap fallback, bounds, and rounding checks."""
from array import array
import argparse
from pathlib import Path
import sys

import ckl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from vector_pipeline import step

N = ckl.dim("N")
Vector = ckl.Buffer[ckl.f32, (N,)]


@ckl.kernel
def multiply(src: Vector, dst: Vector, factor: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < N:
        dst[i] = src[i] * factor


@ckl.kernel
def shift(src: Vector, dst: Vector, bias: ckl.f32) -> None:
    i = ckl.global_id(0)
    if i < N:
        dst[i] = src[i] + bias


@ckl.program
def rounding(src: Vector, dst: Vector, factor: ckl.f32, bias: ckl.f32) -> None:
    temp = ckl.empty_like(src)
    ckl.launch(multiply, src, temp, factor, grid=(ckl.cdiv(N, 256),), block=(256,))
    ckl.launch(shift, temp, dst, bias, grid=(ckl.cdiv(N, 256),), block=(256,))


parser = argparse.ArgumentParser()
parser.add_argument("--toolkit", type=Path, required=True)
parser.add_argument("--chip", required=True)
args = parser.parse_args()
target = ckl.NVIDIATarget(args.chip, "+ptx87", args.toolkit)
spec = ckl.BufferSpec((257,), ckl.f32)
program = step.lower(signature={"src": spec, "dst": spec, "bias": ckl.f32}).compile(target=target)
assert program.executable_ir.count('"ckl_exec.kernel"') == 1
assert '"ckl_exec.heap"' not in program.executable_ir
from ckl.cuda import _native
if not _native().cuda_available():
    print("CUDA unavailable; fused compilation passed")
    sys.exit(77)
device = ckl.Device()
for n in (257, 1048576):
    spec = ckl.BufferSpec((n,), ckl.f32)
    module = step.lower(signature={"src": spec, "dst": spec, "bias": ckl.f32})
    fused = module.compile(target=target)
    baseline = module.compile(target=target, fuse_pointwise=False)
    src, dst = device.empty(spec), device.empty(spec)
    initial = array("f", ((i % 97) / 128 for i in range(n)))
    actual = array("f", [0.]) * n
    expected = array("f", (v + 0.25 for v in initial))
    for compiled, nodes in ((fused, 1), (baseline, 2)):
        src.copy_from_host(initial)
        invocation = compiled.bind(src, dst, 0.125)
        assert invocation.node_count == nodes
        for graph in (False, True):
            invocation.run(graph=graph)
            dst.copy_to_host(actual)
            assert actual == expected

    # Shifted overlapping imports must preserve the original kernel boundary.
    owner = device.empty(ckl.BufferSpec((n + 2,), ckl.f32))
    overlap_src = device.import_buffer(owner.address, spec, owner=owner)
    overlap_dst = device.import_buffer(owner.address + 4, spec, owner=owner)
    initial_full = array("f", initial)
    initial_full.extend([-12345., -12345.])
    wanted = array("f", [initial[0]])
    wanted.extend(expected)
    wanted.append(-12345.)
    overlap = fused.bind(overlap_src, overlap_dst, 0.125)
    assert overlap.node_count == 2
    for graph in (False, True):
        owner.copy_from_host(initial_full)
        overlap.run(graph=graph)
        full = array("f", [0.]) * (n + 2)
        owner.copy_to_host(full)
        assert full == wanted, "shifted overlap changed kernel ordering"

    # Adjacent disjoint slices of one owner may use the fused topology.
    adjacent_owner = device.empty(ckl.BufferSpec((2 * n,), ckl.f32))
    adjacent_src = device.import_buffer(adjacent_owner.address, spec, owner=adjacent_owner)
    adjacent_dst = device.import_buffer(adjacent_owner.address + n * 4, spec, owner=adjacent_owner)
    adjacent_src.copy_from_host(initial)
    adjacent = fused.bind(adjacent_src, adjacent_dst, 0.125)
    assert adjacent.node_count == 1
    adjacent.run(graph=True)
    adjacent_dst.copy_to_host(actual)
    assert actual == expected

    exact = fused.bind(src, src, 0.125)
    assert exact.node_count == 2
    src.copy_from_host(initial)
    exact.run(graph=True)
    src.copy_to_host(actual)
    assert actual == expected

# Without a store/load boundary, multiply-add contraction would change zero
# into -(2**-46). Fusion must retain the original f32 rounding semantics.
spec = ckl.BufferSpec((257,), ckl.f32)
p = rounding.lower(signature={"src": spec, "dst": spec, "factor": ckl.f32,
                              "bias": ckl.f32}).compile(target=target)
src, dst = device.empty(spec), device.empty(spec)
src.copy_from_host(array("f", [1 + 2**-23]) * 257)
invocation = p.bind(src, dst, 1 - 2**-23, -1.)
assert invocation.node_count == 1
for graph in (False, True):
    invocation.run(graph=graph)
    actual = array("f", [1.]) * 257
    dst.copy_to_host(actual)
    assert actual.tobytes() == (array("f", [0.]) * 257).tobytes(), "cross-kernel contraction changed rounding"
print("fusion=true,overlap_fallback=true,adjacent_slices=true,rounding=true,large=true,correct=true")

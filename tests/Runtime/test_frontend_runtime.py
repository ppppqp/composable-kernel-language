"""Numerically validate Python -> in-process compiler -> host JIT -> CUDA."""
from array import array
import argparse
import ctypes
import gc
from pathlib import Path
import sys
from unittest.mock import patch

import ckl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from vector_pipeline import lower

LargeN = ckl.dim("LargeN")
LargeVector = ckl.Buffer[ckl.f32, (LargeN,)]


@ckl.kernel
def large_update(src: LargeVector, dst: LargeVector) -> None:
    i = ckl.global_id(0)
    if i < LargeN:
        dst[i] = src[i] + 0.125


@ckl.program
def large_step(src: LargeVector, dst: LargeVector) -> None:
    ckl.launch(large_update, src, dst, grid=(ckl.cdiv(LargeN, 256),), block=(256,))


@ckl.kernel
def integer_update(src: ckl.Buffer[ckl.i64, (4,)], dst: ckl.Buffer[ckl.i64, (4,)],
                   bias: ckl.i64) -> None:
    i = ckl.global_id(0)
    if i < 4:
        dst[i] = src[i] + bias


@ckl.program
def integer_step(src: ckl.Buffer[ckl.i64, (4,)], dst: ckl.Buffer[ckl.i64, (4,)],
                 bias: ckl.constexpr[ckl.i64]) -> None:
    ckl.launch(integer_update, src, dst, bias, grid=(1,), block=(4,))


@ckl.kernel
def double_update(src: ckl.Buffer[ckl.f64, (4,)], dst: ckl.Buffer[ckl.f64, (4,)],
                  bias: ckl.f64) -> None:
    i = ckl.global_id(0)
    if i < 4:
        dst[i] = src[i] + bias


@ckl.program
def double_step(src: ckl.Buffer[ckl.f64, (4,)], dst: ckl.Buffer[ckl.f64, (4,)],
                bias: ckl.f64) -> None:
    ckl.launch(double_update, src, dst, bias, grid=(1,), block=(4,))

parser = argparse.ArgumentParser()
parser.add_argument("--toolkit", type=Path, required=True)
parser.add_argument("--chip", required=True)
args = parser.parse_args()

# Calling Python subprocess tools would regress the in-process API.
with patch("subprocess.run", side_effect=AssertionError("compiler spawned a subprocess")):
    compiled = lower().compile(target=ckl.NVIDIATarget(args.chip, "+ptx87", args.toolkit),
                               fuse_pointwise=False)
assert "@ckl_entry" in compiled.host_ir
assert "@ckl.binary" in compiled.host_ir
assert 'dependencies = array<i64: 1>' in compiled.executable_ir
assert compiled.executable_ir.count('"ckl_exec.heap"') == 1

cuda = ctypes.CDLL("libcuda.so.1")
devices = ctypes.c_int()
if cuda.cuInit(0) or cuda.cuDeviceGetCount(ctypes.byref(devices)) or not devices.value:
    print("CUDA device unavailable; in-process compilation passed")
    sys.exit(77)

device = ckl.Device()
full_spec = ckl.BufferSpec((512,), ckl.f32)
spec = ckl.BufferSpec((257,), ckl.f32)
input_owner, output_owner = device.empty(full_spec), device.empty(full_spec)
src = device.import_buffer(input_owner.address, spec, owner=input_owner)
dst = device.import_buffer(output_owner.address, spec, owner=output_owner)
initial = array("f", [index * 0.25 for index in range(257)] + [-12345.] * 255)
blank = array("f", [-12345.] * 512)
actual = array("f", blank)
input_owner.copy_from_host(initial)
for bias in (0.125, -0.5):
    bound = compiled.bind(src, dst, bias)
    for graph in (False, True, True, True):
        output_owner.copy_from_host(blank)
        bound.run(graph=graph)
        output_owner.copy_to_host(actual)
        expected = array("f", [value + 2 * bias for value in initial[:257]] + [-12345.] * 255)
        assert actual == expected, "incorrect output or tail guard"
        input_owner.copy_to_host(actual)
        assert actual == initial, "input changed"

    for graph in (False, True):
        burst = bound.benchmark(iterations=3, graph=graph)
        assert burst["stream_us"] > 0 and burst["wall_us"] > 0
    output_owner.copy_to_host(actual)
    assert actual == expected

# The same compiled code can bind through distinct runtime ownership objects.
other_device = ckl.Device()
other_src, other_dst = other_device.empty(spec), other_device.empty(spec)
other_src.copy_from_host(array("f", initial[:257]))
other_bound = compiled.bind(other_src, other_dst, 0.125)
other_bound.run(graph=True)
other_result = array("f", [0.] * 257)
other_dst.copy_to_host(other_result)
assert other_result == array("f", [value + 0.25 for value in initial[:257]])
try:
    compiled.bind(src, other_dst, 0.125)
except ValueError:
    pass
else:
    raise AssertionError("mixed runtime buffer ownership was accepted")
del other_bound, other_src, other_dst, other_device

# Wrong signatures and scalar types must fail before entering the native ABI.
for arguments in ((src, dst), (src, dst, "bad"), (input_owner, dst, 1.0)):
    try:
        compiled.bind(*arguments)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid arguments were accepted")

# The bound invocation retains code, buffers, their original owners, and device.
bound = compiled.bind(src, dst, 0.125)
del compiled, src, dst, input_owner, device
gc.collect()
output_owner.copy_from_host(blank)
bound.run(graph=True)
output_owner.copy_to_host(actual)
assert actual[:257] == array("f", [v + 0.25 for v in initial[:257]])

# Exercise byte-exact constants and 64-bit scalars, including negative values
# outside i32 and double precision differences that f32 would lose.
device = output_owner.device
target = bound.program.target
for definition, dtype, code, bias in (
    (integer_step, ckl.i64, "q", -(2**40 + 3)),
    (double_step, ckl.f64, "d", 2**-42),
):
    spec = ckl.BufferSpec((4,), dtype)
    source, destination = device.empty(spec), device.empty(spec)
    values = array(code, [0, 1, 2, 3])
    source.copy_from_host(values)
    signature = {"src": spec, "dst": spec}
    constants = {"bias": bias} if dtype == ckl.i64 else {}
    if dtype == ckl.f64:
        signature["bias"] = dtype
    executable = definition.lower(signature=signature, constants=constants).compile(target=target)
    arguments = (source, destination) if constants else (source, destination, bias)
    invocation = executable.bind(*arguments)
    for graph in (False, True):
        invocation.run(graph=graph)
        result = array(code, [0] * 4)
        destination.copy_to_host(result)
        assert list(result) == [value + bias for value in values]
# Pageable copies must complete before our nonblocking execution stream reads
# inputs or writes over a poisoned destination. Small transfers hid this race.
count = 1048576
spec = ckl.BufferSpec((count,), ckl.f32)
large = large_step.lower(signature={"src": spec, "dst": spec}).compile(target=target)
large_src, large_dst = device.empty(spec), device.empty(spec)
large_bound = large.bind(large_src, large_dst)
for value, graph in ((0.25, False), (0.5, True), (0.75, False)):
    large_src.copy_from_host(array("f", [value]) * count)
    large_dst.copy_from_host(array("f", [-12345.]) * count)
    large_bound.run(graph=graph)
    result = array("f", [0.]) * count
    large_dst.copy_to_host(result)
    assert result == array("f", [value + 0.125]) * count, "large copy/launch ordering failed"
print("elements=257,nodes=2,biases=2,graph_replays=8,boundary_guard=true,lifetimes=true,"
      "i64_constants=true,f64_scalars=true,large_copy_ordering=true,correct=true")

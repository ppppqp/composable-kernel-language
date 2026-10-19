"""CUDA subview pointers, bounds, and owner lifetimes."""
from array import array
import argparse
import gc
from pathlib import Path
import sys

import ckl
from ckl.cuda import _native

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Python'))
from test_subviews import partitions, temporary_partitions, copy, Vector


@ckl.program
def transfer(src: Vector, dst: Vector) -> None:
    ckl.launch(copy, src, dst, grid=(1,), block=(32,))


def read(buffer, size=16):
    host = array('f', [0.] * size)
    buffer.copy_to_host(host)
    return list(host)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toolkit', type=Path, required=True)
    parser.add_argument('--chip', default='sm_120')
    args = parser.parse_args()
    target = ckl.NVIDIATarget(args.chip, '+ptx87', args.toolkit)
    spec = ckl.BufferSpec((16,), ckl.f32)
    program = partitions.lower(signature=dict(storage=spec, output=spec)).compile(
        target=target, fuse_pointwise=False)
    if not _native().cuda_available():
        print('CUDA unavailable; subview compilation passed')
        return 77
    device = ckl.Device()
    owner = device.empty(ckl.BufferSpec((32,), ckl.f32))
    # Use a nested runtime view too: compiled subviews must be relative to this pointer.
    storage = owner.subview(offset=4, shape=(20,)).subview(offset=0, shape=(16,))
    output = device.empty(spec)
    expected = [-99., -99., 1., 1., 1., 3., 3., 3., 3., 2.] + [-99.] * 6
    bound = program.bind(storage, output)
    for mode in ('ordinary', 'graph', 'capture'):
        owner.copy_from_host(array('f', [-99.] * 32))
        output.copy_from_host(array('f', [-1.] * 16))
        bound.run(mode=mode, batch_size=3)
        assert read(output) == expected, (mode, read(output), expected)
        actual = read(owner, 32)
        assert actual[:4] == [-99.] * 4 and actual[20:] == [-99.] * 12, actual
        assert actual[4:20] == expected

    scratch_program = temporary_partitions.lower(signature=dict(output=spec)).compile(
        target=target, fuse_pointwise=False)
    scratch_bound = scratch_program.bind(output)
    scratch_program.close()
    scratch_bound.run(mode='graph', batch_size=3)
    assert read(output) == [1.] * 8 + [2.] * 8
    scratch_bound.close()

    try: storage.subview(offset=15, shape=(2,))
    except ValueError: pass
    else: raise AssertionError('out-of-bounds runtime view accepted')

    # Imported allocation ownership must survive explicit wrapper closure and GC.
    allocation = device.empty(ckl.BufferSpec((20,), ckl.f32))
    allocation.copy_from_host(array('f', [0.] * 20))
    borrowed = device.import_buffer(allocation.address, spec, owner=allocation)
    transfer_program = transfer.lower(signature=dict(src=spec, dst=spec)).compile(
        target=target, fuse_pointwise=False)
    result = device.empty(spec)
    consumer = transfer_program.bind(borrowed, result)
    allocation.close()
    borrowed.close()
    transfer_program.close()
    device.close()
    del allocation, borrowed
    gc.collect()
    for mode in ('ordinary', 'graph', 'capture'):
        consumer.run(mode=mode)
        assert read(result) == [0.] * 16
    consumer.close()
    print('Nested subview offsets/tails, temporary subviews, runtime bounds, and owner closure passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Shared correctness, paired measurement, and reporting for application components.

Requires CUDA PyTorch and NumPy. CKL uses borrowed PyTorch allocations so both
implementations have identical resident buffers and state ownership. This is
benchmark instrumentation, not a new CKL runtime API.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
import ckl

TAIL = 256
POISON = np.float32(-12345)
WD, MOMENTUM, LR = 0.01, 0.9, 0.01
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def timed(action):
    start = time.perf_counter_ns()
    result = action()
    return result, (time.perf_counter_ns() - start) / 1e6


def summary(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                samples=values)


def plan_stats(ir):
    kernels = [line for line in ir.splitlines() if '"ckl_exec.kernel"' in line]
    topology = {}
    for line in kernels:
        node = int(re.search(r'\bid = (\d+)', line).group(1))
        deps = re.search(r'dependencies = array<i64(?:: ([\d, ]+))?>', line).group(1)
        topology[node] = [int(value) for value in deps.split(",")] if deps else []
    temporaries = [line for line in ir.splitlines() if '"ckl_exec.resource"' in line
                   and 'kind = "temporary"' in line]
    heaps = [line for line in ir.splitlines() if '"ckl_exec.heap"' in line]
    return dict(nodes=len(kernels), topology=topology, temporary_count=len(temporaries),
                logical_temp_bytes=sum(int(re.search(r'\bbytes = (\d+)', line).group(1))
                                       for line in temporaries),
                planned_temp_bytes=sum(int(re.search(r'\bbytes = (\d+)', line).group(1))
                                       for line in heaps))


def load_workloads(path):
    spec = importlib.util.spec_from_file_location("application_workloads", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sgd_reference(p, g, v, steps):
    """Independent f32 recurrence, including materialized-operation rounding."""
    p, v = p.copy(), v.copy()
    for _ in range(steps):
        adjusted = g + np.float32(WD) * p
        v *= np.float32(MOMENTUM)
        v += adjusted
        p += v * np.float32(-LR)
    return p, v


class Buffers:
    def __init__(self, device, arrays, shapes, mutable):
        self.initial = arrays
        self.shapes = shapes
        self.mutable = mutable
        self.owners, self.saved, self.tensors, self.ckl = {}, {}, {}, {}
        for name, values in arrays.items():
            host = np.full(values.size + TAIL, POISON, dtype=np.float32)
            host[:values.size] = values.ravel()
            owner = torch.from_numpy(host).to("cuda")
            self.owners[name] = owner
            self.saved[name] = owner.clone() if name in mutable else None
            self.tensors[name] = owner[:values.size].view(shapes[name])
            self.ckl[name] = device.import_buffer(
                owner.data_ptr(), ckl.BufferSpec(shapes[name], ckl.f32), owner=owner)
        torch.cuda.synchronize()

    def reset(self):
        for name in self.mutable:
            self.owners[name].copy_(self.saved[name])
        # CKL owns a nonblocking stream: finish reset before either implementation.
        torch.cuda.synchronize()

    def verify(self, expected):
        torch.cuda.synchronize()
        max_error = 0.
        for name, owner in self.owners.items():
            actual = owner.cpu().numpy()
            size = self.initial[name].size
            if not np.all(actual[size:] == POISON):
                raise AssertionError(f"{name}: tail sentinel overwritten")
            values = actual[:size].reshape(self.shapes[name])
            if name in self.mutable:
                reference = expected[name]
                error = np.abs(values - reference)
                max_error = max(max_error, float(error.max(initial=0)))
                if not np.all(np.isfinite(values)) or not np.allclose(
                        values, reference, rtol=3e-5, atol=3e-6):
                    raise AssertionError(f"{name}: numerical mismatch; max error {error.max()}")
            elif not np.array_equal(values, self.initial[name]):
                raise AssertionError(f"{name}: immutable input changed")
        return max_error


class CKLBackend:
    def __init__(self, label, definition, signature, buffers, arguments, target, fusion):
        self.label = label
        lowered, lower_ms = timed(lambda: definition.lower(signature=signature))
        program, compile_ms = timed(lambda: lowered.compile(target=target, fuse_pointwise=fusion))
        self.bound, bind_ms = timed(lambda: program.bind(*arguments))
        self.metadata = dict(kind="ckl", lower_ms=lower_ms, compile_ms=compile_ms,
                             bind_ms=bind_ms, **plan_stats(program.executable_ir))
        self.metadata["bound_nodes"] = self.bound.node_count
        self.metadata["ir_sha256"] = hashlib.sha256(lowered.mlir.encode()).hexdigest()
        expected_nodes = (1 if label == "ckl_manual" else 2 if label == "ckl_copy_elided" else
                          4 if "p" in buffers.mutable else 3)
        if self.bound.node_count != expected_nodes:
            raise AssertionError(f"unexpected {label} topology: {self.bound.node_count}")

    def run(self, graph):
        self.bound.run(graph=graph)

    def setup_graph(self):
        return self.bound.benchmark(iterations=1, graph=True)

    def burst(self, iterations, graph):
        return self.bound.benchmark(iterations=iterations, graph=graph)


class TorchBackend:
    def __init__(self, label, action):
        self.label, self.action, self.graph = label, action, None
        self.metadata = dict(kind="pytorch")

    def run(self, graph):
        self.graph.replay() if graph else self.action()
        torch.cuda.synchronize()

    def setup_graph(self):
        # Warmup must be off the capture stream; compile and initialize beforehand.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.action()
            self.action()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.action()
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter_ns() - start) / 1e6
        return dict(graph_setup_ms=setup_ms)

    def burst(self, iterations, graph):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        action = self.graph.replay if graph else self.action
        torch.cuda.synchronize()
        start.record()
        wall = time.perf_counter_ns()
        for _ in range(iterations):
            action()
        end.record()
        end.synchronize()
        return dict(stream_us=start.elapsed_time(end) * 1000 / iterations,
                    wall_us=(time.perf_counter_ns() - wall) / 1000 / iterations)


def torch_backends(kind, buffers, include_compile):
    t = buffers.tensors
    backends = []
    if kind == "momentum-sgd":
        for name, foreach, fused in (("torch_eager", False, False),
                                      ("torch_foreach", True, False),
                                      ("torch_fused", False, True)):
            optimizer = torch.optim.SGD([t["p"]], lr=LR, momentum=MOMENTUM,
                                        weight_decay=WD, foreach=foreach, fused=fused)
            t["p"].grad = t["g"]
            optimizer.state[t["p"]]["momentum_buffer"] = t["v"]
            backends.append(TorchBackend(name, optimizer.step))

        def computation():
            adjusted = t["g"] + WD * t["p"]
            t["v"].mul_(MOMENTUM).add_(adjusted)
            t["p"].add_(t["v"], alpha=-LR)
    else:
        mean, std = t["mean"].view(1, 3, 1), t["std"].view(1, 3, 1)

        def eager():
            # Preallocated clone equivalent: same copy/sub/div device work, no allocation timing.
            t["dst"].copy_(t["src"])
            t["dst"].sub_(mean).div_(std)

        def computation():
            t["dst"].copy_((t["src"] - mean) / std)

        def copy_elided():
            torch.sub(t["src"], mean, out=t["dst"])
            t["dst"].div_(std)

        backends.append(TorchBackend("torch_eager", eager))
        backends.append(TorchBackend("torch_copy_elided", copy_elided))
    if include_compile:
        action = torch.compile(computation, fullgraph=True, options={"triton.cudagraphs": False})
        backends.append(TorchBackend("torch_compiled", action))
    return backends


def make_case(kind, shape, device):
    if kind == "momentum-sgd":
        n = shape[0]
        index = np.arange(n, dtype=np.int32)
        p = ((index % 97 - 48).astype(np.float32) / 64)
        g = ((index % 113 - 56).astype(np.float32) / 512)
        v = ((index % 71 - 35).astype(np.float32) / 1024)
        arrays = dict(p=p, g=g, v=v)
        shapes = {name: shape for name in arrays}
        buffers = Buffers(device, arrays, shapes, ("p", "v"))
        signature = {name: ckl.BufferSpec(shape, ckl.f32) for name in arrays}
        signature.update(wd=ckl.f32, momentum=ckl.f32, lr=ckl.f32)
        arguments = [buffers.ckl[name] for name in arrays] + [WD, MOMENTUM, LR]

        def expected(steps):
            p_new, v_new = sgd_reference(p, g, v, steps)
            return dict(p=p_new, v=v_new)

        return buffers, signature, arguments, expected
    b, h, w = shape
    actual_shape = (b, 3, h * w)
    n = int(np.prod(actual_shape))
    src = ((np.arange(n, dtype=np.int32) % 251).astype(np.float32) / 250).reshape(actual_shape)
    arrays = dict(src=src, dst=np.full(actual_shape, POISON, dtype=np.float32),
                  mean=np.array(MEAN, dtype=np.float32), std=np.array(STD, dtype=np.float32))
    shapes = {name: values.shape for name, values in arrays.items()}
    buffers = Buffers(device, arrays, shapes, ("dst",))
    signature = {name: ckl.BufferSpec(shapes[name], ckl.f32) for name in arrays}
    arguments = [buffers.ckl[name] for name in arrays]
    result = (src - arrays["mean"].reshape(1, 3, 1)) / arrays["std"].reshape(1, 3, 1)
    return buffers, signature, arguments, lambda steps: dict(dst=result)


def measure_case(kind, shape, module, device, target, args):
    buffers, signature, arguments, expected = make_case(kind, shape, device)
    backends = [CKLBackend(label, definition, signature, buffers, arguments, target, fusion)
                for label, definition, fusion in (("ckl_off", module.staged, False),
                                                  ("ckl_on", module.staged, True),
                                                  ("ckl_manual", module.manual, False))]
    if kind == "image-preprocessing":
        backends.append(CKLBackend("ckl_copy_elided", module.copy_elided, signature,
                                   buffers, arguments, target, False))
    backends += torch_backends(kind, buffers, args.torch_compile)
    # Initialize/compile all variants and construct graphs before correctness and timings.
    for backend in backends:
        buffers.reset()
        _, initialize_ms = timed(lambda: backend.run(False))
        setup = backend.setup_graph()
        backend.metadata.update(first_execution_ms=initialize_ms,
                                graph_setup_ms=setup["graph_setup_ms"])
    # Same reset, one complete step, and a recurrence check for each mode/backend.
    errors = {backend.label: 0. for backend in backends}
    for count in (1, 17):
        reference = expected(count)
        for backend in backends:
            for graph in (False, True):
                buffers.reset()
                for _ in range(count):
                    backend.run(graph)
                errors[backend.label] = max(errors[backend.label], buffers.verify(reference))
        del reference
    for backend in backends:
        for graph in (False, True):
            buffers.reset()
            for _ in range(args.warmup):
                backend.run(graph)
    # Paired samples rotate backend order and alternate ordinary/graph ordering.
    samples = {backend.label: {mode: {key: [] for key in
               ("stream_us", "wall_us", "python_latency_us")} for mode in ("ordinary", "graph")}
               for backend in backends}
    for sample in range(args.samples):
        order = backends[sample % len(backends):] + backends[:sample % len(backends)]
        for backend in order:
            for graph in ((False, True) if sample % 2 == 0 else (True, False)):
                mode = "graph" if graph else "ordinary"
                buffers.reset()
                burst = backend.burst(args.iterations, graph)
                for key in ("stream_us", "wall_us"):
                    samples[backend.label][mode][key].append(burst[key])
                buffers.reset()
                start = time.perf_counter_ns()
                for _ in range(args.latency_iterations):
                    backend.run(graph)
                samples[backend.label][mode]["python_latency_us"].append(
                    (time.perf_counter_ns() - start) / 1000 / args.latency_iterations)
    # Repeat measured sequences outside the timing matrix and validate all outputs.
    # The f32 reference is evaluated once per count, not once per backend/sample.
    for count in sorted({args.iterations, args.latency_iterations}):
        reference = expected(count)
        for backend in backends:
            for graph in (False, True):
                buffers.reset()
                backend.burst(count, graph)
                errors[backend.label] = max(errors[backend.label], buffers.verify(reference))
        del reference
    rows = []
    for backend in backends:
        row = dict(backend=backend.label, correct=True, max_abs_error=errors[backend.label],
                   metadata=backend.metadata, timings={mode: {key: summary(values)
                   for key, values in metrics.items()} for mode, metrics in samples[backend.label].items()})
        rows.append(row)
        print(f"{kind} {shape} {backend.label:15s} "
              f"ordinary={row['timings']['ordinary']['stream_us']['median']:.3f} us "
              f"graph={row['timings']['graph']['stream_us']['median']:.3f} us "
              f"python={row['timings']['ordinary']['python_latency_us']['median']:.3f} us correct=true",
              flush=True)
    return dict(shape=shape, elements=int(np.prod(shape)) * (3 if kind == "image-preprocessing" else 1),
                validated_counts=sorted({1, 17, args.iterations, args.latency_iterations}), results=rows)


def report(data):
    if not data.get("complete") or len(data["cases"]) != len(data["case_shapes"]):
        raise ValueError("refusing to report an incomplete matrix")
    for case, shape in zip(data["cases"], data["case_shapes"]):
        labels = {row["backend"] for row in case["results"]}
        if list(case["shape"]) != list(shape) or labels != set(data["backends"]) or len(labels) != len(case["results"]):
            raise ValueError("incomplete or duplicate backend/shape matrix")
        for row in case["results"]:
            if not row["correct"]:
                raise ValueError("refusing to report failed correctness checks")
            for metrics in row["timings"].values():
                if any(len(metric["samples"]) != data["configuration"]["samples"]
                       or not all(np.isfinite(value) and value > 0 for value in metric["samples"])
                       for metric in metrics.values()):
                    raise ValueError("incomplete or invalid timing samples")
    lines = [f"# {data['workload']} measurements", "",
             f"GPU: {data['device']}; PyTorch {data['torch']}; Python {data['python']}.", "",
             "Times are medians in µs per complete component/step. CKL bursts submit from C++; "
             "PyTorch bursts submit from Python. Stream elapsed times include submission gaps and "
             "must not be used as a direct cross-framework kernel-speed comparison. Synchronous "
             "Python latency uses one submission and synchronization per step for both frameworks.", "",
             "| Shape | Backend | Nodes (CKL) | Ordinary stream µs | Graph stream µs | "
             "Python ordinary µs | Python graph µs |", "|---|---|---:|---:|---:|---:|---:|"]
    for case in data["cases"]:
        for row in case["results"]:
            t = row["timings"]
            lines.append(f"| {'×'.join(map(str, case['shape']))} | {row['backend']} | "
                         f"{row['metadata'].get('bound_nodes', '—')} | "
                         f"{t['ordinary']['stream_us']['median']:.3f} | {t['graph']['stream_us']['median']:.3f} | "
                         f"{t['ordinary']['python_latency_us']['median']:.3f} | "
                         f"{t['graph']['python_latency_us']['median']:.3f} |")
    lines += ["", "Manual optimization headroom below compares CKL against CKL with identical "
              "timing boundaries. Values above 1 favor the manually fused control.", "",
              "| Shape | Ordinary off/manual | Graph off/manual | Graph on/off | Staged temp bytes |",
              "|---|---:|---:|---:|---:|"]
    for case in data["cases"]:
        rows = {row["backend"]: row for row in case["results"]}

        def value(name, mode):
            return rows[name]["timings"][mode]["stream_us"]["median"]

        lines.append(f"| {'×'.join(map(str, case['shape']))} | "
                     f"{value('ckl_off', 'ordinary') / value('ckl_manual', 'ordinary'):.2f}x | "
                     f"{value('ckl_off', 'graph') / value('ckl_manual', 'graph'):.2f}x | "
                     f"{value('ckl_on', 'graph') / value('ckl_off', 'graph'):.2f}x | "
                     f"{rows['ckl_off']['metadata']['planned_temp_bytes']:,} |")
    if data["workload"] == "image-preprocessing":
        lines += ["", "Copy-elision versus arithmetic fusion (CKL graph stream medians):", "",
                  "| Shape | Staged/copy-elided | Copy-elided/manual |",
                  "|---|---:|---:|"]
        for case in data["cases"]:
            rows = {row["backend"]: row for row in case["results"]}
            values = {key: row["timings"]["graph"]["stream_us"]["median"] for key, row in rows.items()}
            lines.append(f"| {'×'.join(map(str, case['shape']))} | "
                         f"{values['ckl_off'] / values['ckl_copy_elided']:.2f}x | "
                         f"{values['ckl_copy_elided'] / values['ckl_manual']:.2f}x |")
    lines += ["", "All variants passed numerical, immutable-input, and tail-sentinel checks for "
              "one step, 17 steps, and the measured burst/latency repeat counts. Raw JSON contains "
              "individual samples, setup/compile costs, reference hashes, topology, and errors.", "",
              "These are resident-device, warm repeated-buffer measurements with unlocked clocks. "
              "They exclude transfers and state resets. SGD uses an established momentum buffer "
              "and fixed gradients during each repeated sequence. Image preprocessing measures "
              "float RGB normalization only; resize/crop/decode/conversion are excluded. PyTorch "
              "normalization reuses a preallocated output, excluding allocation overhead.", ""]
    return "\n".join(lines)


def main(kind, workloads_path):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", required=True, type=Path)
    parser.add_argument("--chip", default="sm_120")
    parser.add_argument("--sizes", type=int, nargs="+", default=[257, 4096, 65536, 1048576, 16777216])
    parser.add_argument("--image-shapes", nargs="+", default=["1x257x259", "1x224x224", "8x224x224",
                                                            "32x224x224", "8x1080x1920"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--latency-iterations", type=int, default=10)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.iterations, args.samples, args.warmup, args.latency_iterations, *args.sizes) < 1:
        parser.error("sizes and repeat counts must be positive")
    shapes = [(n,) for n in args.sizes] if kind == "momentum-sgd" else [
        tuple(map(int, shape.split("x"))) for shape in args.image_shapes]
    if any(len(shape) != (1 if kind == "momentum-sgd" else 3) or min(shape) < 1 for shape in shapes):
        parser.error("image shapes must be positive BxHxW triples")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("these comparisons require a CUDA PyTorch installation and a CUDA GPU")
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.cuda.init()
    device = ckl.Device()
    target = ckl.NVIDIATarget(args.chip, "+ptx87", args.toolkit)
    module = load_workloads(workloads_path)
    library = Path(device._lib._name).resolve()
    cache = library.parents[2] / "CMakeCache.txt"
    build_type = next((line.split("=", 1)[1] for line in cache.read_text().splitlines()
                       if line.startswith("CMAKE_BUILD_TYPE:")), "unknown") if cache.exists() else "unknown"
    sgd_source = Path(torch.__file__).parent / "optim/sgd.py"
    data = dict(complete=False, workload=kind, timestamp=datetime.now(timezone.utc).isoformat(),
                device=device.name, torch=torch.__version__, torch_git=torch.version.git_version,
                torch_cuda=torch.version.cuda, python=platform.python_version(), platform=platform.platform(),
                native_library=str(library), native_build=build_type,
                native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                workloads_sha256=hashlib.sha256(workloads_path.read_bytes()).hexdigest(),
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                torch_sgd_source_sha256=hashlib.sha256(sgd_source.read_bytes()).hexdigest(),
                reference=dict(sgd="https://github.com/pytorch/pytorch/blob/" + torch.version.git_version +
                               "/torch/optim/sgd.py",
                               normalize="https://github.com/pytorch/vision/blob/v0.24.0/"
                               "torchvision/transforms/_functional_tensor.py",
                               mean=MEAN, std=STD,
                               weight_decay=WD, momentum=MOMENTUM, learning_rate=LR),
                case_shapes=shapes,
                backends=["ckl_off", "ckl_on", "ckl_manual", "torch_eager"] +
                         (["torch_foreach", "torch_fused"] if kind == "momentum-sgd" else []) +
                         (["ckl_copy_elided", "torch_copy_elided"] if kind == "image-preprocessing" else []) +
                         (["torch_compiled"] if args.torch_compile else []),
                target=target.cache_identity(), configuration={key: value for key, value in vars(args).items()
                if key not in ("toolkit", "output")}, cases=[])
    data["device_properties"] = str(torch.cuda.get_device_properties(0))
    info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.current.sm,"
                           "clocks.current.memory,temperature.gpu,power.limit", "--format=csv"],
                          capture_output=True, text=True)
    data["nvidia_smi"] = info.stdout if info.returncode == 0 else info.stderr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for shape in shapes:
        print(f"Checking and measuring {kind} {shape}...", flush=True)
        data["cases"].append(measure_case(kind, shape, module, device, target, args))
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        gc.collect()
    data["complete"] = True
    rendered = report(data)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(rendered)
    print(f"Saved {args.output} and {args.output.with_suffix('.md')}")

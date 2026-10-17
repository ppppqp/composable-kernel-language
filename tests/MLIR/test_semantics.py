#!/usr/bin/env python3

from __future__ import annotations

import random
import re
import subprocess
import sys
from pathlib import Path


def run(executable: Path, source: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), str(source), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def run_text(executable: Path, source: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), "-", *arguments],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_graph_edges(output: str) -> dict[tuple[int, int], set[str]]:
    edges: dict[tuple[int, int], set[str]] = {}
    for line in output.splitlines():
        if '"ckl.graph_edge"' not in line:
            continue
        source = re.search(r"\bfrom = (\d+)", line)
        target = re.search(r"\bto = (\d+)", line)
        require(source is not None and target is not None, f"malformed graph edge: {line}")
        edges[(int(source.group(1)), int(target.group(1)))] = set(
            re.findall(r'\bkind = "([^"]+)"', line)
        )
    return edges


def make_random_graph_program(
    generator: random.Random, resource_count: int, dispatch_count: int
) -> tuple[str, list[tuple[int, str, int | None]]]:
    operations: list[tuple[int, str, int | None]] = []
    for index in range(dispatch_count):
        resource = generator.randrange(resource_count)
        effect = generator.choice(("read", "write"))
        dependency = None
        if index and generator.random() < 0.35:
            dependency = generator.randrange(index)
        operations.append((resource, effect, dependency))

    lines = [
        "module {",
        "  func.func @read(%resource: memref<4xf32>) {",
        '    %value = "ckl.load"(%resource) : (memref<4xf32>) -> f32',
        "    func.return",
        "  }",
        "  func.func @write(%resource: memref<4xf32>) {",
        "    %value = arith.constant 1.0 : f32",
        '    "ckl.store"(%value, %resource) : (f32, memref<4xf32>) -> ()',
        "    func.return",
        "  }",
        "  func.func @step() {",
    ]
    for resource in range(resource_count):
        lines.append(f'    %r{resource} = "ckl.alloc"() : () -> memref<4xf32>')
    for index, (resource, effect, dependency) in enumerate(operations):
        operands = f"%r{resource}"
        signature = "memref<4xf32>"
        dependency_count = 0
        if dependency is not None:
            operands += f", %d{dependency}"
            signature += ", !ckl.token"
            dependency_count = 1
        lines.extend(
            [
                f'    %d{index} = "ckl.dispatch"({operands}) {{',
                "      block = array<i64: 32, 1, 1>, capabilities = [], device = \"cuda:0\",",
                f'      grid = array<i64: 1, 1, 1>, implementation = "{effect}.v1", kernel = @{effect},',
                f"      operandSegmentSizes = array<i32: 1, {dependency_count}>, shared_memory = 0 : i64",
                f"    }} : ({signature}) -> !ckl.token",
            ]
        )
    lines.extend(["    func.return", "  }", "}"])
    return "\n".join(lines) + "\n", operations


def check_random_graphs(executable: Path) -> None:
    seed = 20260927
    generator = random.Random(seed)
    resource_count = 3
    for case in range(20):
        program, operations = make_random_graph_program(generator, resource_count, 8)
        result = run_text(executable, program, "--ckl-build-graph")
        require(result.returncode == 0, f"random case {case}, seed {seed}:\n{result.stderr}\n{program}")

        expected: set[tuple[int, int]] = set()
        for target, (target_resource, target_effect, dependency) in enumerate(operations):
            if dependency is not None:
                expected.add((dependency, target))
            for source in range(target):
                source_resource, source_effect, _ = operations[source]
                if source_resource == target_resource and "write" in (source_effect, target_effect):
                    expected.add((source, target))

        actual = {
            (source - resource_count, target - resource_count)
            for source, target in parse_graph_edges(result.stdout)
            if source >= resource_count and target >= resource_count
        }
        require(
            actual == expected,
            f"random graph mismatch in case {case}, seed {seed}: "
            f"expected {sorted(expected)}, got {sorted(actual)}\n{program}",
        )


def check_indexed_kernels(executable: Path, inputs: Path) -> None:
    summary = run(executable, inputs / "indexed-kernel.mlir", "--ckl-summarize-effects")
    require(summary.returncode == 0, summary.stderr)
    update = summary.stdout.split("func.func @update", 1)[1].split("func.func @step", 1)[0]
    require('effects = ["read"]' in update and 'effects = ["write"]' in update,
            "guarded indexed accesses did not produce kernel effects")
    require('region = "whole"' in update and "unknown = false" in update,
            "device indices or kernel control flow lost complete effect semantics")
    coordinates = summary.stdout.split("func.func @coordinates", 1)[1]
    require('effects = ["read", "write"]' in coordinates,
            "multidimensional loop accesses lost their effects")
    graph = run(executable, inputs / "indexed-kernel.mlir", "--ckl-build-graph")
    require(graph.returncode == 0, graph.stderr)
    require("memory" in parse_graph_edges(graph.stdout).get((1, 2), set()),
            "indexed kernel composition lost the temporary write/read dependency")
    plan = run(executable, inputs / "indexed-kernel.mlir",
               "--ckl-build-graph", "--ckl-lower-graph-to-exec")
    require(plan.returncode == 0, plan.stderr)
    require(plan.stdout.count('"ckl_exec.kernel"') == 2,
            "indexed kernels did not reach executable plan lowering")

    def kernel(body: str, buffer: str = "memref<4xf32>", marker: str = "ckl.kernel") -> str:
        return f'''module {{
  func.func @test(%buffer: {buffer}) attributes {{{marker}}} {{
    %zero = arith.constant 0 : index
    {body}
    func.return
  }}
}}'''

    invalid_cases = [
        (kernel('%i = "ckl.global_id"() {axis = 3 : i32} : () -> index'),
         "requires axis 0, 1, or 2"),
        (kernel('%i = "ckl.thread_id"() {axis = -1 : i32} : () -> index'),
         "requires axis 0, 1, or 2"),
        (kernel('%i = "ckl.global_id"() {axis = 0 : i32} : () -> index', marker=""),
         "requires an enclosing func.func marked ckl.kernel"),
        (kernel('%v = "ckl.load_element"(%buffer) : (memref<4xf32>) -> f32'),
         "requires one index per buffer dimension"),
        (kernel('%v = "ckl.load_element"(%buffer, %zero) : (memref<4xf32>, index) -> i32'),
         "value type must match the buffer element type"),
        (kernel('''%v = arith.constant 1 : i32
    "ckl.store_element"(%v, %buffer, %zero) : (i32, memref<4xf32>, index) -> ()'''),
         "value type must match the buffer element type"),
        (kernel('''%past = arith.constant 4 : index
    %v = "ckl.load_element"(%buffer, %past) : (memref<4xf32>, index) -> f32'''),
         "constant index is out of bounds for dimension 0"),
        (kernel('''%past = arith.constant 3 : index
    %v = "ckl.load_element"(%buffer, %zero, %past)
      : (memref<2x3xf32>, index, index) -> f32''', buffer="memref<2x3xf32>"),
         "constant index is out of bounds for dimension 1"),
        (kernel('''%negative = arith.constant -1 : index
    %v = arith.constant 1.0 : f32
    "ckl.store_element"(%v, %buffer, %negative) : (f32, memref<?xf32>, index) -> ()''',
                buffer="memref<?xf32>"),
         "constant index is out of bounds for dimension 0"),
        (kernel("", buffer="memref<*xf32>"),
         "ckl.kernel arguments must be numeric scalars or ranked numeric memrefs"),
        (kernel("", marker="ckl.kernel = true"),
         "ckl.kernel requires a unit attribute on func.func"),
        ('module { func.func private @test() attributes {ckl.kernel} }',
         "ckl.kernel requires a defined function with no results"),
        ('''module { func.func @test() -> i32 attributes {ckl.kernel} {
            %v = arith.constant 0 : i32
            func.return %v : i32
        } }''', "ckl.kernel requires a defined function with no results"),
        ('''module {
          func.func @leaf() attributes {ckl.kernel} { func.return }
          func.func @parent() attributes {ckl.kernel} {
            %done = "ckl.dispatch"() {
              kernel = @leaf, grid = array<i64: 1, 1, 1>, block = array<i64: 1, 1, 1>,
              shared_memory = 0 : i64, device = "cuda:0", capabilities = [],
              implementation = "leaf", operandSegmentSizes = array<i32: 0, 0>
            } : () -> !ckl.token
            func.return
          }
        }''', "cannot launch a kernel from a ckl.kernel body"),
    ]
    for source, diagnostic in invalid_cases:
        result = run_text(executable, source)
        require(result.returncode != 0, f"invalid kernel was accepted:\n{source}")
        require(diagnostic in result.stderr, result.stderr)

    # Rank-zero buffers have no indices; dynamic sizes are valid kernel analysis inputs.
    for source in (
        kernel('%v = "ckl.load_element"(%buffer) : (memref<f32>) -> f32', buffer="memref<f32>"),
        kernel('%v = "ckl.load_element"(%buffer, %zero) : (memref<?xf32>, index) -> f32',
               buffer="memref<?xf32>"),
    ):
        result = run_text(executable, source, "--ckl-summarize-effects")
        require(result.returncode == 0, result.stderr)


def main() -> int:
    executable = Path(sys.argv[1])
    inputs = Path(sys.argv[2])
    check_indexed_kernels(executable, inputs)

    effects = run(executable, inputs / "effects.mlir", "--ckl-summarize-effects")
    require(effects.returncode == 0, effects.stderr)
    require('effects = ["read"]' in effects.stdout, "missing inferred read")
    require('effects = ["write"]' in effects.stdout, "missing inferred write")
    require('effects = ["read", "write"]' in effects.stdout, "view did not retain base identity")
    require("may_alias = [[0, 1]]" in effects.stdout, "external arguments must conservatively alias")
    require(
        "func.func @private_allocation() attributes {ckl.effect_summary = {accesses = []" in effects.stdout,
        "private allocation leaked into launch-level effects",
    )
    require(effects.stdout.count('effects = ["read"]') >= 2, "callee summary was not propagated")
    upstream = effects.stdout.split("func.func @upstream_interfaces", 1)[1]
    require('effects = ["read"]' in upstream, "upstream read interface was not consumed")
    require('effects = ["write"]' in upstream, "upstream write interface was not consumed")

    dispatch = run(executable, inputs / "dispatch.mlir", "--ckl-summarize-effects")
    require(dispatch.returncode == 0, dispatch.stderr)
    require(dispatch.stdout.count('effects = ["read"]') >= 2, "dispatch read was not propagated")
    require(
        dispatch.stdout.count('effects = ["read", "write"]') >= 2,
        "atomic dispatch effects were not propagated",
    )
    require(
        dispatch.stdout.count("synchronizes = true") == 2,
        "synchronization behavior was not propagated",
    )
    require(
        dispatch.stdout.count('ordering_scopes = ["acq_rel@device", "sync@device"]') == 2,
        "atomic ordering and synchronization scope were not propagated",
    )

    strict = run(
        executable,
        inputs / "unknown.mlir",
        "--allow-unregistered-dialect",
        "--ckl-summarize-effects",
    )
    require(strict.returncode != 0, "strict mode accepted an unknown operation")
    require("no complete memory-effect model" in strict.stderr, strict.stderr)

    conservative = run(
        executable,
        inputs / "unknown.mlir",
        "--allow-unregistered-dialect",
        "--ckl-summarize-effects=strict=false",
    )
    require(conservative.returncode == 0, conservative.stderr)
    require("unknown = true" in conservative.stdout, "unknown effect was not recorded")
    require('effects = ["read", "write"]' in conservative.stdout, "fallback was not conservative")
    require("synchronizes = true" in conservative.stdout, "unknown operation was not a barrier")

    invalid = run(executable, inputs / "invalid-dispatch.mlir")
    require(invalid.returncode != 0, "invalid launch dimensions passed verification")
    require("requires positive block dimensions" in invalid.stderr, invalid.stderr)

    graph = run(executable, inputs / "graph.mlir", "--ckl-build-graph")
    require(graph.returncode == 0, graph.stderr)
    require(graph.stdout.count('"ckl.graph_node"') == 7, "unexpected deterministic graph nodes")
    require("digraph" in graph.stdout, "missing DOT graph visualization")
    graph_edges = parse_graph_edges(graph.stdout)
    require((2, 3) not in graph_edges, "independent allocations received a false dependency")
    require("memory" in graph_edges.get((2, 4), set()), "missing write/read conflict")
    require("explicit" in graph_edges.get((3, 4), set()), "missing token dependency")
    require("lifetime" in graph_edges.get((4, 5), set()), "missing lifetime dependency")
    require("ssa" in graph_edges.get((0, 2), set()), "missing SSA dependency")
    require("from_location" in graph.stdout and "to_location" in graph.stdout, "missing provenance")
    require('ckl.memory_plan = {assignments =' in graph.stdout, "missing compiler memory plan")
    require('ckl.launch_grid = array<i64: 1, 1, 1>' in graph.stdout, "missing launch metadata")
    memory_plan = run(executable, inputs / "memory-plan.mlir", "--ckl-build-graph")
    require(memory_plan.returncode == 0, memory_plan.stderr)
    require('baseline_bytes = 512 : i64' in memory_plan.stdout, "unexpected memory baseline")
    require('planned_bytes = 256 : i64' in memory_plan.stdout, "ordered buffers were not reused")
    require(
        'resource = "alloc1", shares_with = ["alloc0"], slot = 0 : i64' in memory_plan.stdout,
        "missing compiler-derived reuse provenance",
    )
    assignments = re.findall(
        r'bytes = 64 : i64, heap = 0 : i64, offset = (\d+) : i64, '
        r'resource = "alloc[01]", slot = 0 : i64',
        memory_plan.stdout,
    )
    require(assignments == ["0", "0"], "reused resources did not receive the same offset")
    require(
        'status = "unavailable"' in memory_plan.stdout
        and "requires a static identity-layout integer or float memref" in memory_plan.stdout,
        "dynamic temporary did not produce an explicit planning diagnostic",
    )

    # Exercise CKL's analysis-to-executable path, including inferred dependencies and ABI slots.
    for backend, source, kernels, artifact in (
        ("cuda", inputs.parent / "Runtime" / "host-plan.mlir", 5, "kernels.cubin"),
        ("rocm", inputs.parent / "Runtime" / "hip-host-plan.mlir", 2, "kernels.hsaco"),
    ):
        executable_plan = run(
            executable, source, "--ckl-build-graph", "--ckl-lower-graph-to-exec"
        )
        require(executable_plan.returncode == 0, executable_plan.stderr)
        require('"ckl.graph"' not in executable_plan.stdout, "analysis graph was retained")
        require(
            executable_plan.stdout.count('"ckl_exec.kernel"') == kernels,
            "CKL dispatches did not become executable kernels",
        )
        require(
            "logical_index = 0" in executable_plan.stdout and "slot = 0" in executable_plan.stdout,
            "physical ABI slots were not materialized",
        )
        require(
            "dependencies = array<i64:" in executable_plan.stdout,
            "inferred dependencies did not reach the executable plan",
        )
        if backend == "rocm":
            require(
                "dependencies = array<i64: 0>" in executable_plan.stdout
                and 'packing = "bare_pointer"' in executable_plan.stdout,
                "CKL HIP lowering lost the inferred write conflict or pointer ABI",
            )
        require(artifact in executable_plan.stdout, "executable lowering lost artifact identity")

    unresolved_abi = run(
        executable,
        inputs / "graph.mlir",
        "--ckl-build-graph",
        "--ckl-lower-graph-to-exec",
    )
    require(unresolved_abi.returncode != 0, "executable lowering guessed an undeclared ABI")
    require(
        "requires complete launch and ABI metadata" in unresolved_abi.stderr,
        unresolved_abi.stderr,
    )

    conditional_graph = run(
        executable, inputs / "conditional-dispatch.mlir", "--ckl-build-graph"
    )
    require(conditional_graph.returncode == 0, conditional_graph.stderr)
    require(
        'ckl.execution_control = "unsupported_control_flow"' in conditional_graph.stdout,
        "conditional dispatch was not marked non-executable",
    )
    conditional_exec = run(
        executable,
        inputs / "conditional-dispatch.mlir",
        "--ckl-build-graph",
        "--ckl-lower-graph-to-exec",
    )
    require(conditional_exec.returncode != 0, "conditional dispatch became unconditional")
    require(
        "does not support graph nodes nested in region or CFG control flow"
        in conditional_exec.stderr,
        conditional_exec.stderr,
    )

    subview_graph = run(executable, inputs / "subview-dispatch.mlir", "--ckl-build-graph")
    require(subview_graph.returncode == 0, subview_graph.stderr)
    require(
        'packing = "unsupported_view"' in subview_graph.stdout,
        "subview ABI packing hazard was not retained",
    )
    subview_exec = run(
        executable,
        inputs / "subview-dispatch.mlir",
        "--ckl-build-graph",
        "--ckl-lower-graph-to-exec",
    )
    require(subview_exec.returncode != 0, "subview offset was discarded by executable lowering")
    require(
        "does not support view/subview resource arguments" in subview_exec.stderr,
        subview_exec.stderr,
    )
    invalid_exec = run(executable, inputs / "invalid-exec.mlir")
    require(invalid_exec.returncode != 0, "invalid executable resource placement passed")
    require("placement exceeds its heap" in invalid_exec.stderr, invalid_exec.stderr)
    unknown_graph = run(
        executable,
        inputs / "unknown-graph.mlir",
        "--allow-unregistered-dialect",
        "--ckl-build-graph=strict=false",
    )
    require(unknown_graph.returncode == 0, unknown_graph.stderr)
    unknown_edges = parse_graph_edges(unknown_graph.stdout)
    require(
        "barrier" in unknown_edges.get((2, 3), set()),
        "unknown dispatch did not order an otherwise independent dispatch",
    )
    require("unknown = true" in unknown_graph.stdout, "unknown node was not marked")

    check_random_graphs(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

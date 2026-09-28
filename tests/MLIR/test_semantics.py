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


def main() -> int:
    executable = Path(sys.argv[1])
    inputs = Path(sys.argv[2])

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

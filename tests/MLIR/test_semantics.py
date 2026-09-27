#!/usr/bin/env python3

from __future__ import annotations

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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import importlib.util
import sys
import unittest
from pathlib import Path


ANALYZER = Path(__file__).parents[2] / "benchmarks" / "milestone0" / "analyze.py"
SPEC = importlib.util.spec_from_file_location("milestone0_analyze", ANALYZER)
assert SPEC is not None and SPEC.loader is not None
analyze = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyze
SPEC.loader.exec_module(analyze)


def row(workload: str, mode: str, milliseconds: float, correct: bool = True) -> dict[str, str]:
    return {
        "workload": workload,
        "mode": mode,
        "median_ms": str(milliseconds),
        "correct": str(correct).lower(),
    }


class MilestoneZeroAnalysisTest(unittest.TestCase):
    def test_selects_only_complete_correct_speedup(self) -> None:
        rows = [
            row("multi_field", "sequential", 14.0),
            row("multi_field", "capture", 12.0),
            row("multi_field", "streams", 9.5),
            row("multi_field", "explicit", 9.0),
            row("linear_stencil", "sequential", 11.0),
            row("linear_stencil", "capture", 8.1),
            row("linear_stencil", "streams", 8.0),
            row("linear_stencil", "explicit", 8.0),
        ]
        results = analyze.summarize(rows)
        selected = analyze.select_primary(results)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.workload, "multi_field")
        self.assertAlmostEqual(selected.speedup, 12.0 / 9.0)

    def test_rejects_incorrect_or_incomplete_results(self) -> None:
        incorrect = [
            row("incorrect", "sequential", 12.0),
            row("incorrect", "capture", 12.0),
            row("incorrect", "streams", 8.0, correct=False),
            row("incorrect", "explicit", 8.0),
        ]
        incomplete = [
            row("incomplete", "sequential", 12.0),
            row("incomplete", "capture", 12.0),
            row("incomplete", "explicit", 8.0),
        ]
        self.assertIsNone(analyze.select_primary(analyze.summarize(incorrect + incomplete)))


if __name__ == "__main__":
    unittest.main()

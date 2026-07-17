"""Analyze results for exp01a."""

from __future__ import annotations

import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    runs = []
    for path in sorted(RESULTS_DIR.glob("run_*.json")):
        payload = json.loads(path.read_text())
        runs.extend(payload.get("results", []))

    successes = sum(int(run.get("success", False)) for run in runs)
    mean_seconds = sum(float(run.get("elapsed_seconds", 0.0)) for run in runs) / len(runs) if runs else 0.0
    summary = (
        "# exp01a Results Summary\n\n"
        f"Total trials: {len(runs)}\n\n"
        f"Successful trials: {successes}\n\n"
        f"Mean wall time: {mean_seconds:.2f} seconds\n"
    )
    summary_path = RESULTS_DIR / "summary.md"
    summary_path.write_text(summary)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

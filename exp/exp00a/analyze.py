"""Aggregate exp00a's append-only task ledger."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["task_key"]] = row
    return list(latest.values())


def _metrics(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]
    return {
        "expected_tasks": expected,
        "attempted_tasks": len(rows),
        "complete_tasks": len(complete),
        "infra_errors": sum(row.get("status") == "infra_error" for row in rows),
        "code_execution_successes": sum(row.get("code_execution_rate") == 1.0 for row in complete),
        "task_successes": sum(bool(row.get("task_completed")) for row in complete),
        "code_execution_rate": (
            sum(row.get("code_execution_rate") == 1.0 for row in complete) / len(complete)
            if complete
            else None
        ),
        "task_success_rate": (
            sum(bool(row.get("task_completed")) for row in complete) / len(complete)
            if complete
            else None
        ),
        "wall_seconds": sum(float(row.get("wall_seconds", 0.0)) for row in rows),
    }


def _rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def main() -> None:
    config = yaml.safe_load((REPO_ROOT / "exp/exp00a/config.yaml").read_text())
    rows = _load_rows(REPO_ROOT / config["outputs"]["ledger"])
    by_benchmark = {
        "libero": _metrics(
            [row for row in rows if row["benchmark"] == "libero"],
            int(config["scope"]["expected_libero_tasks"]),
        ),
        "robotwin": _metrics(
            [row for row in rows if row["benchmark"] == "robotwin"],
            int(config["scope"]["expected_robotwin_tasks"]),
        ),
    }

    by_suite: dict[str, dict[str, Any]] = {}
    suite_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["benchmark"] == "libero":
            suite_rows[row["suite"]].append(row)
    for suite, grouped_rows in sorted(suite_rows.items()):
        by_suite[suite] = _metrics(grouped_rows, expected=len(grouped_rows))

    manifest = {
        "experiment": "exp00a",
        "model": config["protocol"]["model"],
        "oracle": False,
        "protocol": "one independent generation and one episode per task",
        "benchmarks": by_benchmark,
        "libero_suites": by_suite,
        "tasks": sorted(rows, key=lambda row: row["task_key"]),
    }
    results_dir = REPO_ROOT / "exp/exp00a/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    lines = [
        "# exp00a Results",
        "",
        f"{config['protocol']['model']}, privileged textual state, one independent generation "
        "and one episode per task.",
        "Repository oracle code is disabled.",
        "",
        "| Benchmark | Expected | Attempted | Complete | Infra errors | Code execution | Task success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ("libero", "robotwin"):
        metrics = by_benchmark[benchmark]
        lines.append(
            f"| {benchmark} | {metrics['expected_tasks']} | {metrics['attempted_tasks']} | "
            f"{metrics['complete_tasks']} | {metrics['infra_errors']} | "
            f"{_rate(metrics['code_execution_rate'])} | {_rate(metrics['task_success_rate'])} |"
        )
    if by_suite:
        lines.extend(
            [
                "",
                "## LIBERO suites",
                "",
                "| Suite | Attempted | Code execution | Task success |",
                "|---|---:|---:|---:|",
            ]
        )
        for suite, metrics in by_suite.items():
            lines.append(
                f"| `{suite}` | {metrics['attempted_tasks']} | "
                f"{_rate(metrics['code_execution_rate'])} | {_rate(metrics['task_success_rate'])} |"
            )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Analyzed {len(rows)} task rows into {results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

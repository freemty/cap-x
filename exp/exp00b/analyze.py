"""Build exp00b metrics and the normalized manifest consumed by the viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                latest[str(row["task_key"])] = row
    return sorted(latest.values(), key=lambda row: str(row["task_key"]))


def _failure_run(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    task = str(row["task"])
    digest = hashlib.sha1(task.encode()).hexdigest()[:10]
    log_path = row.get("log_path")
    failure_type = row.get("failure_type")
    error_stage = "infrastructure"
    error_name = "InfraError"
    message = "CaP-X did not produce a completed trial directory"
    if failure_type == "unstable_scene":
        error_stage = "environment_reset"
        error_name = "UnstableSceneError"
        seeds = [attempt.get("seed") for attempt in row.get("seed_attempts", [])]
        message = f"RoboTwin could not initialize a stable scene for seeds {seeds}"
    elif failure_type == "task_timeout":
        error_stage = "execution"
        error_name = "TaskTimeout"
        message = "CaP-X terminated the task process group after its configured timeout"
    elif failure_type == "cuda_oom":
        error_stage = "environment"
        error_name = "CudaOutOfMemory"
        message = "The RoboTwin worker exhausted GPU memory"
    elif failure_type == "model_api_connection":
        error_stage = "model_api"
        error_name = "ModelApiConnectionError"
        message = "The worker could not reach the configured language-model endpoint"
    if log_path:
        message += f"; inspect {log_path}"
    finished_at = row.get("finished_at") or datetime.now(timezone.utc).isoformat()  # noqa: UP017
    return {
        "schema_version": "capx.result.v1",
        "run_id": f"exp00b-robotwin-glm-5.2-{digest}",
        "benchmark": "robotwin",
        "policy": "capx",
        "status": "failed",
        "experiment": "exp00b",
        "started_at": None,
        "finished_at": finished_at,
        "updated_at": finished_at,
        "runtime": {
            "host": config["runtime"]["host"],
            "gpu": config["runtime"]["gpu"],
            "seed_attempts": row.get("seed_attempts", []),
        },
        "config": {
            "task": task,
            "suite": None,
            "task_id": None,
            "embodiment": "aloha-agilex",
            "model": config["protocol"]["model"],
        },
        "metrics": {
            "episodes_total": 1,
            "episodes_finished": 0,
            "code_execution_rate": 0.0,
            "plan_success_rate": 0.0,
            "task_success_rate": 0.0,
            "mean_reward": 0.0,
            "mean_elapsed_seconds": row.get("wall_seconds"),
        },
        "episodes": [
            {
                "episode_id": "trial-1",
                "seed": row.get("seed", 1),
                "trial": row.get("trial", 1),
                "task": task,
                "status": "failed",
                "metrics": {
                    "code_execution_success": False,
                    "plan_success": False,
                    "task_success": False,
                    "reward": 0.0,
                    "elapsed_seconds": row.get("wall_seconds"),
                    "sandbox_rc": row.get("returncode"),
                    "num_code_blocks": None,
                    "num_regenerations": None,
                },
                "artifacts": {
                    "video": None,
                    "thumbnail": None,
                    "generated_code": None,
                    "summary": None,
                    "events": None,
                    "responses": None,
                },
                "error": {
                    "stage": error_stage,
                    "type": error_name,
                    "message": message,
                },
                "details": None,
            }
        ],
        "error": {
            "stage": error_stage,
            "type": error_name,
            "message": message,
        },
        "source": "exp00b-ledger",
        "source_path": str(config["outputs"]["ledger"]),
        "artifact_root": str(config["outputs"]["root"]),
    }


def build_manifest(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    from viewer.results import discover_runs

    rows = _load_rows(repo_root / config["outputs"]["ledger"])
    discovered = [
        item.to_dict()
        for item in discover_runs(repo_root / config["outputs"]["root"], repo_root=repo_root)
    ]
    by_task = {
        str(run.get("config", {}).get("task")): run
        for run in discovered
        if run.get("config", {}).get("task")
    }
    runs: list[dict[str, Any]] = []
    for row in rows:
        task = str(row["task"])
        run = by_task.pop(task, None)
        if run is None:
            run = _failure_run(row, config)
        else:
            run["experiment"] = "exp00b"
            run.setdefault("runtime", {}).update(
                {"host": config["runtime"]["host"], "gpu": config["runtime"]["gpu"]}
            )
            for episode in run.get("episodes", []):
                episode["seed"] = row.get("seed", 1)
                episode["task"] = task
        runs.append(run)
    runs.extend(by_task.values())
    return {"schema": "capx.result.v1", "runs": runs}


def _rate(successes: int, total: int) -> str:
    return "N/A" if total == 0 else f"{100 * successes / total:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="exp/exp00b/config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text())
    rows = _load_rows(REPO_ROOT / config["outputs"]["ledger"])
    manifest = build_manifest(REPO_ROOT, config)
    manifest_path = REPO_ROOT / config["outputs"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    attempted = len(rows)
    completed = sum(row.get("status") == "complete" for row in rows)
    infra_errors = sum(row.get("status") == "infra_error" for row in rows)
    code_successes = sum(row.get("code_execution_rate") == 1.0 for row in rows)
    task_successes = sum(bool(row.get("task_completed")) for row in rows)
    lines = [
        "# exp00b Results",
        "",
        "GLM-5.2 through CaP-X, one generation and one RoboTwin episode per task; oracle code disabled.",
        "",
        "| Task | Status | Code execution | Task success | Reward | Wall time |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['task']}` | {row.get('status')} | "
            f"{row.get('code_execution_rate', 'N/A')} | {row.get('task_completed', 'N/A')} | "
            f"{row.get('reward', 'N/A')} | {float(row.get('wall_seconds', 0.0)):.1f} s |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Attempted: {attempted}/10",
            f"- Completed trials: {completed}/10",
            f"- Infrastructure errors: {infra_errors}/10",
            f"- Code-execution rate over attempted tasks: {_rate(code_successes, attempted)}",
            f"- Task-success rate over attempted tasks: {_rate(task_successes, attempted)}",
            f"- Frontend manifest runs: {len(manifest['runs'])}",
        ]
    )
    summary_path = REPO_ROOT / config["outputs"]["summary"]
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(manifest['runs'])} frontend runs to {manifest_path}")
    print(f"Wrote quantitative summary to {summary_path}")


if __name__ == "__main__":
    main()

"""Run the exp00a language-model baseline across LIBERO and RoboTwin."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_PROMPT = """\
You are controlling a Franka Emika robot by writing Python code.
Goal: {libero_environment_goal}
Use get_all_object_poses() first to inspect the exact object keys and poses.
Use only the documented APIs. Check the scene state in code, complete the task,
and return only executable Python without prose.
"""
ROBOTWIN_PROMPT = """\
Complete the RoboTwin task: {task_description}.
Start by calling list_actors() and inspect poses or functional points as needed.
Use only the documented planner-backed APIs, check every Boolean action result,
and return only executable Python without prose.
"""


@dataclass(frozen=True)
class TaskSpec:
    benchmark: str
    task: str
    suite: str | None = None
    task_id: int | None = None

    @property
    def key(self) -> str:
        if self.suite is not None:
            return f"{self.benchmark}:{self.suite}:{self.task_id}:{self.task}"
        return f"{self.benchmark}:{self.task}"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload


def _libero_env(config: dict[str, Any]) -> dict[str, str]:
    runtime = config["runtime"]
    env = os.environ.copy()
    python_paths = [
        str(REPO_ROOT / "capx/third_party/libero_dependencies/robosuite"),
        str(REPO_ROOT / "capx/third_party/LIBERO-PRO/libero"),
        str(REPO_ROOT),
    ]
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(runtime["gpu"]),
            "MUJOCO_GL": "egl",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(python_paths),
            "ROBOT_DESCRIPTIONS_CACHE": runtime["robot_descriptions_cache"],
            "ROBOT_DESCRIPTION_COMMIT": runtime["robot_description_commit"],
        }
    )
    return env


def _robotwin_env(config: dict[str, Any]) -> dict[str, str]:
    runtime = config["runtime"]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(runtime["gpu"]),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), runtime["robotwin_root"]]),
            "ROBOTWIN_ROOT": runtime["robotwin_root"],
        }
    )
    return env


def _discover_libero(config: dict[str, Any]) -> list[TaskSpec]:
    suites = config["scope"]["libero_suites"]
    script = f"""
import json
from libero import benchmark
d = benchmark.get_benchmark_dict()
rows = []
for suite_name in {suites!r}:
    suite = d[suite_name]()
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        rows.append({{"suite": suite_name, "task_id": task_id, "task": task.name}})
print("CAPX_TASKS_JSON=" + json.dumps(rows, separators=(",", ":")))
"""
    result = subprocess.run(
        [config["runtime"]["libero_python"], "-c", script],
        cwd=REPO_ROOT,
        env=_libero_env(config),
        text=True,
        capture_output=True,
        check=True,
    )
    marker = next(
        line for line in reversed(result.stdout.splitlines()) if line.startswith("CAPX_TASKS_JSON=")
    )
    rows = json.loads(marker.removeprefix("CAPX_TASKS_JSON="))
    tasks = [TaskSpec(benchmark="libero", **row) for row in rows]
    expected = int(config["scope"]["expected_libero_tasks"])
    if len(tasks) != expected:
        raise RuntimeError(f"Expected {expected} LIBERO tasks, discovered {len(tasks)}")
    return tasks


def _discover_robotwin(config: dict[str, Any]) -> list[TaskSpec]:
    env_dir = Path(config["runtime"]["robotwin_root"]) / "envs"
    tasks = [
        TaskSpec(benchmark="robotwin", task=path.stem)
        for path in sorted(env_dir.glob("*.py"))
        if not path.name.startswith("_")
    ]
    expected = int(config["scope"]["expected_robotwin_tasks"])
    if len(tasks) != expected:
        raise RuntimeError(f"Expected {expected} RoboTwin tasks, discovered {len(tasks)}")
    return tasks


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _configured_tasks(config: dict[str, Any], tasks: list[TaskSpec]) -> list[TaskSpec]:
    """Apply an optional explicit RoboTwin task list while preserving its order."""

    selected_names = config.get("scope", {}).get("selected_robotwin_tasks")
    if selected_names is None:
        return tasks
    if not isinstance(selected_names, list) or not all(
        isinstance(name, str) and name for name in selected_names
    ):
        raise TypeError("scope.selected_robotwin_tasks must be a list of task names")
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("scope.selected_robotwin_tasks contains duplicates")

    by_name = {task.task: task for task in tasks if task.benchmark == "robotwin"}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown configured RoboTwin tasks: {missing}")
    return [by_name[name] for name in selected_names]


def _task_config(task: TaskSpec, output_dir: Path) -> dict[str, Any]:
    if task.benchmark == "libero":
        return {
            "env": {
                "_target_": "capx.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv",
                "cfg": {
                    "_target_": "capx.envs.tasks.base.CodeExecEnvConfig",
                    "apis": ["FrankaLiberoPrivilegedApi"],
                    "low_level": {
                        "_target_": "capx.envs.simulators.libero.FrankaLiberoEnv",
                        "suite_name": task.suite,
                        "task_id": task.task_id,
                        "privileged": True,
                        "max_steps": 4000,
                    },
                    "privileged": True,
                    "prompt": LIBERO_PROMPT,
                },
            },
            "api_servers": [
                {
                    "_target_": "capx.serving.launch_pyroki_server.main",
                    "port": 8116,
                    "host": "127.0.0.1",
                    "robot": "panda_description",
                    "target_link": "panda_hand",
                }
            ],
            "record_video": True,
            "output_dir": str(output_dir),
            "use_oracle_code": False,
            "trials": 1,
            "num_workers": 1,
        }

    return {
        "env": {
            "_target_": "capx.envs.tasks.robotwin.robotwin_env.RoboTwinCodeEnv",
            "cfg": {
                "_target_": "capx.envs.tasks.base.CodeExecEnvConfig",
                "apis": ["RoboTwinPrivilegedApi"],
                "low_level": {
                    "_target_": "capx.envs.simulators.robotwin.RoboTwinEnv",
                    "robotwin_root": None,
                    "task_name": task.task,
                    "task_config": "demo_clean",
                    "embodiment": "aloha-agilex",
                    "privileged": True,
                    "max_steps": 128,
                    "video_stride": 8,
                },
                "privileged": True,
                "prompt": ROBOTWIN_PROMPT.format(task_description=task.task.replace("_", " ")),
            },
        },
        "record_video": True,
        "output_dir": str(output_dir),
        "use_oracle_code": False,
        "trials": 1,
        "num_workers": 1,
    }


def _load_completed(ledger_path: Path) -> set[str]:
    if not ledger_path.is_file():
        return set()
    completed = set()
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "complete":
            completed.add(row["task_key"])
    return completed


def _parse_summary(path: Path) -> dict[str, Any]:
    text = path.read_text()
    match = re.search(
        r"Code generation success rate / Average reward / Task completed:\s*\n"
        r"([0-9.]+)/([0-9.]+)/(\d+)",
        text,
    )
    if match is None:
        raise ValueError(f"Could not parse {path}")
    elapsed_match = re.search(r"Elapsed time: ([0-9.]+) seconds", text)
    return {
        "code_execution_rate": float(match.group(1)),
        "reward": float(match.group(2)),
        "task_completed": bool(int(match.group(3))),
        "elapsed_seconds": float(elapsed_match.group(1)) if elapsed_match else None,
        "summary_path": str(path.relative_to(REPO_ROOT)),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _refresh_analysis(config: dict[str, Any], config_path: str) -> None:
    """Run an optional experiment analyzer after each durable ledger update."""

    script = config.get("outputs", {}).get("analyze_script")
    if not script:
        return
    result = subprocess.run(
        [sys.executable, str(script), "--config", config_path],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        message = detail[-1] if detail else f"return code {result.returncode}"
        print(f"  warning: frontend manifest refresh failed: {message}", flush=True)
        return
    print("  frontend manifest refreshed", flush=True)


def _run_task(config: dict[str, Any], task: TaskSpec, *, dry_run: bool) -> dict[str, Any]:
    protocol = config["protocol"]
    task_parts = [task.benchmark]
    if task.suite:
        task_parts.append(task.suite)
    task_parts.append(f"{task.task_id:03d}_{_safe_name(task.task)}" if task.task_id is not None else task.task)
    output_base = REPO_ROOT / config["outputs"]["root"] / Path(*task_parts)
    config_dir = REPO_ROOT / config["outputs"]["root"] / "configs" / task.benchmark
    config_dir.mkdir(parents=True, exist_ok=True)
    generated_config_path = config_dir / f"{_safe_name(task.key)}.yaml"
    generated_config_path.write_text(yaml.safe_dump(_task_config(task, output_base / "run"), sort_keys=False))

    python = config["runtime"][f"{task.benchmark}_python"]
    command = [
        python,
        "capx/envs/launch.py",
        "--config-path",
        str(generated_config_path),
        "--server-url",
        protocol["server_url"],
        "--model",
        protocol["model"],
        "--temperature",
        str(protocol["temperature"]),
        "--max-tokens",
        str(protocol["max_tokens"]),
        "--reasoning-effort",
        protocol["reasoning_effort"],
        "--total-trials",
        str(protocol["trials_per_task"]),
        "--num-workers",
        str(protocol["num_workers"]),
        "--record-video",
        str(protocol["record_video"]),
        "--use-oracle-code",
        str(protocol["use_oracle_code"]),
    ]
    if dry_run:
        return {"command": command, "generated_config": str(generated_config_path)}

    env = _libero_env(config) if task.benchmark == "libero" else _robotwin_env(config)
    logs_dir = REPO_ROOT / config["outputs"]["root"] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{_safe_name(task.key)}.log"
    started = time.monotonic()
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    model_output = output_base / protocol["model"] / "run"
    summary_path = model_output / "summaries.txt"
    row: dict[str, Any] = {
        **asdict(task),
        "task_key": task.key,
        "model": protocol["model"],
        "oracle": False,
        "trial": 1,
        "seed": 1,
        "status": "complete" if summary_path.is_file() else "infra_error",
        "returncode": result.returncode,
        "wall_seconds": time.monotonic() - started,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "finished_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    if summary_path.is_file():
        row.update(_parse_summary(summary_path))
    return row


def main(default_config: str = "exp/exp00a/config.yaml") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--benchmark", choices=("all", "libero", "robotwin"), default="all")
    parser.add_argument("--task-key", help="Run only one exact task key")
    parser.add_argument("--limit", type=int, help="Run only the first N selected tasks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    config = _load_config(REPO_ROOT / args.config)
    tasks = _discover_libero(config) + _discover_robotwin(config)
    print(f"Discovered {sum(t.benchmark == 'libero' for t in tasks)} LIBERO tasks")
    print(f"Discovered {sum(t.benchmark == 'robotwin' for t in tasks)} RoboTwin tasks")
    tasks = _configured_tasks(config, tasks)
    manifest_path = REPO_ROOT / config["outputs"]["task_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([asdict(task) | {"task_key": task.key} for task in tasks], indent=2))

    print(f"Experiment scope contains {len(tasks)} tasks")
    if args.list_only:
        return

    if args.benchmark != "all":
        tasks = [task for task in tasks if task.benchmark == args.benchmark]
    if args.task_key:
        tasks = [task for task in tasks if task.key == args.task_key]
        if not tasks:
            raise SystemExit(f"Unknown task key: {args.task_key}")
    if args.limit is not None:
        tasks = tasks[: args.limit]

    ledger_path = REPO_ROOT / config["outputs"]["ledger"]
    completed = set() if args.no_resume else _load_completed(ledger_path)
    selected = [task for task in tasks if task.key not in completed]
    print(f"Selected {len(selected)} tasks ({len(tasks) - len(selected)} already complete)")

    for index, task in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {task.key}", flush=True)
        row = _run_task(copy.deepcopy(config), task, dry_run=args.dry_run)
        if args.dry_run:
            print(" ".join(row["command"]))
            continue
        _append_jsonl(ledger_path, row)
        _refresh_analysis(config, args.config)
        print(
            f"  status={row['status']} task_completed={row.get('task_completed')} "
            f"wall={row['wall_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()

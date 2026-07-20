from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import exp.exp00a.run as sweep_runner
import yaml

from exp.exp00a.run import (
    TaskSpec,
    _configured_tasks,
    _load_config,
    _refresh_analysis,
    _task_config,
)
from exp.exp00b.analyze import build_manifest
from viewer.app import create_app
from viewer.results import discover_runs


TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
]


def _config() -> dict:
    return {
        "protocol": {"model": "glm-5.2"},
        "scope": {"selected_robotwin_tasks": TASKS},
        "runtime": {"host": "xdlab23_yang", "gpu": 6},
        "outputs": {
            "root": "outputs/exp00b",
            "ledger": "exp/exp00b/results/ledger.jsonl",
        },
    }


def test_exp00b_selection_is_explicit_and_ordered() -> None:
    discovered = [TaskSpec(benchmark="libero", task="ignored")] + [
        TaskSpec(benchmark="robotwin", task=name) for name in reversed(TASKS)
    ]

    selected = _configured_tasks(_config(), discovered)

    assert [task.task for task in selected] == TASKS
    assert all(task.benchmark == "robotwin" for task in selected)


def test_glm_sweeps_use_non_truncating_budget_and_infra_guards() -> None:
    for path in (Path("exp/exp00a/config.yaml"), Path("exp/exp00b/config.yaml")):
        config = yaml.safe_load(path.read_text())
        assert config["protocol"]["max_tokens"] == 65536
        assert config["protocol"]["task_timeout_seconds"] == 600
        assert config["protocol"]["robotwin_seed_candidates"] == [1, 0, 2, 3, 4]


def test_robotwin_task_config_forwards_explicit_seed(tmp_path: Path) -> None:
    task = TaskSpec(benchmark="robotwin", task="click_bell")

    config = _task_config(
        task,
        tmp_path / "run",
        seed=100000,
        robotwin_task_config="demo_randomized",
    )

    assert config["env"]["cfg"]["low_level"]["seed"] == 100000
    assert config["env"]["cfg"]["low_level"]["task_config"] == "demo_randomized"


def test_load_config_accepts_outer_benchmark_runtime_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("runtime:\n  robotwin_root: old\n  robotwin_python: old-python\n  gpu: 5\n")
    monkeypatch.setenv("CAPX_BENCH_ROBOTWIN_ROOT", "/runtime/robotwin")
    monkeypatch.setenv("CAPX_BENCH_ROBOTWIN_PYTHON", "/runtime/python")
    monkeypatch.setenv("CAPX_BENCH_GPU", "7")

    config = _load_config(path)

    assert config["runtime"] == {
        "robotwin_root": "/runtime/robotwin",
        "robotwin_python": "/runtime/python",
        "gpu": 7,
    }


def test_exp00b_refreshes_frontend_manifest_after_ledger_update(monkeypatch) -> None:
    captured = {}

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr("exp.exp00a.run.subprocess.run", fake_run)
    config = _config()
    config["outputs"]["analyze_script"] = "exp/exp00b/analyze.py"

    _refresh_analysis(config, "exp/exp00b/config.yaml")

    assert captured["command"][1:] == [
        "exp/exp00b/analyze.py",
        "--config",
        "exp/exp00b/config.yaml",
    ]
    assert captured["kwargs"]["check"] is False


def test_task_timeout_terminates_the_dedicated_process_group(
    tmp_path: Path, monkeypatch
) -> None:
    waits = []
    signals = []
    captured = {}

    class FakeProcess:
        pid = 4242

        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired(["worker"], timeout)
            return -signal.SIGTERM

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(sweep_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        sweep_runner.os,
        "killpg",
        lambda process_group, sig: signals.append((process_group, sig)),
    )

    with (tmp_path / "task.log").open("w") as log:
        returncode, timed_out = sweep_runner._run_logged_process(
            ["worker"], env={}, log=log, timeout_seconds=0.01
        )

    assert returncode == 124
    assert timed_out is True
    assert captured["kwargs"]["start_new_session"] is True
    assert waits == [0.01, 10, None]
    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]


def test_unstable_robotwin_scene_falls_back_without_retrying_other_failures(
    monkeypatch,
) -> None:
    calls = []
    rows = [
        {
            "seed": 1,
            "status": "infra_error",
            "returncode": 1,
            "failure_type": "unstable_scene",
            "wall_seconds": 3.0,
            "log_path": "seed-1.log",
        },
        {
            "seed": 0,
            "status": "complete",
            "returncode": 0,
            "failure_type": None,
            "wall_seconds": 5.0,
            "log_path": "seed-0.log",
        },
    ]

    def fake_run(config, task, *, dry_run, seed, robotwin_task_config):
        del config, task, dry_run, robotwin_task_config
        calls.append(seed)
        return rows[len(calls) - 1].copy()

    monkeypatch.setattr(sweep_runner, "_run_task", fake_run)
    config = _config()
    config["protocol"]["robotwin_seed_candidates"] = [1, 0, 2]

    row = sweep_runner._run_task_with_seed_fallback(
        config,
        TaskSpec(benchmark="robotwin", task="dump_bin_bigbin"),
        dry_run=False,
        seed=None,
        robotwin_task_config="demo_clean",
    )

    assert calls == [1, 0]
    assert row["status"] == "complete"
    assert row["seed"] == 0
    assert row["seed_fallback_used"] is True
    assert [attempt["failure_type"] for attempt in row["seed_attempts"]] == [
        "unstable_scene",
        None,
    ]
    assert row["final_attempt_wall_seconds"] == 5.0
    assert row["wall_seconds"] == 8.0


def test_exp00b_manifest_updates_frontend_with_complete_and_failed_tasks(tmp_path: Path) -> None:
    config = _config()
    results = tmp_path / "exp/exp00b/results"
    results.mkdir(parents=True)
    rows = []
    for index, task in enumerate(TASKS):
        rows.append(
            {
                "benchmark": "robotwin",
                "task": task,
                "suite": None,
                "task_id": None,
                "task_key": f"robotwin:{task}",
                "model": "glm-5.2",
                "oracle": False,
                "trial": 1,
                "seed": 1,
                "status": "complete" if index == 0 else "infra_error",
                "returncode": 0 if index == 0 else 1,
                "wall_seconds": 2.5 + index,
                "task_completed": True if index == 0 else None,
                "code_execution_rate": 1.0 if index == 0 else None,
                "reward": 1.0 if index == 0 else None,
                "log_path": f"outputs/exp00b/logs/robotwin_{task}.log",
                "finished_at": "2026-07-20T00:00:00+00:00",
                "failure_type": "unstable_scene" if task == "handover_mic" else None,
                "seed_attempts": (
                    [{"seed": 1, "failure_type": "unstable_scene"}]
                    if task == "handover_mic"
                    else []
                ),
            }
        )
    (results / "ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    run = tmp_path / "outputs/exp00b/robotwin/adjust_bottle/glm-5.2/run"
    trial = run / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    trial.mkdir(parents=True)
    (trial / "video_combined.mp4").write_bytes(b"robotwin-video")
    (trial / "code.py").write_text("RESULT = True\n")
    (trial / "summary.txt").write_text("success\n")
    generated_config = tmp_path / "outputs/exp00b/configs/robotwin/adjust_bottle.yaml"
    generated_config.parent.mkdir(parents=True)
    generated_config.write_text(
        """env:
  cfg:
    low_level:
      _target_: capx.envs.simulators.robotwin.RoboTwinEnv
      task_name: adjust_bottle
      embodiment: aloha-agilex
"""
    )
    (run / "summaries.txt").write_text(
        "Model: glm-5.2\n"
        "Config Path: outputs/exp00b/configs/robotwin/adjust_bottle.yaml\n"
        "Elapsed Time: 2.5 seconds\n"
    )

    manifest = build_manifest(tmp_path, config)
    manifest_path = results / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    assert len(manifest["runs"]) == 10
    complete = next(run for run in manifest["runs"] if run["config"]["task"] == "adjust_bottle")
    assert complete["episodes"][0]["metrics"]["task_success"] is True
    failed = next(run for run in manifest["runs"] if run["config"]["task"] == "handover_mic")
    assert failed["status"] == "failed"
    assert failed["metrics"]["episodes_finished"] == 0
    assert failed["episodes"][0]["error"]["stage"] == "environment_reset"
    assert failed["episodes"][0]["error"]["type"] == "UnstableSceneError"

    client = create_app(tmp_path, testing=True).test_client()
    response = client.get("/api/runs?benchmark=robotwin")
    assert response.status_code == 200
    assert len(response.get_json()["runs"]) == 10
    detail = client.get(f"/api/runs/{complete['run_id']}").get_json()
    video_url = detail["episodes"][0]["artifacts"]["video"]
    assert video_url
    video = client.get(video_url)
    assert video.status_code == 200
    assert video.data == b"robotwin-video"


def test_discover_runs_relocates_remote_absolute_config(tmp_path: Path) -> None:
    config = tmp_path / "outputs/exp00b/configs/robotwin/robotwin_click_bell.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "env:\n"
        "  cfg:\n"
        "    low_level:\n"
        "      _target_: capx.envs.simulators.robotwin.RoboTwinEnv\n"
        "      task_name: click_bell\n"
    )
    run = tmp_path / "outputs/exp00b/robotwin/click_bell/glm-5.2/run"
    trial = run / "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0"
    trial.mkdir(parents=True)
    (trial / "code.py").write_text("print('ok')\n")
    (run / "summaries.txt").write_text(
        "Model: glm-5.2\n"
        "Config Path: /remote/worker/cap-x/outputs/exp00b/configs/robotwin/"
        "robotwin_click_bell.yaml\n"
    )

    runs = discover_runs(tmp_path / "outputs/exp00b", repo_root=tmp_path)

    assert len(runs) == 1
    payload = runs[0].to_dict()
    assert payload["benchmark"] == "robotwin"
    assert payload["config"]["task"] == "click_bell"
    assert payload["config"]["config_path"] == config.relative_to(tmp_path).as_posix()

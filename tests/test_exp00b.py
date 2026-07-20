from __future__ import annotations

import json
from pathlib import Path

from exp.exp00a.run import TaskSpec, _configured_tasks, _refresh_analysis, _task_config
from exp.exp00b.analyze import build_manifest
from viewer.app import create_app


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
    assert failed["episodes"][0]["error"]["stage"] == "infrastructure"

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

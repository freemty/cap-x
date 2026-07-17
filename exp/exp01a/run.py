"""Run experiment exp01a."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _prepare_task(config: dict[str, Any]):
    robotwin_root = (REPO_ROOT / config["runtime"]["robotwin_root"]).resolve()
    if not robotwin_root.is_dir():
        raise FileNotFoundError(f"RoboTwin checkout not found: {robotwin_root}")

    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))

    task_name = config["robotwin"]["task"]
    module = importlib.import_module(f"envs.{task_name}")
    task = getattr(module, task_name)()

    args = _load_yaml(robotwin_root / "task_config" / f"{config['robotwin']['task_config']}.yml")
    args["task_name"] = task_name
    args["render_freq"] = 0
    args["collect_data"] = False
    args["save_data"] = False
    args["save_freq"] = None
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["domain_randomization"]["random_embodiment"] = False

    embodiment = config["robotwin"]["embodiment"]
    args["embodiment"] = [embodiment]
    embodiment_index = _load_yaml(robotwin_root / "task_config" / "_embodiment_config.yml")
    embodiment_root = (robotwin_root / embodiment_index[embodiment]["file_path"]).resolve()
    embodiment_config = _load_yaml(embodiment_root / "config.yml")
    args["left_robot_file"] = str(embodiment_root)
    args["right_robot_file"] = str(embodiment_root)
    args["left_embodiment_config"] = embodiment_config
    args["right_embodiment_config"] = embodiment_config
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = embodiment
    return task, args


def _run_trial(config: dict[str, Any], seed: int) -> dict[str, Any]:
    task, args = _prepare_task(config)
    started = time.monotonic()
    result: dict[str, Any] = {
        "seed": seed,
        "task": config["robotwin"]["task"],
        "embodiment": config["robotwin"]["embodiment"],
        "success": False,
        "plan_success": False,
        "error": "",
    }
    try:
        task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        info = task.play_once()
        result["plan_success"] = bool(task.plan_success)
        result["success"] = bool(task.plan_success and task.check_success())
        result["info"] = info
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        try:
            task.close_env(clear_cache=True)
        except Exception:
            pass
        result["elapsed_seconds"] = time.monotonic() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exp01a")
    parser.add_argument("--config", default="exp/exp01a/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=str, help="Resume from RUNID")
    args = parser.parse_args()

    config = _load_yaml(REPO_ROOT / args.config)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    run_id = args.resume or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    first_seed = int(config["robotwin"]["seed"])
    trials = int(config["robotwin"]["trials"])
    results = [_run_trial(config, first_seed + offset) for offset in range(trials)]

    result_path = REPO_ROOT / "exp" / "exp01a" / "results" / f"{run_id}.json"
    result_path.write_text(json.dumps({"run_id": run_id, "results": results}, indent=2, default=str))
    success_count = sum(int(item["success"]) for item in results)
    log_path = REPO_ROOT / "exp" / "exp01a" / "results" / "runs.log"
    timestamp = datetime.now(timezone.utc).isoformat()
    with log_path.open("a") as handle:
        handle.write(f"{timestamp} robotwin_oracle {run_id} {result_path.relative_to(REPO_ROOT)}\n")

    print(f"{success_count}/{trials} successful; results: {result_path}")
    if success_count != trials:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

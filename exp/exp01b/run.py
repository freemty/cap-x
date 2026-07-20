"""Run the exp01b CaP-X dual-benchmark integration checks."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOSUITE_PRIVATE_MACROS = (
    REPO_ROOT
    / "capx/third_party/libero_dependencies/robosuite/robosuite/macros_private.py"
)
ROBOSUITE_PRIVATE_MACROS_TEMPLATE = REPO_ROOT / "exp/exp01b/robosuite_macros_private.py"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload


def _prepare_libero_runtime() -> None:
    """Create the ignored robosuite host override without replacing user settings."""

    if ROBOSUITE_PRIVATE_MACROS.is_file():
        return
    ROBOSUITE_PRIVATE_MACROS.write_text(ROBOSUITE_PRIVATE_MACROS_TEMPLATE.read_text())
    print(f"[libero] created {ROBOSUITE_PRIVATE_MACROS}")


def _command(config: dict[str, Any], benchmark: str) -> tuple[list[str], dict[str, str]]:
    runtime = config["runtime"]
    spec = config["benchmarks"][benchmark]
    python_key = "robotwin_python" if benchmark == "robotwin" else "libero_python"
    command = [
        runtime[python_key],
        "capx/envs/launch.py",
        "--config-path",
        spec["config"],
        "--total-trials",
        "1",
        "--num-workers",
        "1",
        "--record-video",
        "True",
        "--use-oracle-code",
        "True",
        "--output-dir",
        spec["output_dir"],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(runtime["gpu"])
    env["MUJOCO_GL"] = "egl"
    python_paths = [str(REPO_ROOT)]
    if benchmark == "robotwin":
        env["ROBOTWIN_ROOT"] = runtime["robotwin_root"]
        env["PYTHONNOUSERSITE"] = "1"
        python_paths.append(runtime["robotwin_root"])
    else:
        python_paths[:0] = [
            str(REPO_ROOT / "capx/third_party/libero_dependencies/robosuite"),
            str(REPO_ROOT / "capx/third_party/LIBERO-PRO/libero"),
        ]
        env["PYTHONNOUSERSITE"] = "1"
        env["ROBOT_DESCRIPTIONS_CACHE"] = runtime["robot_descriptions_cache"]
        env["ROBOT_DESCRIPTION_COMMIT"] = runtime["robot_description_commit"]
    env["PYTHONPATH"] = os.pathsep.join(python_paths) + os.pathsep + env.get("PYTHONPATH", "")
    return command, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="exp/exp01b/config.yaml")
    parser.add_argument(
        "--benchmark",
        choices=("robotwin", "libero", "all"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _load_config(REPO_ROOT / args.config)
    benchmarks = ("robotwin", "libero") if args.benchmark == "all" else (args.benchmark,)
    for benchmark in benchmarks:
        if benchmark == "libero" and not args.dry_run:
            _prepare_libero_runtime()
        command, env = _command(config, benchmark)
        print(f"[{benchmark}] {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

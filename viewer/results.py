"""Build portable normalized manifests from CaP-X output folders.

This module is the producer-side companion to :mod:`viewer.adapters`.  The
exp01b analyzer imports :func:`discover_runs`, serializes each ``ResultRun``,
and writes the resulting ``{schema, runs}`` wrapper to its experiment folder.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import (
    SCHEMA_VERSION,
    TRIAL_DIR_RE,
    _contained,
    _iso_mtime,
    _libero_trial_episode,
    _read_yaml,
    aggregate_episode_metrics,
)


@dataclass(frozen=True)
class ResultRun:
    """One JSON-serializable normalized benchmark run."""

    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _summary_metadata(parent: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    path = parent / "summaries.txt"
    if not path.is_file():
        return result
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return result
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower().replace(" ", "_")
        if normalized in {"model", "config_path", "git_commit", "elapsed_time"}:
            result[normalized] = value.strip()
    return result


def _resolve_config_path(value: str | None, repo_root: Path) -> Path | None:
    if not value:
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [repo_root / raw]
    return next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate.is_file() and _contained(candidate, repo_root)
        ),
        None,
    )


def _config_metadata(config_path: Path | None, repo_root: Path) -> dict[str, Any]:
    payload = _read_yaml(config_path) if config_path is not None else {}
    env = payload.get("env") if isinstance(payload.get("env"), dict) else {}
    cfg = env.get("cfg") if isinstance(env.get("cfg"), dict) else {}
    low_level = cfg.get("low_level") if isinstance(cfg.get("low_level"), dict) else {}
    target = str(low_level.get("_target_") or "").lower()
    benchmark = "robotwin" if "robotwin" in target else "libero" if "libero" in target else None
    suite = low_level.get("suite_name")
    task_id = low_level.get("task_id")
    task = low_level.get("task_name")
    if task is None and suite is not None:
        task = f"{suite}[{task_id}]" if task_id is not None else str(suite)
    relative_config = None
    if config_path is not None:
        try:
            relative_config = config_path.relative_to(repo_root).as_posix()
        except ValueError:
            pass
    return {
        "benchmark": benchmark,
        "task": task,
        "suite": suite,
        "task_id": task_id,
        "embodiment": low_level.get("embodiment") or ("franka" if benchmark == "libero" else None),
        "config_path": relative_config,
    }


def _benchmark_from_parent(parent: Path, config: dict[str, Any]) -> str:
    if config.get("benchmark") in {"robotwin", "libero"}:
        return str(config["benchmark"])
    lowered = [part.lower() for part in parent.parts]
    if "robotwin" in lowered:
        return "robotwin"
    if "libero" in lowered or any(part.startswith("libero_") for part in lowered):
        return "libero"
    return "unknown"


def _portable_artifacts(
    episodes: list[dict[str, Any]],
    registry: dict[str, str],
    repo_root: Path,
) -> None:
    for episode in episodes:
        for label, key in list(episode.get("artifacts", {}).items()):
            target_value = registry.get(key) if isinstance(key, str) else None
            if target_value is None:
                episode["artifacts"][label] = None
                continue
            target = Path(target_value).resolve()
            try:
                episode["artifacts"][label] = target.relative_to(repo_root).as_posix()
            except ValueError:
                episode["artifacts"][label] = None


def _experiment_name(parent: Path, repo_root: Path) -> str | None:
    try:
        parts = parent.relative_to(repo_root).parts
    except ValueError:
        return None
    for part in parts:
        if part.startswith("exp"):
            return part
    return None


def _build_run(
    parent: Path,
    trial_items: list[tuple[Path, Any]],
    *,
    repo_root: Path,
) -> ResultRun:
    summary = _summary_metadata(parent)
    config_path = _resolve_config_path(summary.get("config_path"), repo_root)
    config = _config_metadata(config_path, repo_root)
    benchmark = _benchmark_from_parent(parent, config)
    registry: dict[str, str] = {}
    episodes = [
        _libero_trial_episode(
            trial_dir,
            match,
            repo_root=repo_root,
            registry=registry,
        )
        for trial_dir, match in sorted(
            trial_items, key=lambda item: int(item[1].group("trial"))
        )
    ]
    elapsed_match = re.search(r"[0-9]+(?:\.[0-9]+)?", summary.get("elapsed_time", ""))
    if len(episodes) == 1 and elapsed_match is not None:
        episodes[0]["metrics"]["elapsed_seconds"] = float(elapsed_match.group(0))
    for episode in episodes:
        episode["task"] = config.get("task")
        if benchmark == "robotwin" and episode["metrics"].get("task_success") is True:
            # RoboTwin reward is defined as plan_success AND task_success. A
            # successful task therefore proves planning succeeded; failures do
            # not identify which of the two predicates failed.
            episode["metrics"]["plan_success"] = True
    _portable_artifacts(episodes, registry, repo_root)

    relative = parent.resolve().relative_to(repo_root.resolve()).as_posix()
    digest = hashlib.sha1(relative.encode()).hexdigest()[:10]
    model = summary.get("model")
    if not model and parent.parent != parent:
        model = parent.parent.name
    policy = "capx-oracle" if model == "oracle" else "capx"
    done_flag = parent / "aaa_done_flag" / "aaa_done_flag.txt"
    completed = done_flag.is_file() or (parent / "summaries.txt").is_file()
    status = "complete" if completed else "running"
    metrics = aggregate_episode_metrics(episodes)
    source_path = relative
    run_id = f"{_experiment_name(parent, repo_root) or 'run'}-{benchmark}-{model or 'unknown'}-{digest}"

    return ResultRun(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "benchmark": benchmark,
            "policy": policy,
            "status": status,
            "experiment": _experiment_name(parent, repo_root),
            "started_at": None,
            "finished_at": _iso_mtime(parent) if completed else None,
            "updated_at": _iso_mtime(parent),
            "runtime": {"git_commit": summary.get("git_commit")},
            "config": {
                "task": config.get("task"),
                "suite": config.get("suite"),
                "task_id": config.get("task_id"),
                "embodiment": config.get("embodiment"),
                "model": model,
                "config_path": config.get("config_path"),
            },
            "metrics": metrics,
            "episodes": episodes,
            "error": None,
            "source": "capx-output-folders",
            "source_path": source_path,
            "artifact_root": source_path,
        }
    )


def discover_runs(
    output_root: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> list[ResultRun]:
    """Discover normalized runs below an arbitrary CaP-X output directory."""

    output = Path(output_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve() if repo_root is not None else output.parent
    if not output.is_dir() or not _contained(output, repo):
        return []

    groups: dict[Path, list[tuple[Path, Any]]] = {}
    for trial_dir in output.rglob("trial_*"):
        if not trial_dir.is_dir():
            continue
        match = TRIAL_DIR_RE.match(trial_dir.name)
        if match is not None:
            groups.setdefault(trial_dir.parent, []).append((trial_dir, match))

    return [
        _build_run(parent, trial_items, repo_root=repo)
        for parent, trial_items in sorted(groups.items(), key=lambda item: str(item[0]))
    ]

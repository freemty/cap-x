"""Adapters from CaP-X benchmark artifacts to one read-only viewer schema.

The viewer intentionally accepts both the normalized ``manifest.json`` format
and the two formats that pre-date it:

* RoboTwin ``exp/*/results/run_*.json`` files.
* CaP-X/LIBERO ``trial_*_sandboxrc_*_reward_*_taskcompleted_*`` folders.

All artifact paths are resolved while indexing and retained in a private
allow-list.  The Flask layer only serves entries from that allow-list.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:  # PyYAML is part of CaP-X, but the viewer can still run without it.
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal deployments.
    yaml = None


SCHEMA_VERSION = "capx.result.v1"
TRIAL_DIR_RE = re.compile(
    r"^trial_(?P<trial>\d+)_sandboxrc_(?P<sandbox_rc>-?\d+)_"
    r"reward_(?P<reward>-?(?:\d+(?:\.\d*)?|\.\d+))_"
    r"taskcompleted_(?P<task_completed>[01])$"
)


def _iso_mtime(path: Path) -> str:
    try:
        timestamp = path.stat().st_mtime
    except OSError:
        timestamp = 0
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def safe_artifact_key(value: str) -> bool:
    """Return whether *value* is a safe relative URL/path fragment."""

    if not value or "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def _artifact_key(path: Path, base_dir: Path, repo_root: Path, label: str) -> str:
    for root in (base_dir, repo_root):
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        if safe_artifact_key(relative):
            return relative
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:10]
    return f"{label}-{digest}{path.suffix}"


def _register_artifact(
    registry: dict[str, str],
    *,
    repo_root: Path,
    base_dir: Path,
    value: Any,
    label: str,
) -> str | None:
    """Resolve one declared artifact and add it to the run allow-list."""

    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value.strip()).expanduser()
    # A producer may only declare a normal relative artifact path.  Do not
    # silently collapse ``..`` even when the collapsed target remains inside
    # the repository: that would make unrelated repo files addressable.
    if not raw.is_absolute() and not safe_artifact_key(raw.as_posix()):
        return None
    candidates = [raw] if raw.is_absolute() else [base_dir / raw, repo_root / raw]
    target = next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate.is_file() and _contained(candidate, repo_root)
        ),
        None,
    )
    if target is None:
        return None

    key = _artifact_key(target, base_dir, repo_root, label)
    if key in registry and Path(registry[key]).resolve() != target:
        digest = hashlib.sha1(str(target).encode()).hexdigest()[:8]
        key = f"{label}-{digest}{target.suffix}"
    registry[key] = str(target)
    return key


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "success", "succeeded"}:
            return True
        if lowered in {"0", "false", "no", "failed", "failure"}:
            return False
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # Reject NaN.


def _zero_return_code(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return None


def _rate(values: Iterable[bool | None]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return sum(bool(value) for value in observed) / len(observed)


def _mean(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return sum(observed) / len(observed)


def aggregate_episode_metrics(
    episodes: list[dict[str, Any]], total_hint: int | None = None
) -> dict[str, Any]:
    metrics = [episode.get("metrics", {}) for episode in episodes]
    finished = sum(episode.get("status") in {"complete", "failed"} for episode in episodes)
    return {
        "episodes_total": max(total_hint or 0, len(episodes)),
        "episodes_finished": finished,
        "task_success_rate": _rate(_as_bool(item.get("task_success")) for item in metrics),
        "code_execution_rate": _rate(
            _as_bool(item.get("code_execution_success")) for item in metrics
        ),
        "plan_success_rate": _rate(_as_bool(item.get("plan_success")) for item in metrics),
        "mean_reward": _mean(_as_float(item.get("reward")) for item in metrics),
        "mean_elapsed_seconds": _mean(
            _as_float(item.get("elapsed_seconds")) for item in metrics
        ),
    }


def _error_payload(value: Any, *, stage: str = "execution") -> dict[str, Any] | None:
    if value in (None, "", False):
        return None
    if isinstance(value, dict):
        result = dict(value)
        result.setdefault("stage", stage)
        if "message" not in result:
            result["message"] = str(result.get("traceback", result))
        return result
    text = str(value)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "stage": stage,
        "type": "RuntimeError",
        "message": lines[-1] if lines else text,
        "traceback": text,
    }


def _normalise_status(value: Any, *, error: Any = None) -> str:
    status = str(value or "").strip().lower()
    aliases = {
        "completed": "complete",
        "succeeded": "complete",
        "success": "complete",
        "errored": "failed",
        "error": "failed",
        "pending": "running",
    }
    status = aliases.get(status, status)
    if status in {"running", "complete", "failed"}:
        return status
    return "failed" if error not in (None, "", False) else "complete"


def _experiment_root(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name.startswith("exp") and parent.parent.name == "exp":
            return parent
    return None


def _normalise_manifest_episode(
    item: dict[str, Any],
    *,
    index: int,
    benchmark: str,
    repo_root: Path,
    artifact_root: Path,
    registry: dict[str, str],
) -> dict[str, Any]:
    raw_metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    task_success_fallback = item.get("success") if benchmark == "robotwin" else None
    metrics = {
        "code_execution_success": _as_bool(
            _coalesce(
                raw_metrics.get("code_execution_success"),
                item.get("code_execution_success"),
                item.get("execution_success"),
                _zero_return_code(item.get("sandbox_rc")),
            )
        ),
        "plan_success": _as_bool(
            _coalesce(raw_metrics.get("plan_success"), item.get("plan_success"))
        ),
        "task_success": _as_bool(
            _coalesce(
                raw_metrics.get("task_success"),
                item.get("task_success"),
                item.get("task_completed"),
                task_success_fallback,
            )
        ),
        "reward": _as_float(_coalesce(raw_metrics.get("reward"), item.get("reward"))),
        "elapsed_seconds": _as_float(
            _coalesce(raw_metrics.get("elapsed_seconds"), item.get("elapsed_seconds"))
        ),
        "sandbox_rc": _coalesce(raw_metrics.get("sandbox_rc"), item.get("sandbox_rc")),
        "num_code_blocks": _coalesce(
            raw_metrics.get("num_code_blocks"), item.get("num_code_blocks")
        ),
        "num_regenerations": _coalesce(
            raw_metrics.get("num_regenerations"), item.get("num_regenerations")
        ),
    }

    declared = item.get("artifacts") if isinstance(item.get("artifacts"), dict) else {}
    aliases = {
        "video": (declared.get("video"), declared.get("video_path"), item.get("video"), item.get("video_path")),
        "thumbnail": (declared.get("thumbnail"), item.get("thumbnail")),
        "generated_code": (
            declared.get("generated_code"),
            declared.get("code"),
            item.get("generated_code"),
            item.get("code_path"),
        ),
        "summary": (declared.get("summary"), item.get("summary_path")),
        "events": (declared.get("events"), item.get("events_path")),
        "responses": (declared.get("responses"), declared.get("all_responses")),
    }
    artifacts: dict[str, str | None] = {}
    for label, choices in aliases.items():
        value = _coalesce(*choices)
        artifacts[label] = _register_artifact(
            registry,
            repo_root=repo_root,
            base_dir=artifact_root,
            value=value,
            label=label,
        )

    error = _error_payload(item.get("error"))
    episode_id = _coalesce(item.get("episode_id"), item.get("id"))
    if episode_id is None:
        if item.get("seed") is not None:
            episode_id = f"seed-{item['seed']}"
        elif item.get("trial") is not None:
            episode_id = f"trial-{item['trial']}"
        else:
            episode_id = f"episode-{index + 1}"

    return {
        "episode_id": str(episode_id),
        "seed": item.get("seed"),
        "trial": item.get("trial"),
        "task": item.get("task"),
        "status": _normalise_status(item.get("status"), error=error),
        "metrics": metrics,
        "artifacts": artifacts,
        "error": error,
        "details": item.get("info") if isinstance(item.get("info"), dict) else None,
    }


def _normalise_manifest_payload(
    payload: dict[str, Any],
    path: Path,
    repo_root: Path,
    *,
    wrapper_schema: str | None = None,
) -> dict[str, Any]:
    benchmark = str(payload.get("benchmark") or "unknown").strip().lower()
    artifact_root = path.parent
    root_hint = payload.get("artifact_root")
    if isinstance(root_hint, str) and root_hint:
        hinted = Path(root_hint).expanduser()
        candidates = [hinted] if hinted.is_absolute() else [path.parent / hinted, repo_root / hinted]
        resolved = next(
            (
                candidate.resolve()
                for candidate in candidates
                if candidate.is_dir() and _contained(candidate, repo_root)
            ),
            None,
        )
        if resolved is not None:
            artifact_root = resolved

    registry: dict[str, str] = {}
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list):
        raw_episodes = payload.get("results") if isinstance(payload.get("results"), list) else []
    episodes = [
        _normalise_manifest_episode(
            item,
            index=index,
            benchmark=benchmark,
            repo_root=repo_root,
            artifact_root=artifact_root,
            registry=registry,
        )
        for index, item in enumerate(raw_episodes)
        if isinstance(item, dict)
    ]

    provided_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    total_hint = provided_metrics.get("episodes_total")
    try:
        total_hint = int(total_hint) if total_hint is not None else None
    except (TypeError, ValueError):
        total_hint = None
    metrics = aggregate_episode_metrics(episodes, total_hint)
    for key, value in provided_metrics.items():
        if value is not None:
            metrics[key] = value

    run_id = str(payload.get("run_id") or f"manifest-{hashlib.sha1(str(path).encode()).hexdigest()[:12]}")
    status = _normalise_status(payload.get("status"), error=payload.get("error"))
    return {
        "schema_version": str(
            payload.get("schema_version") or payload.get("schema") or wrapper_schema or SCHEMA_VERSION
        ),
        "run_id": run_id,
        "benchmark": benchmark,
        "policy": str(payload.get("policy") or "capx"),
        "status": status,
        "experiment": payload.get("experiment") or (_experiment_root(path).name if _experiment_root(path) else None),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "updated_at": payload.get("updated_at") or _iso_mtime(path),
        "runtime": payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {},
        "config": payload.get("config") if isinstance(payload.get("config"), dict) else {},
        "metrics": metrics,
        "episodes": episodes,
        "error": _error_payload(payload.get("error"), stage="run"),
        "source": "manifest",
        "source_path": str(path.resolve().relative_to(repo_root.resolve())),
        "_artifact_files": registry,
        "_covered_roots": [str(artifact_root.resolve())],
    }


def load_manifests(path: Path, repo_root: Path) -> list[dict[str, Any]]:
    """Load a single-run manifest or the exp01b ``{schema, runs}`` wrapper."""

    payload = _read_json(path)
    if payload is None:
        return []
    raw_runs = payload.get("runs")
    if isinstance(raw_runs, list):
        wrapper_schema = str(payload.get("schema") or payload.get("schema_version") or SCHEMA_VERSION)
        return [
            _normalise_manifest_payload(
                item,
                path,
                repo_root,
                wrapper_schema=wrapper_schema,
            )
            for item in raw_runs
            if isinstance(item, dict)
        ]
    return [_normalise_manifest_payload(payload, path, repo_root)]


def load_manifest(path: Path, repo_root: Path) -> dict[str, Any] | None:
    """Backward-compatible helper for callers expecting one manifest run."""

    runs = load_manifests(path, repo_root)
    return runs[0] if runs else None


def load_legacy_robotwin(path: Path, repo_root: Path) -> dict[str, Any] | None:
    payload = _read_json(path)
    if payload is None or not isinstance(payload.get("results"), list):
        return None
    exp_root = _experiment_root(path)
    exp_id = exp_root.name if exp_root else path.parent.parent.name
    config_payload = _read_yaml(exp_root / "config.yaml") if exp_root else {}
    robotwin_config = config_payload.get("robotwin", {}) if isinstance(config_payload.get("robotwin"), dict) else {}
    runtime = config_payload.get("runtime", {}) if isinstance(config_payload.get("runtime"), dict) else {}

    episodes: list[dict[str, Any]] = []
    for index, item in enumerate(payload["results"]):
        if not isinstance(item, dict):
            continue
        error = _error_payload(item.get("error"))
        episodes.append(
            {
                "episode_id": f"seed-{item.get('seed', index)}",
                "seed": item.get("seed"),
                "trial": index + 1,
                "task": item.get("task") or robotwin_config.get("task"),
                "status": _normalise_status(None, error=error),
                "metrics": {
                    "code_execution_success": None,
                    "plan_success": _as_bool(item.get("plan_success")),
                    "task_success": _as_bool(item.get("success")),
                    "reward": _as_float(item.get("reward")),
                    "elapsed_seconds": _as_float(item.get("elapsed_seconds")),
                    "sandbox_rc": None,
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
                "error": error,
                "details": item.get("info") if isinstance(item.get("info"), dict) else None,
            }
        )

    raw_run_id = str(payload.get("run_id") or path.stem)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{exp_id}-{raw_run_id}",
        "benchmark": "robotwin",
        "policy": str(payload.get("policy") or "oracle"),
        "status": "complete",
        "experiment": exp_id,
        "started_at": None,
        "finished_at": _iso_mtime(path),
        "updated_at": _iso_mtime(path),
        "runtime": runtime,
        "config": {
            "task": robotwin_config.get("task") or (episodes[0].get("task") if episodes else None),
            "suite": None,
            "task_id": None,
            "embodiment": robotwin_config.get("embodiment") or (
                payload["results"][0].get("embodiment") if payload["results"] else None
            ),
            "model": None,
            "seed_start": robotwin_config.get("seed"),
        },
        "metrics": aggregate_episode_metrics(episodes),
        "episodes": episodes,
        "error": None,
        "source": "robotwin-legacy",
        "source_path": str(path.resolve().relative_to(repo_root.resolve())),
        "_artifact_files": {},
        "_covered_roots": [],
    }


def _infer_libero_config(parent: Path, outputs_root: Path) -> dict[str, Any]:
    try:
        parts = parent.resolve().relative_to(outputs_root.resolve()).parts
    except ValueError:
        parts = parent.parts
    suite_index = next(
        (index for index, part in enumerate(parts) if part.startswith("libero_")),
        None,
    )
    suite = parts[suite_index] if suite_index is not None else None
    task = parts[suite_index + 1] if suite_index is not None and suite_index + 1 < len(parts) else None
    model = parts[suite_index + 2] if suite_index is not None and suite_index + 2 < len(parts) else None

    summary_path = parent / "summaries.txt"
    config_path = None
    if summary_path.is_file():
        try:
            for line in summary_path.read_text(errors="replace").splitlines():
                if line.startswith("Config Path:"):
                    config_path = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {
        "task": task,
        "suite": suite,
        "task_id": None,
        "embodiment": "franka",
        "model": model,
        "config_path": config_path,
    }


def _libero_trial_episode(
    trial_dir: Path,
    match: re.Match[str],
    *,
    repo_root: Path,
    registry: dict[str, str],
) -> dict[str, Any]:
    sandbox_rc = int(match.group("sandbox_rc"))
    task_completed = bool(int(match.group("task_completed")))
    reward = float(match.group("reward"))
    declared: dict[str, Path | None] = {
        "video": next(
            iter(
                sorted(trial_dir.glob("video_combined*.mp4"))
                or sorted(trial_dir.glob("video_*.mp4"))
            ),
            None,
        ),
        "thumbnail": next(iter(sorted(trial_dir.glob("visual_feedback_*.png"))), None),
        "generated_code": trial_dir / "code.py",
        "summary": trial_dir / "summary.txt",
        "responses": trial_dir / "all_responses.json",
        "events": next(iter(sorted(trial_dir.glob("execution_history*.json"))), None),
    }
    artifacts = {
        label: _register_artifact(
            registry,
            repo_root=repo_root,
            base_dir=trial_dir.parent,
            value=str(path) if path is not None and path.is_file() else None,
            label=label,
        )
        for label, path in declared.items()
    }

    error = None
    if sandbox_rc != 0:
        summary = trial_dir / "summary.txt"
        message = f"Sandbox exited with code {sandbox_rc}"
        traceback_text = None
        if summary.is_file():
            try:
                traceback_text = summary.read_text(errors="replace")
            except OSError:
                pass
        error = {
            "stage": "execution",
            "type": "SandboxError",
            "message": message,
            "traceback": traceback_text,
        }

    return {
        "episode_id": f"trial-{int(match.group('trial'))}",
        "seed": None,
        "trial": int(match.group("trial")),
        "task": None,
        "status": "failed" if sandbox_rc != 0 else "complete",
        "metrics": {
            "code_execution_success": sandbox_rc == 0,
            "plan_success": None,
            "task_success": task_completed,
            "reward": reward,
            "elapsed_seconds": None,
            "sandbox_rc": sandbox_rc,
            "num_code_blocks": None,
            "num_regenerations": None,
        },
        "artifacts": artifacts,
        "error": error,
        "details": None,
    }


def discover_libero_runs(
    repo_root: Path, *, excluded_roots: Iterable[Path] = ()
) -> list[dict[str, Any]]:
    outputs_root = repo_root / "outputs"
    if not outputs_root.is_dir():
        return []
    excluded = [path.resolve() for path in excluded_roots if path.exists()]
    groups: dict[Path, list[tuple[Path, re.Match[str]]]] = {}
    for trial_dir in outputs_root.rglob("trial_*"):
        if not trial_dir.is_dir():
            continue
        match = TRIAL_DIR_RE.match(trial_dir.name)
        if match is None:
            continue
        if any(_contained(trial_dir, root) for root in excluded):
            continue
        groups.setdefault(trial_dir.parent, []).append((trial_dir, match))

    runs: list[dict[str, Any]] = []
    for parent, trial_items in groups.items():
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
        config = _infer_libero_config(parent, outputs_root)
        for episode in episodes:
            episode["task"] = config.get("task")
        relative = parent.resolve().relative_to(repo_root.resolve()).as_posix()
        digest = hashlib.sha1(relative.encode()).hexdigest()[:10]
        done_flag = parent / "aaa_done_flag" / "aaa_done_flag.txt"
        summary_path = parent / "summaries.txt"
        status = "complete" if done_flag.is_file() or summary_path.is_file() else "running"
        run_id = f"libero-{parent.name}-{digest}"
        runs.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "benchmark": "libero",
                "policy": "capx",
                "status": status,
                "experiment": None,
                "started_at": None,
                "finished_at": _iso_mtime(parent) if status == "complete" else None,
                "updated_at": _iso_mtime(parent),
                "runtime": {},
                "config": config,
                "metrics": aggregate_episode_metrics(episodes),
                "episodes": episodes,
                "error": None,
                "source": "libero-trial-folders",
                "source_path": relative,
                "_artifact_files": registry,
                "_covered_roots": [str(parent.resolve())],
            }
        )
    return runs


def discover_runs(repo_root: Path | str) -> list[dict[str, Any]]:
    root = Path(repo_root).expanduser().resolve()
    runs: list[dict[str, Any]] = []

    manifest_path_set: set[Path] = set()
    for base in (root / "exp", root / "outputs"):
        if base.is_dir():
            manifest_path_set.update(base.glob("**/manifest.json"))
    manifest_paths = sorted(manifest_path_set)
    manifest_exp_roots: set[Path] = set()
    for path in manifest_paths:
        manifest_runs = load_manifests(path, root)
        if not manifest_runs:
            continue
        runs.extend(manifest_runs)
        exp_root = _experiment_root(path)
        if exp_root is not None:
            manifest_exp_roots.add(exp_root.resolve())

    results_root = root / "exp"
    if results_root.is_dir():
        for path in sorted(results_root.glob("*/results/run_*.json")):
            exp_root = _experiment_root(path)
            if exp_root is not None and exp_root.resolve() in manifest_exp_roots:
                continue
            run = load_legacy_robotwin(path, root)
            if run is not None:
                runs.append(run)

    covered_roots = {
        Path(value)
        for run in runs
        for value in run.get("_covered_roots", [])
        if isinstance(value, str)
    }
    runs.extend(discover_libero_runs(root, excluded_roots=covered_roots))

    # A malformed producer should not make two objects addressable by one ID.
    deduplicated: dict[str, dict[str, Any]] = {}
    for run in runs:
        deduplicated.setdefault(str(run["run_id"]), run)
    return sorted(
        deduplicated.values(),
        key=lambda run: str(run.get("updated_at") or ""),
        reverse=True,
    )


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    benchmark_names = sorted({str(run.get("benchmark") or "unknown") for run in runs})
    benchmarks = []
    for name in benchmark_names:
        selected = [run for run in runs if run.get("benchmark") == name]
        episodes = [episode for run in selected for episode in run.get("episodes", [])]
        metrics = aggregate_episode_metrics(episodes)
        metrics.update(
            {
                "runs": len(selected),
                "runs_running": sum(run.get("status") == "running" for run in selected),
                "runs_failed": sum(run.get("status") == "failed" for run in selected),
            }
        )
        benchmarks.append({"benchmark": name, "metrics": metrics})

    all_episodes = [episode for run in runs for episode in run.get("episodes", [])]
    return {
        "schema_version": SCHEMA_VERSION,
        "runs": len(runs),
        "metrics": aggregate_episode_metrics(all_episodes),
        "benchmarks": benchmarks,
        "updated_at": max(
            (str(run.get("updated_at")) for run in runs if run.get("updated_at")),
            default=None,
        ),
    }

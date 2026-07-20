"""Flask entry point for the CaP-X dual-benchmark result viewer."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for

try:  # Support both ``python -m viewer.app`` and ``python viewer/app.py``.
    from .adapters import discover_runs, safe_artifact_key, summarize_runs
except ImportError:  # pragma: no cover - only used by the script entry point.
    from adapters import discover_runs, safe_artifact_key, summarize_runs


def _default_repo_root() -> Path:
    configured = os.environ.get("CAPX_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _public_run(run: dict[str, Any], *, include_episodes: bool = True) -> dict[str, Any]:
    """Copy a normalized run and replace artifact keys with same-origin URLs."""

    public = copy.deepcopy({key: value for key, value in run.items() if not key.startswith("_")})
    episodes = public.get("episodes", [])
    for episode in episodes:
        artifacts = episode.get("artifacts", {})
        for label, key in list(artifacts.items()):
            artifacts[label] = (
                url_for("artifact", run_id=public["run_id"], artifact_path=key)
                if isinstance(key, str) and key
                else None
            )
    if not include_episodes:
        public.pop("episodes", None)
    return public


def create_app(repo_root: Path | str | None = None, *, testing: bool = False) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    # Do not redirect encoded/doubled slashes into a canonical artifact URL.
    # Suspicious artifact paths should terminate as 404 at the first request.
    app.url_map.merge_slashes = False
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _default_repo_root()
    app.config.update(CAPX_REPO_ROOT=root, TESTING=testing, JSON_SORT_KEYS=False)

    def indexed_runs() -> list[dict[str, Any]]:
        return discover_runs(app.config["CAPX_REPO_ROOT"])

    def find_run(run_id: str) -> dict[str, Any]:
        run = next((item for item in indexed_runs() if item.get("run_id") == run_id), None)
        if run is None:
            abort(404, description=f"Unknown run: {run_id}")
        return run

    @app.after_request
    def set_response_headers(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "repo_root": str(app.config["CAPX_REPO_ROOT"]),
                "repo_root_exists": app.config["CAPX_REPO_ROOT"].is_dir(),
            }
        )

    @app.get("/api/summary")
    def summary():
        return jsonify(summarize_runs(indexed_runs()))

    @app.get("/api/runs")
    def runs():
        selected = indexed_runs()
        benchmark = request.args.get("benchmark", "").strip().lower()
        status = request.args.get("status", "").strip().lower()
        task_query = request.args.get("task", "").strip().lower()
        if benchmark:
            selected = [run for run in selected if run.get("benchmark") == benchmark]
        if status:
            selected = [run for run in selected if run.get("status") == status]
        if task_query:
            def searchable(run: dict[str, Any]) -> str:
                config = run.get("config", {})
                episode_tasks = [str(item.get("task") or "") for item in run.get("episodes", [])]
                return " ".join(
                    [str(config.get("task") or ""), str(config.get("suite") or ""), *episode_tasks]
                ).lower()

            selected = [run for run in selected if task_query in searchable(run)]
        return jsonify({"runs": [_public_run(run, include_episodes=False) for run in selected]})

    @app.get("/api/runs/<run_id>")
    def run_detail(run_id: str):
        return jsonify(_public_run(find_run(run_id)))

    @app.get("/api/runs/<run_id>/episodes/<episode_id>")
    def episode_detail(run_id: str, episode_id: str):
        public = _public_run(find_run(run_id))
        episode = next(
            (item for item in public.get("episodes", []) if item.get("episode_id") == episode_id),
            None,
        )
        if episode is None:
            abort(404, description=f"Unknown episode: {episode_id}")
        return jsonify({"run_id": run_id, "benchmark": public["benchmark"], "episode": episode})

    @app.get("/artifacts/<run_id>/<path:artifact_path>")
    def artifact(run_id: str, artifact_path: str):
        # Reject both direct and percent-decoded traversal before touching disk.
        if not safe_artifact_key(artifact_path):
            abort(404)
        run = find_run(run_id)
        registered = run.get("_artifact_files", {})
        target_value = registered.get(artifact_path)
        if not isinstance(target_value, str):
            abort(404)
        target = Path(target_value).resolve()
        repo = app.config["CAPX_REPO_ROOT"].resolve()
        try:
            target.relative_to(repo)
        except ValueError:
            abort(404)
        if not target.is_file():
            abort(404)
        return send_file(target, conditional=True, etag=True, max_age=0)

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/") or request.path.startswith("/artifacts/"):
            return jsonify({"error": "not_found", "message": getattr(error, "description", "Not found")}), 404
        return error

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve CaP-X benchmark results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--repo-root", default=os.environ.get("CAPX_REPO_ROOT"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app = create_app(args.repo_root)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

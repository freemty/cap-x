"""Build the normalized exp01b result manifest consumed by the viewer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/exp01b")
    parser.add_argument("--manifest", default="exp/exp01b/results/manifest.json")
    parser.add_argument("--summary", default="exp/exp01b/results/summary.md")
    args = parser.parse_args()

    # Import here so this utility remains cheap to inspect in simulator envs.
    from viewer.results import discover_runs

    runs = discover_runs((REPO_ROOT / args.output_root).resolve(), repo_root=REPO_ROOT)
    manifest = {
        "schema": "capx.result.v1",
        "runs": [run.to_dict() for run in runs],
    }
    destination = (REPO_ROOT / args.manifest).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    lines = [
        "# exp01b Results",
        "",
        "Both rows below are real CaP-X `CodeExecutionEnvBase` trials on xdlab23;",
        "the earlier `exp01a` direct RoboTwin oracle is not included.",
        "",
        "| Benchmark | Task | Code execution | Plan | Task success | Reward | Elapsed | Video |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        data = run.to_dict()
        episode = data["episodes"][0]
        metrics = episode["metrics"]
        artifact = episode["artifacts"].get("video")
        plan = metrics.get("plan_success")
        plan_text = "N/A" if plan is None else str(bool(plan))
        elapsed = metrics.get("elapsed_seconds")
        elapsed_text = "N/A" if elapsed is None else f"{elapsed:.2f} s"
        lines.append(
            "| {benchmark} | `{task_name}` | {code} | {plan} | {task_success} | {reward:.3f} | {elapsed} | `{video}` |".format(
                benchmark=data["benchmark"],
                task_name=data["config"].get("task"),
                code=bool(metrics.get("code_execution_success")),
                plan=plan_text,
                task_success=bool(metrics.get("task_success")),
                reward=float(metrics.get("reward") or 0.0),
                elapsed=elapsed_text,
                video=artifact,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Both simulators completed one repository-oracle program through the CaP-X trial harness with `sandbox_rc=0`, reward 1.0, and task success.",
            "- RoboTwin additionally reports planner success because its reward is the conjunction of planner and task success. LIBERO has no separate planner predicate in this integration.",
            "- These trials establish adapter and execution correctness. They do not measure learned-policy quality or constitute an RL training result.",
        ]
    )
    summary_path = (REPO_ROOT / args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(runs)} runs to {destination}")
    print(f"Wrote quantitative summary to {summary_path}")


if __name__ == "__main__":
    main()

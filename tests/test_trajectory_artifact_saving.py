from __future__ import annotations

from pathlib import Path

from capx.utils.launch_utils import _save_trial_artifacts
from capx.visualization.trajectory import ArmState, TrajectoryArtifact, TrajectoryRecorder


def test_trial_artifacts_include_replayable_trajectory(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder(metadata={"simulator": "test"})
    recorder.append_executed(
        step=2,
        arms={
            "left": ArmState(
                joint_names=("joint",),
                joint_positions=(0.25,),
            )
        },
    )

    code_path = _save_trial_artifacts(
        {"output_dir": str(tmp_path)},
        trial=1,
        sandbox_rc=0,
        reward=1.0,
        task_completed=True,
        final_code="RESULT = True\n",
        raw_code=None,
        all_responses=[],
        log_lines=["complete"],
        visual_feedback_imgs=[],
        trajectory_artifact=recorder.snapshot(),
    )

    assert code_path is not None
    trajectory_path = Path(code_path).with_name("trajectory.json")
    restored = TrajectoryArtifact.load_json(trajectory_path)
    assert restored.metadata["simulator"] == "test"
    assert restored.samples_for("executed")[0].step == 2

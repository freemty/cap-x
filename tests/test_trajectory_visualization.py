from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from capx.visualization.replay import build_parser, _joint_configuration_for_urdf
from capx.visualization.trajectory import (
    ArmState,
    EndEffectorPose,
    FeasibilityMetrics,
    TrajectoryArtifact,
    TrajectoryLayer,
    TrajectoryRecorder,
)
from capx.visualization.viser_trajectory import ViserTrajectoryRenderer


def pose(
    x: float,
    y: float = 0.0,
    z: float = 0.2,
    *,
    frame_id: str = "world",
) -> EndEffectorPose:
    return EndEffectorPose(
        position=(x, y, z),
        wxyz=(2.0, 0.0, 0.0, 0.0),
        frame_id=frame_id,
    )


def arm_state(
    x: float,
    *,
    arm: str = "",
    include_pose: bool = True,
) -> ArmState:
    return ArmState(
        arm=arm,
        joint_names=("joint_a", "joint_b"),
        joint_positions=(x, -x),
        ee_pose=pose(x) if include_pose else None,
        gripper=0.5,
    )


def test_record_layers_summary_and_metadata_only_raw_sample() -> None:
    recorder = TrajectoryRecorder(max_samples_per_layer=4, metadata={"episode": 3})
    recorder.append_raw(step=0, payload={"primitive": "grasp", "args": np.array([1, 2])})
    recorder.append_planned(step=0, arms={"left": arm_state(0.0, include_pose=False)})
    recorder.append_commanded(step=0, arms={"left": arm_state(0.1)})
    executed = recorder.append_executed(
        step=0,
        arms={
            "left": arm_state(0.08),
            "right": ArmState(ee_pose=pose(-0.2), gripper=1.0),
        },
        feasibility=FeasibilityMetrics(
            ik_ok=True,
            joint_limit_ok=True,
            collision_free=False,
            min_clearance_m=-0.002,
            tracking_error_m=0.02,
        ),
    )

    assert executed.arm_state("left").arm == "left"
    assert executed.arm_state("right").joint_names == ()
    assert executed.arm_state("missing") is None
    assert executed.arms[0].arm == "left"
    assert executed.arms[1].arm == "right"
    assert executed.arms[0].ee_pose.wxyz == (1.0, 0.0, 0.0, 0.0)

    summary = recorder.summary()
    assert summary["num_samples"] == 4
    assert summary["layers"] == {
        "raw": 1,
        "planned": 1,
        "commanded": 1,
        "executed": 1,
    }
    assert summary["arms"] == ["left", "right"]
    assert summary["frames"] == ["world"]
    assert summary["feasibility"]["collision_free"] == {
        "true": 0,
        "false": 1,
        "unknown": 3,
    }
    assert summary["feasibility"]["min_clearance_m"]["min"] == -0.002
    # Summary is directly safe to return through a web API.
    json.dumps(summary, allow_nan=False)


def test_bounded_history_reset_and_subscriber_lifecycle() -> None:
    recorder = TrajectoryRecorder(max_samples_per_layer=2, metadata={"seed": 1})
    seen = []
    unsubscribe = recorder.subscribe(seen.append)
    for step in range(4):
        recorder.append_executed(step=step, arms={"left": arm_state(float(step))})

    artifact = recorder.snapshot()
    assert [sample.step for sample in artifact.samples] == [2, 3]
    assert [sample.step for sample in seen] == [0, 1, 2, 3]

    unsubscribe()
    unsubscribe()
    recorder.append_executed(step=4)
    assert len(seen) == 4

    recorder.reset()
    assert len(recorder) == 0
    assert recorder.metadata == {"seed": 1}
    recorder.append_raw(payload={"after_reset": True})
    assert recorder.snapshot().samples[0].sequence == 0

    recorder.reset(metadata={"seed": 9})
    assert recorder.metadata == {"seed": 9}


def test_artifact_json_round_trip(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder(metadata={"simulator": "fake", "seed": np.int64(7)})
    recorder.append_planned(
        step=8,
        timestamp_s=1.25,
        arms={"left": arm_state(0.4)},
        feasibility=FeasibilityMetrics(ik_ok=True, metadata={"cost": np.float32(2.5)}),
        metadata={"chunk": (1, 2)},
    )
    path = tmp_path / "trajectory.json"
    recorder.save_json(path)

    restored = TrajectoryArtifact.load_json(path)
    assert restored.to_dict() == recorder.snapshot().to_dict()
    assert restored.samples[0].arm_state("left").ee_pose.frame_id == "world"
    assert TrajectoryArtifact.loads(restored.dumps()).to_dict() == restored.to_dict()
    assert restored.samples_for(layer="planned", arm="left", through_step=8)
    assert restored.samples_for(layer="executed") == ()

    legacy_data = restored.to_dict()
    del legacy_data["samples"][0]["arms"][0]["ee_pose"]["frame_id"]
    migrated = TrajectoryArtifact.from_dict(legacy_data)
    assert migrated.samples[0].arm_state("left").ee_pose.frame_id == "world"

    legacy_data["metadata"]["simulator"] = "libero"
    migrated_libero = TrajectoryArtifact.from_dict(legacy_data)
    assert (
        migrated_libero.samples[0].arm_state("left").ee_pose.frame_id
        == "robot0_base"
    )
    legacy_data["metadata"]["eef_frame_id"] = "panda_link0"
    configured_frame = TrajectoryArtifact.from_dict(legacy_data)
    assert (
        configured_frame.samples[0].arm_state("left").ee_pose.frame_id
        == "panda_link0"
    )


def test_end_effector_pose_frame_round_trip_and_legacy_default() -> None:
    pose_in_base = EndEffectorPose(
        position=(0.1, 0.2, 0.3),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        frame_id="robot_base",
    )

    assert EndEffectorPose.from_dict(pose_in_base.to_dict()) == pose_in_base
    assert EndEffectorPose.from_dict(
        {"position": [0.0, 0.0, 0.0], "wxyz": [1.0, 0.0, 0.0, 0.0]}
    ).frame_id == "world"
    with pytest.raises(ValueError, match="frame_id"):
        EndEffectorPose(
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            frame_id=" ",
        )


def test_recorder_is_thread_safe_and_sequences_are_unique() -> None:
    recorder = TrajectoryRecorder(max_samples_per_layer=512)

    def append_batch(worker: int) -> None:
        for index in range(100):
            recorder.append_executed(
                step=worker * 100 + index,
                arms={"left": arm_state(float(index), include_pose=False)},
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append_batch, range(4)))

    samples = recorder.snapshot().samples
    assert len(samples) == 400
    assert len({sample.sequence for sample in samples}) == 400
    assert [sample.sequence for sample in samples] == list(range(400))


def test_core_package_import_does_not_import_viser() -> None:
    code = (
        "import sys; import capx.visualization; "
        "assert not any(x == 'viser' or x.startswith('viser.') for x in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


class FakeHandle:
    def __init__(self, *, name: str = "", value: Any = None, content: str = "") -> None:
        self.name = name
        self.value = value
        self.content = content
        self.removed = False
        self.callbacks = []

    def on_update(self, callback):  # type: ignore[no-untyped-def]
        self.callbacks.append(callback)
        return callback

    def trigger(self, value: Any) -> None:
        self.value = value
        for callback in tuple(self.callbacks):
            callback(SimpleNamespace(target=self))

    def remove(self) -> None:
        self.removed = True


class FakeScene:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any], FakeHandle]] = []

    def add_line_segments(self, name: str, **kwargs: Any) -> FakeHandle:
        handle = FakeHandle(name=name)
        self.calls.append(("line", name, kwargs, handle))
        return handle

    def add_batched_axes(self, name: str, **kwargs: Any) -> FakeHandle:
        handle = FakeHandle(name=name)
        self.calls.append(("axes", name, kwargs, handle))
        return handle


class FakeGui:
    def __init__(self) -> None:
        self.handles: list[FakeHandle] = []

    def add_slider(self, label: str, **kwargs: Any) -> FakeHandle:
        handle = FakeHandle(name=label, value=kwargs["initial_value"])
        self.handles.append(handle)
        return handle

    def add_checkbox(self, label: str, **kwargs: Any) -> FakeHandle:
        handle = FakeHandle(name=label, value=kwargs["initial_value"])
        self.handles.append(handle)
        return handle

    def add_markdown(self, content: str) -> FakeHandle:
        handle = FakeHandle(content=content)
        self.handles.append(handle)
        return handle


class FakeServer:
    def __init__(self) -> None:
        self.scene = FakeScene()
        self.gui = FakeGui()
        self.atomic_entries = 0

    @contextmanager
    def atomic(self):  # type: ignore[no-untyped-def]
        self.atomic_entries += 1
        yield


def populated_recorder() -> TrajectoryRecorder:
    recorder = TrajectoryRecorder(max_samples_per_layer=8)
    for step in range(2):
        recorder.append_planned(
            step=step,
            arms={"left": arm_state(0.1 * step)},
            feasibility=FeasibilityMetrics(ik_ok=True),
        )
        recorder.append_executed(
            step=step,
            arms={"left": arm_state(0.08 * step)},
            feasibility=FeasibilityMetrics(
                collision_free=False if step == 1 else True
            ),
        )
    return recorder


def test_renderer_batches_paths_axes_timeline_and_feasibility_colors() -> None:
    recorder = populated_recorder()
    server = FakeServer()
    robot_states: list[ArmState] = []
    renderer = ViserTrajectoryRenderer(
        server,
        source=recorder.snapshot(),
        robot_state_setters={"left": robot_states.append},
    )

    lines = [call for call in server.scene.calls if call[0] == "line"]
    axes = [call for call in server.scene.calls if call[0] == "axes"]
    assert len(lines) == 2
    assert len(axes) == 2
    assert all(call[2]["points"].shape == (1, 2, 3) for call in lines)
    assert all(call[2]["batched_positions"].shape == (2, 3) for call in axes)
    executed_line = next(call for call in lines if "/executed/" in call[1])
    assert np.all(executed_line[2]["colors"] == np.array([225, 45, 45]))
    assert robot_states[-1].joint_positions == (0.08, -0.08)
    assert "Flagged infeasible:** 1" in renderer.status_handle.content
    assert "Frame:** world" in renderer.status_handle.content
    assert "Planner:** unknown" in renderer.status_handle.content
    assert "Task:** unknown" in renderer.status_handle.content

    replacement_states: list[ArmState] = []
    renderer.set_robot_state_setter("left", replacement_states.append)
    assert replacement_states[-1].joint_positions == (0.08, -0.08)
    renderer.set_robot_state_setter("left", None)

    call_count_before_scrub = len(server.scene.calls)
    renderer.set_timestep(0)
    assert not [
        call
        for call in server.scene.calls[call_count_before_scrub:]
        if call[0] == "line"
    ]
    assert "Timestep:** 0 / 1" in renderer.status_handle.content

    renderer.set_layer_visible(TrajectoryLayer.PLANNED, False)
    assert renderer.layer_handles[TrajectoryLayer.PLANNED].value is False
    renderer.axes_handle.trigger(False)
    assert all("/axes" not in key for key in renderer._scene_handles)

    owned_gui = tuple(server.gui.handles)
    owned_scene = tuple(renderer._scene_handles.values())
    renderer.close()
    renderer.close()
    assert all(handle.removed for handle in owned_gui)
    assert all(handle.removed for handle in owned_scene)
    assert renderer._scene_handles == {}


def test_live_renderer_subscribes_and_close_unsubscribes() -> None:
    recorder = TrajectoryRecorder(max_samples_per_layer=4)
    server = FakeServer()
    renderer = ViserTrajectoryRenderer(server, source=recorder)
    initial_atomic_entries = server.atomic_entries

    recorder.append_executed(step=0, arms={"left": arm_state(0.0)})
    recorder.append_executed(step=1, arms={"left": arm_state(0.1)})
    assert server.atomic_entries > initial_atomic_entries
    assert any(call[0] == "line" for call in server.scene.calls)

    old_slider = renderer.timeline_handle
    recorder.append_executed(step=10, arms={"left": arm_state(0.2)})
    assert old_slider.removed is True
    assert renderer._timeline_max >= 10

    renderer.close()
    calls_after_close = len(server.scene.calls)
    recorder.append_executed(step=2, arms={"left": arm_state(0.2)})
    assert len(server.scene.calls) == calls_after_close


def test_renderer_breaks_plan_segments_and_does_not_connect_raw_targets() -> None:
    recorder = TrajectoryRecorder(max_samples_per_layer=16)
    for step, x in enumerate((0.0, 1.0)):
        recorder.append_planned(
            step=step,
            arms={"left": ArmState(ee_pose=pose(x))},
            metadata={"plan_id": "plan-a"},
        )
    for step, x in enumerate((100.0, 101.0), start=2):
        recorder.append_planned(
            step=step,
            arms={"left": ArmState(ee_pose=pose(x))},
            metadata={"plan_id": "plan-b"},
        )
    for step, x in enumerate((10.0, 20.0)):
        recorder.append_raw(step=step, arms={"left": ArmState(ee_pose=pose(x))})

    server = FakeServer()
    ViserTrajectoryRenderer(server, source=recorder.snapshot())
    planned_lines = [
        call
        for call in server.scene.calls
        if call[0] == "line" and "/planned/" in call[1]
    ]
    raw_lines = [
        call for call in server.scene.calls if call[0] == "line" and "/raw/" in call[1]
    ]
    raw_axes = [
        call for call in server.scene.calls if call[0] == "axes" and "/raw/" in call[1]
    ]

    assert len(planned_lines) == 2
    assert all(call[2]["points"].shape == (1, 2, 3) for call in planned_lines)
    assert all(np.max(np.abs(np.diff(call[2]["points"], axis=1))) <= 1.0 for call in planned_lines)
    assert raw_lines == []
    assert len(raw_axes) == 1


def test_segment_id_takes_priority_over_plan_id() -> None:
    recorder = TrajectoryRecorder(max_samples_per_layer=8)
    for step in range(3):
        recorder.append_commanded(
            step=step,
            arms={"left": ArmState(ee_pose=pose(float(step)))},
            metadata={"segment_id": "motion-1", "plan_id": f"plan-{step}"},
        )
    server = FakeServer()
    ViserTrajectoryRenderer(server, source=recorder.snapshot())
    lines = [
        call
        for call in server.scene.calls
        if call[0] == "line" and "/commanded/" in call[1]
    ]
    assert len(lines) == 1
    assert lines[0][2]["points"].shape == (2, 2, 3)


def test_renderer_never_overlays_different_pose_frames() -> None:
    recorder = TrajectoryRecorder(max_samples_per_layer=8)
    for step, x in enumerate((0.0, 1.0)):
        recorder.append_executed(
            step=step,
            arms={"left": ArmState(ee_pose=pose(x, frame_id="world"))},
        )
    for step, x in enumerate((100.0, 101.0), start=2):
        recorder.append_executed(
            step=step,
            arms={"left": ArmState(ee_pose=pose(x, frame_id="robot_base"))},
        )

    server = FakeServer()
    renderer = ViserTrajectoryRenderer(server, source=recorder.snapshot())
    assert renderer._active_frame_id == "world"
    assert all("/world/" in key for key in renderer._scene_handles)
    assert "Frame:** world (hidden: robot_base)" in renderer.status_handle.content

    renderer.set_frame_id("robot_base")
    assert renderer._active_frame_id == "robot_base"
    assert all("/robot_base/" in key for key in renderer._scene_handles)
    base_line = [
        call
        for call in server.scene.calls
        if call[0] == "line" and "/robot_base/" in call[1]
    ][-1]
    assert float(base_line[2]["points"].min()) >= 0.0
    assert float(base_line[2]["points"][0, :, 0].min()) >= 100.0
    assert "hidden: world" in renderer.status_handle.content

    renderer.set_frame_id("missing_frame")
    assert renderer._active_frame_id == "missing_frame"
    assert renderer._scene_handles == {}
    assert "missing_frame (not present; available: robot_base, world)" in (
        renderer.status_handle.content
    )


def test_renderer_status_shows_latest_completion_and_tracking_metrics() -> None:
    recorder = TrajectoryRecorder(max_samples_per_layer=8)
    recorder.append_executed(
        step=0,
        arms={"left": ArmState(ee_pose=pose(0.0))},
        feasibility=FeasibilityMetrics(
            planner_success=True,
            task_success=False,
            tracking_error_m=0.04,
        ),
    )
    recorder.append_executed(
        step=1,
        arms={"left": ArmState(ee_pose=pose(0.1))},
        feasibility=FeasibilityMetrics(
            planner_success=False,
            task_success=True,
            tracking_error_m=0.01,
        ),
    )
    recorder.append_executed(
        step=2,
        arms={"left": ArmState(ee_pose=pose(0.2))},
        feasibility=FeasibilityMetrics(tracking_error_m=0.02),
    )
    server = FakeServer()
    renderer = ViserTrajectoryRenderer(server, source=recorder.snapshot())

    assert "Planner:** failed" in renderer.status_handle.content
    assert "Task:** success" in renderer.status_handle.content
    assert "Tracking error:** latest 0.0200 m / max 0.0400 m" in (
        renderer.status_handle.content
    )

    renderer.set_timestep(0)
    assert "Planner:** success" in renderer.status_handle.content
    assert "Task:** incomplete" in renderer.status_handle.content
    assert "Tracking error:** latest 0.0400 m / max 0.0400 m" in (
        renderer.status_handle.content
    )


def test_replay_joint_mapping_is_strict_and_supports_libero_panda_aliases() -> None:
    robot0_names = tuple(f"robot0_joint{index}" for index in range(1, 8))
    panda_names = tuple(f"panda_joint{index}" for index in range(1, 8))
    state = ArmState(joint_names=robot0_names, joint_positions=tuple(range(1, 8)))

    np.testing.assert_array_equal(
        _joint_configuration_for_urdf(state, panda_names),
        np.arange(1, 8),
    )
    np.testing.assert_array_equal(
        _joint_configuration_for_urdf(state, tuple(reversed(panda_names))),
        np.arange(7, 0, -1),
    )

    exact_state = ArmState(
        joint_names=("shoulder", "elbow"), joint_positions=(0.2, 0.7)
    )
    np.testing.assert_array_equal(
        _joint_configuration_for_urdf(exact_state, ("elbow", "shoulder")),
        np.asarray([0.7, 0.2]),
    )

    unknown_aliases = tuple(f"other_joint{index}" for index in range(1, 8))
    assert _joint_configuration_for_urdf(state, unknown_aliases) is None
    assert _joint_configuration_for_urdf(state, panda_names + ("finger",)) is None
    short_state = ArmState(
        joint_names=robot0_names[:6], joint_positions=tuple(range(1, 7))
    )
    assert _joint_configuration_for_urdf(short_state, panda_names[:6]) is None


def test_replay_cli_parser() -> None:
    args = build_parser().parse_args(
        ["episode.json", "--host", "127.0.0.1", "--port", "9090", "--urdf", "robot.urdf"]
    )
    assert args.artifact == Path("episode.json")
    assert args.host == "127.0.0.1"
    assert args.port == 9090
    assert args.urdf == Path("robot.urdf")


@pytest.mark.parametrize(
    ("constructor", "match"),
    [
        (lambda: EndEffectorPose((0, 0, 0), (0, 0, 0, 0)), "non-zero norm"),
        (lambda: EndEffectorPose((0, 0, 0), (1, 0, 0, 0), ""), "frame_id"),
        (
            lambda: ArmState(joint_names=("a",), joint_positions=()),
            "equal length",
        ),
        (lambda: TrajectoryRecorder(max_samples_per_layer=0), "positive"),
    ],
)
def test_invalid_trajectory_inputs_are_rejected(constructor, match: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises((TypeError, ValueError), match=match):
        constructor()

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from capx.envs.simulators.robotwin import RoboTwinEnv
from capx.visualization.trajectory import TrajectoryLayer


class FakeArmTag(str):
    pass


class FakeRootPose:
    def __init__(self) -> None:
        self.p = np.asarray([0.0, 0.0, 0.0])
        self.q = np.asarray([1.0, 0.0, 0.0, 0.0])


class FakeAction:
    def __init__(
        self,
        arm: str,
        *,
        target_pose: list[float] | None = None,
        target_gripper_pos: float | None = None,
    ) -> None:
        self.arm_tag = FakeArmTag(arm)
        self.action = "move" if target_pose is not None else "gripper"
        self.target_pose = target_pose
        self.target_gripper_pos = target_gripper_pos
        self.args: dict[str, Any] = {"constraint_pose": None}


class FakeRobot:
    def __init__(self) -> None:
        self.is_dual_arm = True
        self.left_arm_joints_name = ["left_shoulder", "left_elbow"]
        self.right_arm_joints_name = ["right_shoulder", "right_elbow"]
        self.drive = {
            "left": np.asarray([0.0, 0.0]),
            "right": np.asarray([0.0, 0.0]),
        }
        self.real = {
            "left": np.asarray([-0.05, -0.05]),
            "right": np.asarray([-0.05, -0.05]),
        }
        self.gripper = {"left": 1.0, "right": 1.0}

    def set_arm_joints(
        self, target_position: np.ndarray, target_velocity: np.ndarray, arm: str
    ) -> None:
        del target_velocity
        self.drive[arm] = np.asarray(target_position, dtype=float).copy()

    def get_left_arm_jointState(self) -> list[float]:
        return [*self.drive["left"], self.gripper["left"]]

    def get_right_arm_jointState(self) -> list[float]:
        return [*self.drive["right"], self.gripper["right"]]

    def get_left_arm_real_jointState(self) -> list[float]:
        return [*self.real["left"], self.gripper["left"]]

    def get_right_arm_real_jointState(self) -> list[float]:
        return [*self.real["right"], self.gripper["right"]]

    def get_left_gripper_val(self) -> float:
        return self.gripper["left"]

    def get_right_gripper_val(self) -> float:
        return self.gripper["right"]

    def get_left_ee_pose(self) -> list[float]:
        return [*self.real["left"], 0.5, 1.0, 0.0, 0.0, 0.0]

    def get_right_ee_pose(self) -> list[float]:
        return [*self.real["right"], 0.5, 1.0, 0.0, 0.0, 0.0]


class FakeScene:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        # Deliberately retain tracking error so drive targets and measured qpos
        # are observably different in the recorder.
        for arm in ("left", "right"):
            self.robot.real[arm] = self.robot.drive[arm] - 0.05


class NativeFakeTask:
    instances: list[NativeFakeTask] = []

    def __init__(self) -> None:
        self.robot = FakeRobot()
        self.scene = FakeScene(self.robot)
        self.plan_success = True
        self.task_success = False
        self.close_calls = 0
        NativeFakeTask.instances.append(self)

    def setup_demo(self, **kwargs: Any) -> None:
        robot_root = Path(kwargs["left_robot_file"])
        self.robot.left_urdf_path = str(robot_root / "robot.urdf")
        self.robot.right_urdf_path = str(robot_root / "robot.urdf")
        self.robot.left_ee_name = "left_elbow"
        self.robot.right_ee_name = "right_elbow"
        self.robot.left_entity_origion_pose = FakeRootPose()
        self.robot.right_entity_origion_pose = FakeRootPose()
        self.robot.left_gripper_bias = 0.12
        self.robot.right_gripper_bias = 0.12
        self.robot.left_global_trans_matrix = np.diag([1.0, -1.0, -1.0])
        self.robot.right_global_trans_matrix = np.diag([1.0, -1.0, -1.0])
        self.robot.left_delta_matrix = np.eye(3)
        self.robot.right_delta_matrix = np.eye(3)
        self.robot.left_gripper_name = {"base": "left_gripper"}
        self.robot.right_gripper_name = {"base": "right_gripper"}
        self.robot.left_gripper_scale = [-0.01, 0.045]
        self.robot.right_gripper_scale = [-0.01, 0.045]

    def _take_picture(self) -> None:
        pass

    def get_obs(self) -> dict[str, Any]:
        return {
            "observation": {
                "head_camera": {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)}
            }
        }

    @staticmethod
    def _groups(
        actions_by_arm1: tuple[FakeArmTag, list[FakeAction]],
        actions_by_arm2: tuple[FakeArmTag, list[FakeAction]] | None,
    ) -> list[tuple[FakeArmTag, list[FakeAction]]]:
        return [
            group for group in (actions_by_arm1, actions_by_arm2) if group is not None
        ]

    def move(
        self,
        actions_by_arm1: tuple[FakeArmTag, list[FakeAction]],
        actions_by_arm2: tuple[FakeArmTag, list[FakeAction]] | None = None,
        save_freq: int | None = None,
    ) -> bool:
        del save_freq
        control_seq: dict[str, Any] = {
            "left_arm": None,
            "left_gripper": None,
            "right_arm": None,
            "right_gripper": None,
        }
        for arm_tag, actions in self._groups(actions_by_arm1, actions_by_arm2):
            arm = str(arm_tag)
            if not actions:
                continue
            sign = 1.0 if arm == "left" else -1.0
            control_seq[f"{arm}_arm"] = {
                "position": np.asarray(
                    [[0.1 * sign, 0.2 * sign], [0.3 * sign, 0.4 * sign]]
                ),
                "velocity": np.asarray([[0.1, 0.1], [0.2, 0.2]]),
            }
        return self.take_dense_action(control_seq)

    def take_dense_action(self, control_seq: dict[str, Any]) -> bool:
        max_length = max(
            (
                value["position"].shape[0]
                for key, value in control_seq.items()
                if key.endswith("_arm") and value is not None
            ),
            default=0,
        )
        for index in range(max_length):
            for arm in ("left", "right"):
                arm_control = control_seq[f"{arm}_arm"]
                if arm_control is not None and index < len(arm_control["position"]):
                    self.robot.set_arm_joints(
                        arm_control["position"][index],
                        arm_control["velocity"][index],
                        arm,
                    )
            self.scene.step()
        return True

    def move_by_displacement(
        self, *, arm_tag: FakeArmTag, x: float, y: float, z: float
    ) -> tuple[FakeArmTag, list[FakeAction]]:
        return (
            arm_tag,
            [
                FakeAction(
                    str(arm_tag),
                    target_pose=[x, y, z, 1.0, 0.0, 0.0, 0.0],
                )
            ],
        )

    def check_success(self) -> bool:
        return self.task_success

    def close_env(self, *, clear_cache: bool) -> None:
        assert clear_cache is True
        self.close_calls += 1


@pytest.fixture
def robotwin_root(tmp_path: Path) -> Path:
    task_config = tmp_path / "task_config"
    embodiment_root = tmp_path / "embodiments" / "aloha-agilex"
    task_config.mkdir()
    embodiment_root.mkdir(parents=True)
    (task_config / "demo_clean.yml").write_text(
        "domain_randomization:\n  random_embodiment: true\n"
    )
    (task_config / "_embodiment_config.yml").write_text(
        "aloha-agilex:\n  file_path: embodiments/aloha-agilex\n"
    )
    (embodiment_root / "config.yml").write_text(
        "robot: fake-aloha\nurdf_path: robot.urdf\n"
    )
    (embodiment_root / "robot.urdf").write_text(
        """<?xml version="1.0"?>
<robot name="fake_dual_arm">
  <link name="base"/>
  <link name="left_link"/>
  <link name="left_tcp"/>
  <link name="right_link"/>
  <link name="right_tcp"/>
  <joint name="left_shoulder" type="revolute">
    <parent link="base"/><child link="left_link"/>
    <origin xyz="0.2 0.3 0.4"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="left_elbow" type="revolute">
    <parent link="left_link"/><child link="left_tcp"/>
    <origin xyz="0.5 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="right_shoulder" type="revolute">
    <parent link="base"/><child link="right_link"/>
    <origin xyz="-0.2 0.3 0.4"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="right_elbow" type="revolute">
    <parent link="right_link"/><child link="right_tcp"/>
    <origin xyz="-0.5 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>
"""
    )
    return tmp_path


@pytest.fixture
def instrumented_env(
    monkeypatch: pytest.MonkeyPatch, robotwin_root: Path
) -> RoboTwinEnv:
    envs_package = types.ModuleType("envs")
    envs_package.__path__ = []  # type: ignore[attr-defined]
    task_module = types.ModuleType("envs.native_fake")
    task_module.native_fake = NativeFakeTask  # type: ignore[attr-defined]
    utils_module = types.ModuleType("envs.utils")
    utils_module.ArmTag = FakeArmTag  # type: ignore[attr-defined]
    envs_package.native_fake = task_module  # type: ignore[attr-defined]
    envs_package.utils = utils_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "envs", envs_package)
    monkeypatch.setitem(sys.modules, "envs.native_fake", task_module)
    monkeypatch.setitem(sys.modules, "envs.utils", utils_module)
    monkeypatch.syspath_prepend(str(robotwin_root))
    NativeFakeTask.instances.clear()

    # Debug recording must remain usable on workers without Viser.
    def unavailable_server(**kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("Viser intentionally unavailable in this test")

    monkeypatch.setattr("capx.visualization.create_viser_server", unavailable_server)
    env = RoboTwinEnv(
        task_name="native_fake",
        robotwin_root=str(robotwin_root),
        viser_debug=True,
    )
    try:
        yield env
    finally:
        env.close()


def test_native_dense_action_records_dual_arm_control_layers_and_real_state(
    instrumented_env: RoboTwinEnv,
) -> None:
    observation, info = instrumented_env.reset(seed=31)
    task = NativeFakeTask.instances[-1]

    assert instrumented_env.viser_server is None
    assert observation["trajectory"] == info["trajectory"]
    assert info["trajectory"]["metadata"]["embodiment"] == "aloha-agilex"
    assert info["trajectory"]["metadata"]["robot"]["dual_arm"] is True
    assert info["trajectory"]["metadata"]["robot"]["planned_fk"] is True
    assert info["trajectory"]["metadata"]["robot"]["shared_dual_urdf"] is True
    assert info["trajectory"]["metadata"]["arms"] == {
        "left": {"joint_names": ["left_shoulder", "left_elbow"]},
        "right": {"joint_names": ["right_shoulder", "right_elbow"]},
    }

    left_action = FakeAction(
        "left", target_pose=[0.3, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]
    )
    right_action = FakeAction(
        "right", target_pose=[-0.3, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]
    )
    assert task.move(
        (FakeArmTag("left"), [left_action]),
        (FakeArmTag("right"), [right_action]),
        save_freq=None,
    )

    artifact = instrumented_env.trajectory_snapshot()
    raw_samples = artifact.samples_for(TrajectoryLayer.RAW)
    planned_samples = artifact.samples_for(TrajectoryLayer.PLANNED)
    commanded_samples = artifact.samples_for(TrajectoryLayer.COMMANDED)
    executed_samples = artifact.samples_for(TrajectoryLayer.EXECUTED)

    assert len(raw_samples) == 1
    assert len(planned_samples) == 2
    assert len(commanded_samples) == 4  # initial + target + two drive targets
    assert len(executed_samples) == 3  # initial state + two measured physics ticks
    assert raw_samples[0].payload["actions"][0]["action"]["target_pose"][:3] == [
        0.3,
        0.1,
        0.2,
    ]

    planned_left = planned_samples[-1].arm_state("left")
    planned_right = planned_samples[-1].arm_state("right")
    assert planned_left is not None and planned_right is not None
    assert planned_left.joint_names == ("left_shoulder", "left_elbow")
    assert planned_right.joint_names == ("right_shoulder", "right_elbow")
    assert planned_left.joint_positions == pytest.approx((0.3, 0.4))
    assert planned_right.joint_positions == pytest.approx((-0.3, -0.4))
    assert planned_left.ee_pose is not None
    assert planned_right.ee_pose is not None
    assert planned_left.ee_pose.position == pytest.approx(
        (0.2 + 0.5 * np.cos(0.3), 0.3 + 0.5 * np.sin(0.3), 0.4)
    )
    first_planned_left = planned_samples[0].arm_state("left")
    assert first_planned_left is not None and first_planned_left.ee_pose is not None
    assert planned_left.ee_pose.position != pytest.approx(
        first_planned_left.ee_pose.position
    )

    high_level_target = next(
        sample
        for sample in commanded_samples
        if sample.metadata.get("source") == "high_level_target"
    )
    assert high_level_target.arm_state("left").ee_pose.position == pytest.approx(
        (0.3, 0.1, 0.2)
    )

    final_drive = next(
        sample
        for sample in reversed(commanded_samples)
        if sample.metadata.get("source") == "drive_target"
    )
    final_real = executed_samples[-1]
    assert final_drive.arm_state("left").joint_positions == pytest.approx((0.3, 0.4))
    assert final_real.arm_state("left").joint_positions == pytest.approx((0.25, 0.35))
    assert final_real.arm_state("left").ee_pose.position == pytest.approx(
        (0.25, 0.35, 0.5)
    )
    assert final_drive.arm_state("left").metadata["state_source"] == "drive_target"
    assert final_real.arm_state("left").metadata["state_source"] == "real_qpos"
    assert {sample.metadata["segment_id"] for sample in planned_samples} == {"plan-0"}
    assert {
        sample.metadata["segment_id"]
        for sample in commanded_samples
        if sample.metadata.get("source") == "drive_target"
    } == {"initial", "plan-0"}
    assert planned_samples[0].feasibility.planner_success is True

    summary = instrumented_env.trajectory_summary()
    assert "samples" not in summary
    assert summary["layers"] == {
        "raw": 1,
        "planned": 2,
        "commanded": 4,
        "executed": 3,
    }
    assert summary["arms"] == ["left", "right"]


def test_live_renderer_updates_at_bounded_physics_stride(
    instrumented_env: RoboTwinEnv,
) -> None:
    instrumented_env.reset(seed=5)
    task = NativeFakeTask.instances[-1]

    class FakeRenderer:
        def __init__(self) -> None:
            self.update_count = 0

        def update(self) -> None:
            self.update_count += 1

        def close(self) -> None:
            pass

    renderer = FakeRenderer()
    instrumented_env._trajectory.renderer = renderer
    instrumented_env._trajectory._render_every_physics_steps = 2

    task.scene.step()
    assert renderer.update_count == 0
    task.scene.step()
    assert renderer.update_count == 1
    assert instrumented_env.trajectory_summary()["layers"]["executed"] == 3


def test_close_restores_every_native_instance_monkeypatch(
    instrumented_env: RoboTwinEnv,
) -> None:
    instrumented_env.reset(seed=7)
    task = NativeFakeTask.instances[-1]
    scene = task.scene

    assert {"move", "take_dense_action", "_take_picture"} <= set(task.__dict__)
    assert "step" in scene.__dict__

    instrumented_env.close()

    assert "move" not in task.__dict__
    assert "take_dense_action" not in task.__dict__
    assert "_take_picture" not in task.__dict__
    assert "step" not in scene.__dict__
    assert task.close_calls == 1
    assert task.move.__func__ is NativeFakeTask.move
    assert scene.step.__func__ is FakeScene.step

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from capx.envs.simulators.robotwin import RoboTwinEnv
from capx.integrations.robotwin import RoboTwinPrivilegedApi


class FakePose:
    def __init__(self, position: list[float], quaternion: list[float] | None = None) -> None:
        self.p = np.asarray(position, dtype=float)
        self.q = np.asarray(quaternion or [1.0, 0.0, 0.0, 0.0], dtype=float)


class FakeActor:
    def __init__(self, position: list[float]) -> None:
        self._pose = FakePose(position)

    def get_pose(self) -> FakePose:
        return self._pose

    def get_functional_point(self, index: int) -> np.ndarray:
        return np.asarray([float(index), 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])


class FakeArmTag(str):
    pass


class FakeTask:
    instances: list[FakeTask] = []

    def __init__(self) -> None:
        self.block_left = FakeActor([-0.2, 0.0, 0.1])
        self.block_right = FakeActor([0.3, 0.0, 0.1])
        self._hidden_actor = FakeActor([0.0, 0.0, 0.0])
        self.non_actor = object()
        self.plan_success = True
        self.task_success = False
        self.move_success = True
        self.setup_calls: list[dict[str, Any]] = []
        self.primitive_calls: list[dict[str, Any]] = []
        self.move_calls: list[dict[str, Any]] = []
        self.close_calls: list[dict[str, Any]] = []
        self.frame_index = 0
        FakeTask.instances.append(self)

    def setup_demo(
        self,
        *,
        now_ep_num: int,
        seed: int,
        is_test: bool,
        **kwargs: Any,
    ) -> None:
        self.setup_calls.append(
            {
                "now_ep_num": now_ep_num,
                "seed": seed,
                "is_test": is_test,
                "kwargs": kwargs,
                "cwd": Path.cwd(),
            }
        )

    def get_obs(self) -> dict[str, Any]:
        self.frame_index += 1
        rgb = np.full((3, 4, 3), self.frame_index, dtype=np.uint8)
        return {"observation": {"head_camera": {"rgb": rgb}}}

    def move(self, action: dict[str, Any], *, save_freq: int | None) -> bool:
        self.move_calls.append(
            {"action": action, "save_freq": save_freq, "cwd": Path.cwd()}
        )
        if save_freq is not None:
            self._take_picture()
        return self.move_success

    def grasp_actor(
        self,
        actor: FakeActor,
        *,
        arm_tag: FakeArmTag,
        pre_grasp_dis: float,
    ) -> dict[str, Any]:
        return self._primitive(
            "grasp_actor",
            actor=actor,
            arm_tag=arm_tag,
            pre_grasp_dis=pre_grasp_dis,
        )

    def place_actor(
        self,
        actor: FakeActor,
        *,
        arm_tag: FakeArmTag,
        target_pose: list[float],
        functional_point_id: int | None,
        pre_dis: float,
        dis: float,
        pre_dis_axis: str,
    ) -> dict[str, Any]:
        return self._primitive(
            "place_actor",
            actor=actor,
            arm_tag=arm_tag,
            target_pose=target_pose,
            functional_point_id=functional_point_id,
            pre_dis=pre_dis,
            dis=dis,
            pre_dis_axis=pre_dis_axis,
        )

    def move_by_displacement(
        self,
        *,
        arm_tag: FakeArmTag,
        x: float,
        y: float,
        z: float,
    ) -> dict[str, Any]:
        return self._primitive(
            "move_by_displacement", arm_tag=arm_tag, x=x, y=y, z=z
        )

    def back_to_origin(self, arm_tag: FakeArmTag) -> dict[str, Any]:
        return self._primitive("back_to_origin", arm_tag=arm_tag)

    def open_gripper(self, arm_tag: FakeArmTag) -> dict[str, Any]:
        return self._primitive("open_gripper", arm_tag=arm_tag)

    def close_gripper(self, arm_tag: FakeArmTag) -> dict[str, Any]:
        return self._primitive("close_gripper", arm_tag=arm_tag)

    def _primitive(self, name: str, **kwargs: Any) -> dict[str, Any]:
        call = {"name": name, **kwargs, "cwd": Path.cwd()}
        self.primitive_calls.append(call)
        return call

    def check_success(self) -> bool:
        return self.task_success

    def close_env(self, *, clear_cache: bool) -> None:
        self.close_calls.append({"clear_cache": clear_cache, "cwd": Path.cwd()})


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
    (embodiment_root / "config.yml").write_text("robot: fake-aloha\n")
    return tmp_path


@pytest.fixture
def robotwin_env(monkeypatch: pytest.MonkeyPatch, robotwin_root: Path) -> RoboTwinEnv:
    envs_package = types.ModuleType("envs")
    envs_package.__path__ = []  # type: ignore[attr-defined]
    task_module = types.ModuleType("envs.fake_stack")
    task_module.fake_stack = FakeTask  # type: ignore[attr-defined]
    utils_module = types.ModuleType("envs.utils")
    utils_module.ArmTag = FakeArmTag  # type: ignore[attr-defined]
    envs_package.fake_stack = task_module  # type: ignore[attr-defined]
    envs_package.utils = utils_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "envs", envs_package)
    monkeypatch.setitem(sys.modules, "envs.fake_stack", task_module)
    monkeypatch.setitem(sys.modules, "envs.utils", utils_module)
    # Prevent RoboTwinEnv from leaving the temporary checkout in sys.path.
    monkeypatch.syspath_prepend(str(robotwin_root))
    FakeTask.instances.clear()

    env = RoboTwinEnv(
        task_name="fake_stack",
        robotwin_root=str(robotwin_root),
        max_steps=2,
        video_stride=3,
    )
    try:
        yield env
    finally:
        env.close()


def test_reset_discovers_dynamic_actors_and_closes_the_previous_task(
    robotwin_env: RoboTwinEnv, robotwin_root: Path
) -> None:
    observation, info = robotwin_env.reset(seed=17, options={"trial": 4})
    first_task = FakeTask.instances[-1]
    setup = first_task.setup_calls[0]

    assert info == {
        "task": "fake_stack",
        "embodiment": "aloha-agilex",
        "seed": 17,
    }
    assert setup["now_ep_num"] == 4
    assert setup["seed"] == 17
    assert setup["is_test"] is True
    assert setup["cwd"] == robotwin_root
    assert setup["kwargs"]["task_name"] == "fake_stack"
    assert setup["kwargs"]["domain_randomization"]["random_embodiment"] is False
    assert setup["kwargs"]["left_embodiment_config"] == {"robot": "fake-aloha"}
    assert setup["kwargs"]["left_robot_file"] == str(
        robotwin_root / "embodiments" / "aloha-agilex"
    )

    assert robotwin_env.actor_names() == ["block_left", "block_right"]
    assert robotwin_env.choose_arm("block_left") == "left"
    assert robotwin_env.choose_arm("block_right") == "right"
    assert robotwin_env.functional_point("block_left", index=5) == [
        5.0,
        2.0,
        3.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert observation["head_camera"]["images"]["rgb"].shape == (3, 4, 3)
    assert sorted(observation["actors"]) == ["block_left", "block_right"]

    first_task.dynamic_block = FakeActor([0.7, 0.0, 0.1])
    assert robotwin_env.actor_names() == ["block_left", "block_right", "dynamic_block"]
    assert robotwin_env.actor_pose("dynamic_block")[:3] == [0.7, 0.0, 0.1]
    with pytest.raises(KeyError, match="Unknown actor 'missing'"):
        robotwin_env.actor_pose("missing")

    _, second_info = robotwin_env.reset(seed=23, options={"trial": 5})
    second_task = FakeTask.instances[-1]
    assert first_task.close_calls == [{"clear_cache": True, "cwd": robotwin_root}]
    assert second_task is not first_task
    assert second_task.setup_calls[0]["seed"] == 23
    assert second_task.setup_calls[0]["now_ep_num"] == 5
    assert second_info["seed"] == 23


def test_unprivileged_observation_does_not_leak_actor_or_task_state(
    robotwin_env: RoboTwinEnv,
) -> None:
    robotwin_env.reset(seed=11)
    robotwin_env.privileged = False

    observation = robotwin_env.get_observation()

    assert set(observation) == {"head_camera"}
    assert "actors" not in observation
    assert "status" not in observation


def test_privileged_api_forwards_named_actor_and_control_arguments(
    robotwin_env: RoboTwinEnv, robotwin_root: Path
) -> None:
    robotwin_env.reset(seed=3)
    task = FakeTask.instances[-1]
    api = RoboTwinPrivilegedApi(robotwin_env)

    expected_functions = {
        "list_actors",
        "get_actor_pose",
        "get_functional_point",
        "choose_arm",
        "grasp_actor",
        "place_actor",
        "move_by_displacement",
        "back_to_origin",
        "open_gripper",
        "close_gripper",
        "get_task_status",
    }
    assert set(api.functions()) == expected_functions
    assert api.list_actors() == ["block_left", "block_right"]
    assert api.choose_arm("block_left") == "left"

    assert api.grasp_actor("block_left", pre_grasp_dis=0.12) is True
    grasp = task.primitive_calls[-1]
    assert grasp["name"] == "grasp_actor"
    assert grasp["actor"] is task.block_left
    assert grasp["arm_tag"] == "left"
    assert grasp["pre_grasp_dis"] == pytest.approx(0.12)

    target_pose = [0.4, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]
    assert api.place_actor(
        "block_right",
        target_pose,
        arm="right",
        functional_point_id=2,
        pre_dis=0.08,
        dis=0.01,
        pre_dis_axis="z",
    ) is True
    place = task.primitive_calls[-1]
    assert place["name"] == "place_actor"
    assert place["actor"] is task.block_right
    assert place["arm_tag"] == "right"
    assert place["target_pose"] is target_pose
    assert place["functional_point_id"] == 2
    assert place["pre_dis"] == pytest.approx(0.08)
    assert place["dis"] == pytest.approx(0.01)
    assert place["pre_dis_axis"] == "z"

    assert api.move_by_displacement("left", x=0.1, y=-0.2, z=0.3) is True
    assert api.back_to_origin("right") is True
    assert api.open_gripper("left") is True
    assert api.close_gripper("right") is True

    assert [call["name"] for call in task.primitive_calls] == [
        "grasp_actor",
        "place_actor",
        "move_by_displacement",
        "back_to_origin",
        "open_gripper",
        "close_gripper",
    ]
    assert all(call["cwd"] == robotwin_root for call in task.primitive_calls)
    assert all(call["cwd"] == robotwin_root for call in task.move_calls)
    assert robotwin_env.status()["sim_steps"] == 6


def test_reward_keeps_planner_and_task_success_separate(robotwin_env: RoboTwinEnv) -> None:
    robotwin_env.reset(seed=9)
    task = FakeTask.instances[-1]

    task.plan_success = False
    task.task_success = True
    assert robotwin_env.task_completed() is True
    assert robotwin_env.compute_reward() == 0.0
    assert robotwin_env.status() == {
        "code_execution_success": None,
        "plan_success": False,
        "task_success": True,
        "sim_steps": 0,
    }

    task.plan_success = True
    task.task_success = False
    assert robotwin_env.compute_reward() == 0.0

    task.task_success = True
    observation, reward, terminated, truncated, status = robotwin_env.step(None)
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert status["plan_success"] is True
    assert status["task_success"] is True
    assert observation["status"]["task_success"] is True

    task.task_success = False
    assert robotwin_env.move_by_displacement("left", x=0.1) is True
    assert robotwin_env.move_by_displacement("left", x=0.1) is True
    _, reward, terminated, truncated, status = robotwin_env.step(None)
    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert status["sim_steps"] == 2


def test_video_capture_uses_move_hook_and_returns_defensive_copies(
    robotwin_env: RoboTwinEnv
) -> None:
    robotwin_env.reset(seed=11)
    task = FakeTask.instances[-1]
    api = RoboTwinPrivilegedApi(robotwin_env)

    robotwin_env.enable_video_capture(True)
    assert robotwin_env.get_video_frame_count() == 1

    assert api.grasp_actor("block_left") is True
    assert task.move_calls[-1]["save_freq"] == 3
    # One initial frame, one through RoboTwin's _take_picture hook, and one
    # explicit post-action frame from RoboTwinEnv._execute.
    assert robotwin_env.get_video_frame_count() == 3

    frames = robotwin_env.get_video_frames()
    frames[0][:] = 255
    assert not np.all(robotwin_env.get_video_frames_range(0, 1)[0] == 255)
    assert len(robotwin_env.get_video_frames(clear=True)) == 3
    assert robotwin_env.get_video_frame_count() == 0

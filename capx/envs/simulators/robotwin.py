from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from capx.envs.base import BaseEnv
from capx.envs.simulators.robotwin_instrumentation import (
    RestorableAttributePatches,
    RoboTwinTrajectoryInstrumentation,
)


class RoboTwinEnv(BaseEnv):
    """CaP-X low-level adapter for a RoboTwin task.

    RoboTwin still resolves some assets relative to its repository root, so all
    simulator calls are made inside a short-lived cwd context.  The surrounding
    CaP-X process keeps its original cwd for artifact writing.
    """

    def __init__(
        self,
        task_name: str,
        embodiment: str = "aloha-agilex",
        task_config: str = "demo_clean",
        robotwin_root: str | None = None,
        privileged: bool = True,
        max_steps: int = 128,
        seed: int | None = None,
        enable_render: bool = False,
        viser_debug: bool = False,
        video_stride: int = 8,
        trajectory_max_samples_per_layer: int = 10_000,
        trajectory_render_every_physics_steps: int = 50,
    ) -> None:
        super().__init__()
        root_value = robotwin_root or os.environ.get("ROBOTWIN_ROOT")
        if not root_value:
            raise ValueError("robotwin_root or ROBOTWIN_ROOT must be set")

        self.robotwin_root = Path(root_value).expanduser().resolve()
        if not self.robotwin_root.is_dir():
            raise FileNotFoundError(f"RoboTwin checkout not found: {self.robotwin_root}")

        self.task_name = task_name
        self.embodiment = embodiment
        self.task_config = task_config
        self.privileged = privileged
        self.max_steps = max_steps
        self.seed = seed
        self.enable_render = enable_render
        self.viser_debug = viser_debug
        self.video_stride = max(1, int(video_stride))

        root_text = str(self.robotwin_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        with self._cwd():
            task_module = importlib.import_module(f"envs.{task_name}")
            self._task_cls = getattr(task_module, task_name)
            self._arm_tag_cls = importlib.import_module("envs.utils").ArmTag

        self._task: Any | None = None
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._sim_step_count = 0
        self._episode_seed: int | None = None
        self._task_patches = RestorableAttributePatches()
        self._trajectory = RoboTwinTrajectoryInstrumentation(
            metadata=self._trajectory_metadata(seed=self.seed),
            max_samples_per_layer=trajectory_max_samples_per_layer,
            render_every_physics_steps=trajectory_render_every_physics_steps,
        )
        self.viser_server: Any | None = None
        self._trajectory_renderer_attempted = False

    @contextmanager
    def _cwd(self) -> Iterator[None]:
        previous = Path.cwd()
        os.chdir(self.robotwin_root)
        try:
            yield
        finally:
            os.chdir(previous)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a mapping in {path}")
        return payload

    def _task_args(self) -> dict[str, Any]:
        args = self._load_yaml(self.robotwin_root / "task_config" / f"{self.task_config}.yml")
        args["task_name"] = self.task_name
        args["render_freq"] = 0
        args["collect_data"] = False
        args["save_data"] = False
        args["save_freq"] = None
        args["eval_video_log"] = False
        args["need_plan"] = True
        args["domain_randomization"]["random_embodiment"] = False

        args["embodiment"] = [self.embodiment]
        embodiment_index = self._load_yaml(
            self.robotwin_root / "task_config" / "_embodiment_config.yml"
        )
        embodiment_root = (
            self.robotwin_root / embodiment_index[self.embodiment]["file_path"]
        ).resolve()
        embodiment_config = self._load_yaml(embodiment_root / "config.yml")
        args["left_robot_file"] = str(embodiment_root)
        args["right_robot_file"] = str(embodiment_root)
        args["left_embodiment_config"] = embodiment_config
        args["right_embodiment_config"] = embodiment_config
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = self.embodiment
        return args

    def _trajectory_metadata(self, *, seed: int | None) -> dict[str, Any]:
        return {
            "simulator": "robotwin",
            "task": self.task_name,
            "embodiment": self.embodiment,
            "task_config": self.task_config,
            "seed": seed,
        }

    def _connect_trajectory_renderer(self) -> None:
        if (
            not self.viser_debug
            or self._trajectory.renderer is not None
            or self._trajectory_renderer_attempted
        ):
            return
        self._trajectory_renderer_attempted = True
        try:
            from capx.visualization import create_viser_server

            self.viser_server = create_viser_server()
        except Exception:
            # Recording remains available when Viser is absent or cannot bind a
            # port (for example in a headless batch worker).
            self.viser_server = None
            return
        connected = self._trajectory.connect_renderer(
            self.viser_server,
            timeline_max=max(1, self.max_steps),
        )
        if not connected:
            server, self.viser_server = self.viser_server, None
            stop = getattr(server, "stop", None)
            if callable(stop):
                with suppress(Exception):
                    stop()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._close_task()
        self._task = self._task_cls()
        self._sim_step_count = 0
        self._frame_buffer.clear()
        self._episode_seed = self.seed if seed is None else seed
        trial = int((options or {}).get("trial", 0))

        with self._cwd():
            self._task.setup_demo(
                now_ep_num=trial,
                seed=int(self._episode_seed or 0),
                is_test=True,
                **self._task_args(),
            )
            # task.move(save_freq=...) calls this hook at a fixed stride.  The
            # adapter owns the frame buffer, so RoboTwin data collection stays off.
            self._task_patches.patch(self._task, "_take_picture", self._record_frame)

        self._trajectory.reset(
            self._task,
            metadata=self._trajectory_metadata(seed=self._episode_seed),
        )
        self._trajectory.install(self._task, self._task_patches)
        self._connect_trajectory_renderer()
        self._trajectory.record_current_state()

        obs = self.get_observation()
        return obs, {
            "task": self.task_name,
            "embodiment": self.embodiment,
            "seed": self._episode_seed,
            "trajectory": self.trajectory_summary(),
        }

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        del action
        reward = self.compute_reward()
        return (
            self.get_observation(),
            reward,
            bool(reward == 1.0),
            self._sim_step_count >= self.max_steps,
            self.status(),
        )

    def _require_task(self) -> Any:
        if self._task is None:
            raise RuntimeError("RoboTwin environment has not been reset")
        return self._task

    @staticmethod
    def _pose_list(pose: Any) -> list[float]:
        if hasattr(pose, "p") and hasattr(pose, "q"):
            return np.concatenate([np.asarray(pose.p), np.asarray(pose.q)]).astype(float).tolist()
        return np.asarray(pose, dtype=float).reshape(-1).tolist()

    def actor_names(self) -> list[str]:
        task = self._require_task()
        names = []
        for name, value in vars(task).items():
            if not name.startswith("_") and hasattr(value, "get_pose"):
                names.append(name)
        return sorted(names)

    def _actor(self, name: str) -> Any:
        task = self._require_task()
        actor = getattr(task, name, None)
        if actor is None or not hasattr(actor, "get_pose"):
            raise KeyError(f"Unknown actor {name!r}; available: {self.actor_names()}")
        return actor

    def actor_pose(self, name: str) -> list[float]:
        with self._cwd():
            return self._pose_list(self._actor(name).get_pose())

    def functional_point(self, name: str, index: int = 1) -> list[float]:
        with self._cwd():
            return self._pose_list(self._actor(name).get_functional_point(index))

    def choose_arm(self, name: str) -> str:
        return "left" if self.actor_pose(name)[0] < 0 else "right"

    def _execute(self, action: Any) -> bool:
        task = self._require_task()
        save_freq = self.video_stride if self._record_frames else None
        with self._cwd():
            ok = bool(task.move(action, save_freq=save_freq))
        self._sim_step_count += 1
        if self._record_frames:
            self._record_frame()
        return ok and bool(task.plan_success)

    def grasp_actor(self, name: str, arm: str = "auto", pre_grasp_dis: float = 0.09) -> bool:
        task = self._require_task()
        arm_name = self.choose_arm(name) if arm == "auto" else arm
        with self._cwd():
            action = task.grasp_actor(
                self._actor(name),
                arm_tag=self._arm_tag_cls(arm_name),
                pre_grasp_dis=pre_grasp_dis,
            )
        return self._execute(action)

    def place_actor(
        self,
        name: str,
        target_pose: list[float],
        arm: str = "auto",
        functional_point_id: int | None = None,
        pre_dis: float = 0.05,
        dis: float = 0.0,
        pre_dis_axis: str = "fp",
    ) -> bool:
        task = self._require_task()
        arm_name = self.choose_arm(name) if arm == "auto" else arm
        with self._cwd():
            action = task.place_actor(
                self._actor(name),
                arm_tag=self._arm_tag_cls(arm_name),
                target_pose=target_pose,
                functional_point_id=functional_point_id,
                pre_dis=pre_dis,
                dis=dis,
                pre_dis_axis=pre_dis_axis,
            )
        return self._execute(action)

    def move_by_displacement(
        self,
        arm: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> bool:
        task = self._require_task()
        with self._cwd():
            action = task.move_by_displacement(
                arm_tag=self._arm_tag_cls(arm), x=x, y=y, z=z
            )
        return self._execute(action)

    def back_to_origin(self, arm: str) -> bool:
        task = self._require_task()
        with self._cwd():
            action = task.back_to_origin(self._arm_tag_cls(arm))
        return self._execute(action)

    def set_gripper(self, arm: str, *, open_gripper: bool) -> bool:
        task = self._require_task()
        with self._cwd():
            action = (
                task.open_gripper(self._arm_tag_cls(arm))
                if open_gripper
                else task.close_gripper(self._arm_tag_cls(arm))
            )
        return self._execute(action)

    def plan_success(self) -> bool:
        task = self._require_task()
        return bool(task.plan_success)

    def task_completed(self) -> bool:
        task = self._require_task()
        with self._cwd():
            return bool(task.check_success())

    def compute_reward(self) -> float:
        return float(self.plan_success() and self.task_completed())

    def status(self) -> dict[str, Any]:
        return {
            "code_execution_success": None,
            "plan_success": self.plan_success(),
            "task_success": self.task_completed(),
            "sim_steps": self._sim_step_count,
        }

    def trajectory_summary(self) -> dict[str, Any]:
        """Return a small, JSON-safe overview without trajectory arrays."""

        return self._trajectory.summary()

    @property
    def trajectory_recorder(self) -> Any:
        return self._trajectory.recorder

    @property
    def trajectory_renderer(self) -> Any | None:
        return self._trajectory.renderer

    def trajectory_snapshot(self) -> Any:
        """Return an immutable artifact containing all retained trajectory samples."""

        return self._trajectory.snapshot()

    def _render_rgb(self, camera: str = "head_camera") -> np.ndarray:
        task = self._require_task()
        with self._cwd():
            obs = task.get_obs()
        return np.asarray(obs["observation"][camera]["rgb"], dtype=np.uint8).copy()

    def get_observation(self) -> dict[str, Any]:
        rgb = self._render_rgb()
        observation: dict[str, Any] = {
            "head_camera": {"images": {"rgb": rgb}},
            "trajectory": self.trajectory_summary(),
        }
        if self.privileged:
            observation.update(
                {
                    "actors": {
                        name: self.actor_pose(name) for name in self.actor_names()
                    },
                    "status": self.status(),
                }
            )
        return observation

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        return self._render_rgb()

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        del wrist_camera
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled and self._task is not None:
            self._record_frame()

    def _record_frame(self) -> None:
        if self._record_frames and self._task is not None:
            self._frame_buffer.append(self._render_rgb())

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def _close_task(self) -> None:
        self._task_patches.restore_all()
        if self._task is None:
            return
        task, self._task = self._task, None
        try:
            with self._cwd():
                task.close_env(clear_cache=True)
        except Exception:
            # Closing is best-effort because SAPIEN may already have released
            # global resources after a failed setup or planner exception.
            pass

    def close(self) -> None:
        self._close_task()
        self._trajectory.close_renderer()
        server, self.viser_server = self.viser_server, None
        if server is not None:
            stop = getattr(server, "stop", None)
            if callable(stop):
                with suppress(Exception):
                    stop()
        self._trajectory_renderer_attempted = False


__all__ = ["RoboTwinEnv"]

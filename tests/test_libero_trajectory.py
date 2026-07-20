from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import numpy as np
import pytest

from capx.visualization.trajectory import TrajectoryRecorder


def _import_libero_simulator(monkeypatch: Any) -> Any:
    """Import the adapter without installing the real LIBERO/MuJoCo stack."""
    robosuite = types.ModuleType("robosuite")
    robosuite.__path__ = []  # type: ignore[attr-defined]
    robosuite_utils = types.ModuleType("robosuite.utils")
    robosuite_utils.__path__ = []  # type: ignore[attr-defined]
    camera_utils = types.ModuleType("robosuite.utils.camera_utils")
    camera_utils.get_real_depth_map = lambda sim, depth: depth  # type: ignore[attr-defined]

    libero = types.ModuleType("libero")
    libero.__path__ = []  # type: ignore[attr-defined]
    libero.benchmark = types.SimpleNamespace()  # type: ignore[attr-defined]
    libero_envs = types.ModuleType("libero.envs")
    libero_envs.OffScreenRenderEnv = object  # type: ignore[attr-defined]
    libero_utils = types.ModuleType("libero.utils")
    libero_utils.get_libero_path = lambda name: f"/{name}"  # type: ignore[attr-defined]

    depth_utils = types.ModuleType("capx.utils.depth_utils")
    depth_utils.depth_color_to_pointcloud = lambda *args, **kwargs: (  # type: ignore[attr-defined]
        np.zeros((0, 3)),
        np.zeros((0, 3)),
    )
    depth_utils.deproject_pixel_to_camera = lambda *args, **kwargs: np.zeros(3)  # type: ignore[attr-defined]
    depth_utils.depth_to_pointcloud = lambda *args, **kwargs: np.zeros((0, 3))  # type: ignore[attr-defined]
    depth_utils.depth_to_rgb = lambda *args, **kwargs: np.zeros((1, 1, 3))  # type: ignore[attr-defined]

    for name, module in {
        "robosuite": robosuite,
        "robosuite.utils": robosuite_utils,
        "robosuite.utils.camera_utils": camera_utils,
        "libero": libero,
        "libero.envs": libero_envs,
        "libero.utils": libero_utils,
        "capx.utils.depth_utils": depth_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("capx.envs.simulators.libero", None)
    return importlib.import_module("capx.envs.simulators.libero")


def _import_libero_api(monkeypatch: Any) -> Any:
    """Import the large control module with perception/planner dependencies mocked."""
    _import_libero_simulator(monkeypatch)

    open3d = types.ModuleType("open3d")
    scipy = types.ModuleType("scipy")
    scipy.__path__ = []  # type: ignore[attr-defined]
    scipy_spatial = types.ModuleType("scipy.spatial")
    scipy_spatial.__path__ = []  # type: ignore[attr-defined]
    scipy_transform = types.ModuleType("scipy.spatial.transform")
    scipy_transform.Rotation = object  # type: ignore[attr-defined]
    sklearn = types.ModuleType("sklearn")
    sklearn.__path__ = []  # type: ignore[attr-defined]
    sklearn_cluster = types.ModuleType("sklearn.cluster")
    sklearn_cluster.DBSCAN = object  # type: ignore[attr-defined]

    dependency_symbols = {
        "capx.integrations.vision.graspnet": (
            "init_contact_graspnet",
            "init_contact_graspnet_point_clouds",
        ),
        "capx.integrations.vision.molmo": ("init_molmo",),
        "capx.integrations.vision.sam2": ("init_sam2_point_prompt",),
        "capx.integrations.vision.sam3": ("init_sam3", "init_sam3_point_prompt"),
        "capx.integrations.motion.pyroki": ("init_pyroki",),
    }
    for module_name, symbols in dependency_symbols.items():
        dependency = types.ModuleType(module_name)
        for symbol in symbols:
            setattr(dependency, symbol, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, module_name, dependency)

    for name, module in {
        "open3d": open3d,
        "scipy": scipy,
        "scipy.spatial": scipy_spatial,
        "scipy.spatial.transform": scipy_transform,
        "sklearn": sklearn,
        "sklearn.cluster": sklearn_cluster,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("capx.integrations.franka.libero", None)
    return importlib.import_module("capx.integrations.franka.libero")


class _FakeData:
    def __init__(self) -> None:
        self.qpos = np.zeros(7, dtype=np.float64)
        self.xquat = np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.xpos = np.zeros((2, 3), dtype=np.float64)


class _FakeSim:
    def __init__(self) -> None:
        self.data = _FakeData()


class _FakeModel:
    @staticmethod
    def body_name2id(name: str) -> int:
        return 0 if name == "robot0_base" else 1

    @staticmethod
    def get_joint_qpos_addr(name: str) -> int:
        return int(name.removeprefix("robot0_joint")) - 1


class _FakeLowLevelEnv:
    def __init__(self) -> None:
        self.sim = _FakeSim()
        self.sim.model = _FakeModel()
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _FakeHandle:
    def __init__(self) -> None:
        self.env = _FakeLowLevelEnv()
        self.init_states = None
        self.task_language = "move the arm"

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed
        self.env.sim.data.qpos[:] = 0.0
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.env.sim.data.qpos[:] += np.asarray(action[:7]) / 20.0
        self.env.sim.data.xpos[1] = self.env.sim.data.qpos[:3]
        return self._observation(), 0.0, False, {}

    def _observation(self) -> dict[str, Any]:
        return {
            "robot0_joint_pos": self.env.sim.data.qpos.copy(),
            "robot0_gripper_qpos": np.array([0.04], dtype=np.float64),
        }


def _bare_env(module: Any) -> Any:
    env = module.FrankaLiberoEnv.__new__(module.FrankaLiberoEnv)
    env.handle = _FakeHandle()
    env.seed = 3
    env._suite_name = "libero_10"
    env._task_id = 0
    env._control_freq = 20
    env._panda_joint_names = tuple(f"robot0_joint{i}" for i in range(1, 8))
    env._panda_joint_qpos_addrs = list(range(7))
    env.base_link_idx = 0
    env.gripper_link_idx = 1
    env.base_link_wxyz_xyz = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.gripper_link_wxyz_xyz = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.gripper_metric_length = 0.04
    env._gripper_fraction = 1.0
    env._current_joints = np.zeros(7)
    env._current_obs = env.handle._observation()
    env._current_reward = 0.0
    env._current_done = False
    env._current_info = {}
    env._step_count = 0
    env._sim_step_count = 0
    env.max_steps = 100
    env._record_frames = False
    env._subsample_rate = 4
    env._full_viser_rate = 20
    env.viser_debug = False
    env.viser_server = None
    env.trajectory_renderer = None
    env.trajectory_recorder = TrajectoryRecorder(
        metadata=env._trajectory_episode_metadata(seed=env.seed)
    )
    env._trajectory_plan_count = 0
    env._closed = False
    return env


def test_move_records_commanded_action_and_executed_pose(monkeypatch: Any) -> None:
    module = _import_libero_simulator(monkeypatch)
    env = _bare_env(module)
    target = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4])

    env.move_to_joints_blocking(target, max_steps=4)

    artifact = env.trajectory_snapshot()
    commanded = artifact.samples_for("commanded")
    executed = artifact.samples_for("executed")
    assert len(commanded) == len(executed) == 1
    assert commanded[0].arm_state("panda").joint_positions == tuple(target)
    assert len(commanded[0].metadata["controller_action"]) == 8
    actual = executed[0].arm_state("panda")
    assert actual.joint_positions == tuple(target)
    assert actual.ee_pose is not None
    assert executed[0].step == commanded[0].step == 1
    assert env.trajectory_summary()["layers"] == {
        "raw": 0,
        "planned": 0,
        "commanded": 1,
        "executed": 1,
    }


def test_planned_path_is_not_mislabeled_as_executed(monkeypatch: Any) -> None:
    module = _import_libero_simulator(monkeypatch)
    env = _bare_env(module)
    trajectory = np.arange(21, dtype=np.float64).reshape(3, 7) / 100.0

    plan_id = env.record_planned_joint_trajectory(
        trajectory,
        source="curobo_grasp",
        metadata={"object_name": "mug"},
    )

    artifact = env.trajectory_snapshot()
    planned = artifact.samples_for("planned")
    assert plan_id == 0
    assert len(planned) == 3
    assert artifact.samples_for("commanded") == ()
    assert artifact.samples_for("executed") == ()
    assert planned[2].arm_state("panda").joint_positions == tuple(trajectory[2])
    assert planned[0].metadata["source"] == "curobo_grasp"
    assert planned[0].metadata["object_name"] == "mug"
    assert planned[0].feasibility.planner_success is True


def test_planned_path_uses_urdf_fk_and_restores_live_configuration(
    monkeypatch: Any,
) -> None:
    module = _import_libero_simulator(monkeypatch)
    env = _bare_env(module)
    trajectory = np.arange(14, dtype=np.float64).reshape(2, 7) / 100.0

    class FakeUrdf:
        def __init__(self) -> None:
            self.configuration = np.zeros(8)

        def update_cfg(self, configuration: np.ndarray) -> None:
            self.configuration = np.asarray(configuration, dtype=np.float64).copy()

        def get_transform(self, frame_to: str, frame_from: str) -> np.ndarray:
            assert (frame_to, frame_from) == ("panda_hand", "panda_link0")
            transform = np.eye(4)
            transform[:3, 3] = self.configuration[:3]
            return transform

    class FakeViserUrdf:
        def __init__(self, urdf: FakeUrdf) -> None:
            self.urdf = urdf

        def update_cfg(self, configuration: np.ndarray) -> None:
            self.urdf.update_cfg(configuration)

    urdf = FakeUrdf()
    env.urdf = urdf
    env.urdf_vis = FakeViserUrdf(urdf)

    env.record_planned_joint_trajectory(trajectory)

    planned = env.trajectory_snapshot().samples_for("planned")
    assert all(sample.arm_state("panda").ee_pose is not None for sample in planned)
    expected_position = trajectory[1, :3].copy()
    expected_position[2] -= 0.107
    np.testing.assert_allclose(
        planned[1].arm_state("panda").ee_pose.position,
        expected_position,
    )
    np.testing.assert_allclose(urdf.configuration[:7], np.zeros(7))
    assert urdf.configuration[7] == pytest.approx(0.04)


def test_commanded_path_uses_fk_and_reports_cartesian_tracking_error(
    monkeypatch: Any,
) -> None:
    module = _import_libero_simulator(monkeypatch)
    env = _bare_env(module)

    class FakeUrdf:
        def __init__(self) -> None:
            self.configuration = np.zeros(8)

        def update_cfg(self, configuration: np.ndarray) -> None:
            self.configuration = np.asarray(configuration, dtype=np.float64).copy()

        def get_transform(self, frame_to: str, frame_from: str) -> np.ndarray:
            assert (frame_to, frame_from) == ("panda_hand", "panda_link0")
            transform = np.eye(4)
            transform[:3, 3] = self.configuration[:3]
            return transform

    env.urdf = FakeUrdf()
    target = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4])

    env.move_to_joints_blocking(target, max_steps=4)

    artifact = env.trajectory_snapshot()
    commanded = artifact.samples_for("commanded")[0]
    executed = artifact.samples_for("executed")[0]
    assert commanded.arm_state("panda").ee_pose is not None
    assert executed.feasibility.tracking_error_m == pytest.approx(0.0)


def test_reset_clears_trajectory_and_exposes_summary(monkeypatch: Any) -> None:
    module = _import_libero_simulator(monkeypatch)
    env = _bare_env(module)
    env.record_planned_joint_trajectory(np.zeros((2, 7)))
    env._step_once = lambda: None
    env.get_observation = lambda: {"trajectory": env.trajectory_summary()}

    observation, info = env.reset(seed=11)

    assert env.trajectory_summary()["num_samples"] == 0
    assert observation["trajectory"]["num_samples"] == 0
    assert info["trajectory"]["num_samples"] == 0
    assert info["task_prompt"] == "move the arm"
    assert env.trajectory_snapshot().metadata["seed"] == 11


def test_close_releases_renderer_server_and_simulator_once(monkeypatch: Any) -> None:
    module = _import_libero_simulator(monkeypatch)
    env = _bare_env(module)

    class Resource:
        def __init__(self) -> None:
            self.close_count = 0
            self.stop_count = 0

        def close(self) -> None:
            self.close_count += 1

        def stop(self) -> None:
            self.stop_count += 1

    renderer = Resource()
    server = Resource()
    env.trajectory_renderer = renderer
    env.viser_server = server

    env.close()
    env.close()

    assert renderer.close_count == 1
    assert server.stop_count == 1
    assert env.handle.env.close_count == 1


def test_viser_asset_failure_degrades_without_breaking_environment(monkeypatch: Any) -> None:
    module = _import_libero_simulator(monkeypatch)
    handle = _FakeHandle()
    broken_loader = types.ModuleType("robot_descriptions.loaders.yourdfpy")

    def fail_to_load(description: str) -> None:
        raise RuntimeError(f"missing {description}")

    broken_loader.load_robot_description = fail_to_load  # type: ignore[attr-defined]
    fake_viser_extras = types.ModuleType("viser.extras")
    fake_viser_extras.ViserUrdf = object  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "robot_descriptions.loaders.yourdfpy",
        broken_loader,
    )
    monkeypatch.setitem(sys.modules, "viser.extras", fake_viser_extras)
    monkeypatch.setattr(module, "load_libero_task", lambda **kwargs: handle)
    monkeypatch.setattr(module.FrankaLiberoEnv, "reset", lambda self: ({}, {}))

    with pytest.warns(RuntimeWarning, match="Disabling LIBERO Viser debugging"):
        env = module.FrankaLiberoEnv("libero_10", 0, viser_debug=True)

    assert env.viser_debug is False
    assert env.viser_server is None
    assert env.trajectory_renderer is None
    assert env.trajectory_summary()["num_samples"] == 0
    env.close()


def test_curobo_success_records_planned_once_not_during_execution(monkeypatch: Any) -> None:
    module = _import_libero_api(monkeypatch)
    trajectory = np.arange(28, dtype=np.float64).reshape(4, 7) / 100.0

    class FakeCurobo:
        @staticmethod
        def plan_to_grasp_poses(*args: Any, **kwargs: Any) -> tuple[bool, np.ndarray, int]:
            return True, trajectory, 2

        @staticmethod
        def plan_with_grasped_object(*args: Any, **kwargs: Any) -> tuple[bool, np.ndarray]:
            return True, trajectory[::-1]

    class FakeEnv:
        def __init__(self) -> None:
            self.plans: list[tuple[np.ndarray, str, dict[str, Any]]] = []
            self.moves: list[np.ndarray] = []

        def get_observation(self) -> dict[str, np.ndarray]:
            return {"robot_joint_pos": np.zeros(7)}

        def record_planned_joint_trajectory(
            self,
            path: np.ndarray,
            *,
            source: str,
            metadata: dict[str, Any],
        ) -> None:
            self.plans.append((np.asarray(path), source, metadata))

        def move_to_joints_blocking(self, joints: np.ndarray, **kwargs: Any) -> None:
            self.moves.append(np.asarray(joints))

    fake_env = FakeEnv()
    api = module.FrankaLiberoApi.__new__(module.FrankaLiberoApi)
    api._env = fake_env
    api._curobo_world_config = object()
    monkeypatch.setattr(module, "_curobo_api", FakeCurobo())

    success, path, goalset_index = api.plan_grasp_trajectory(
        "mug",
        object_mask=np.ones((1, 1), dtype=bool),
        grasp_poses=[(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))],
        world_config=object(),
    )
    api.execute_joint_trajectory(path, subsample=2)
    grasped_success, grasped_path = api.plan_with_grasped_object(
        (np.ones(3), np.array([1.0, 0.0, 0.0, 0.0])),
        "mug",
    )

    assert success is True
    assert goalset_index == 2
    assert grasped_success is True
    np.testing.assert_array_equal(grasped_path, trajectory[::-1])
    assert len(fake_env.plans) == 2
    recorded_path, source, metadata = fake_env.plans[0]
    np.testing.assert_array_equal(recorded_path, trajectory)
    assert source == "curobo_grasp"
    assert metadata == {"object_name": "mug", "goalset_index": 2}
    _, source, metadata = fake_env.plans[1]
    assert source == "curobo_grasped_object"
    assert metadata == {"object_name": "mug"}
    assert len(fake_env.moves) == 3  # waypoints 0, 2, and the final waypoint

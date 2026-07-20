"""Non-invasive trajectory instrumentation for native RoboTwin tasks.

RoboTwin owns its control loop: a high-level ``task.move`` call produces dense
joint plans, writes articulation drive targets, and finally advances SAPIEN via
``scene.step``.  This module instruments those three boundaries on a task
instance without modifying the external RoboTwin checkout.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from capx.visualization.trajectory import (
    ArmState,
    EndEffectorPose,
    FeasibilityMetrics,
    TrajectoryArtifact,
    TrajectoryRecorder,
)

_MISSING = object()


@dataclass(slots=True)
class _PatchedAttribute:
    owner: Any
    name: str
    had_instance_value: bool
    instance_value: Any
    resolved_value: Any


class RestorableAttributePatches:
    """Install instance-level hooks and restore the original descriptor state."""

    def __init__(self) -> None:
        self._patches: list[_PatchedAttribute] = []
        self._keys: set[tuple[int, str]] = set()

    def patch(self, owner: Any, name: str, value: Any) -> bool:
        key = (id(owner), name)
        if key in self._keys:
            return False
        try:
            resolved_value = getattr(owner, name, _MISSING)
            instance_dict = getattr(owner, "__dict__", {})
            had_instance_value = name in instance_dict
            instance_value = instance_dict.get(name, _MISSING)
            setattr(owner, name, value)
        except Exception:
            return False
        self._patches.append(
            _PatchedAttribute(
                owner=owner,
                name=name,
                had_instance_value=had_instance_value,
                instance_value=instance_value,
                resolved_value=resolved_value,
            )
        )
        self._keys.add(key)
        return True

    def restore_all(self) -> None:
        for patch in reversed(self._patches):
            try:
                if patch.had_instance_value:
                    setattr(patch.owner, patch.name, patch.instance_value)
                else:
                    delattr(patch.owner, patch.name)
            except Exception:
                # Some extension types do not allow deleting an instance
                # attribute.  Restoring the previously resolved callable is a
                # safe fallback even though it leaves an instance override.
                if patch.resolved_value is not _MISSING:
                    with suppress(Exception):
                        setattr(patch.owner, patch.name, patch.resolved_value)
        self._patches.clear()
        self._keys.clear()

    def __len__(self) -> int:
        return len(self._patches)


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    """Best-effort conversion of RoboTwin action objects into JSON data."""

    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), depth=depth + 1)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), depth=depth + 1)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item, depth=depth + 1) for item in value]

    action_fields = (
        "arm_tag",
        "action",
        "target_pose",
        "target_gripper_pos",
        "args",
    )
    if any(hasattr(value, field) for field in action_fields):
        return {
            field: _jsonable(getattr(value, field), depth=depth + 1)
            for field in action_fields
            if hasattr(value, field)
        }
    if hasattr(value, "p") and hasattr(value, "q"):
        return {
            "position": _jsonable(value.p, depth=depth + 1),
            "wxyz": _jsonable(value.q, depth=depth + 1),
        }
    return repr(value)


def _arm_name(value: Any) -> str | None:
    if value is None:
        return None
    candidate = getattr(value, "arm", value)
    text = str(candidate).strip().lower()
    return text if text in {"left", "right"} else None


def _as_finite_vector(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    return vector


def _as_pose(value: Any) -> EndEffectorPose | None:
    if value is None:
        return None
    if hasattr(value, "p") and hasattr(value, "q"):
        try:
            values = np.concatenate(
                [np.asarray(value.p, dtype=float), np.asarray(value.q, dtype=float)]
            ).reshape(-1)
        except Exception:
            return None
    else:
        vector = _as_finite_vector(value)
        if vector is None:
            return None
        values = vector
    if values.size < 7:
        return None
    try:
        return EndEffectorPose(
            position=tuple(float(item) for item in values[:3]),
            wxyz=tuple(float(item) for item in values[3:7]),
        )
    except (TypeError, ValueError):
        return None


def _mapping_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _rotation_matrix_from_wxyz(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _wxyz_from_rotation_matrix(value: Any) -> tuple[float, float, float, float]:
    matrix = np.asarray(value, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(item) for item in quaternion)


@dataclass(frozen=True, slots=True)
class _ArmKinematicsSpec:
    joint_names: tuple[str, ...]
    ee_frame: str
    root_position: tuple[float, float, float]
    root_wxyz: tuple[float, float, float, float]
    gripper_bias: float
    global_rotation: np.ndarray
    delta_rotation: np.ndarray
    urdf_path: Path


class RoboTwinForwardKinematics:
    """URDF FK matching RoboTwin's transformed EE/TCP convention.

    ALOHA is represented by one shared dual-arm articulation.  The helper
    therefore reuses one YourDFPy model for both arm branches and applies joint
    updates by name, preserving the other branch's state.
    """

    def __init__(self, task: Any) -> None:
        robot = getattr(task, "robot", None)
        if robot is None:
            raise ValueError("task.robot is unavailable")
        self._lock = threading.RLock()
        self._specs: dict[str, _ArmKinematicsSpec] = {}
        self._models: dict[Path, Any] = {}
        self._model_by_arm: dict[str, Any] = {}
        for arm in ("left", "right"):
            path_value = getattr(robot, f"{arm}_urdf_path", None)
            names_value = getattr(robot, f"{arm}_arm_joints_name", None)
            ee_name = getattr(robot, f"{arm}_ee_name", None)
            root_pose = getattr(robot, f"{arm}_entity_origion_pose", None)
            if path_value is None or names_value is None or ee_name is None or root_pose is None:
                continue
            urdf_path = Path(path_value).expanduser().resolve()
            if not urdf_path.is_file():
                continue
            root_position = tuple(float(item) for item in np.asarray(root_pose.p).reshape(3))
            root_wxyz = tuple(float(item) for item in np.asarray(root_pose.q).reshape(4))
            self._specs[arm] = _ArmKinematicsSpec(
                joint_names=tuple(str(name) for name in names_value),
                ee_frame=str(ee_name),
                root_position=root_position,
                root_wxyz=root_wxyz,
                gripper_bias=float(getattr(robot, f"{arm}_gripper_bias", 0.12)),
                global_rotation=np.asarray(
                    getattr(robot, f"{arm}_global_trans_matrix", np.eye(3)),
                    dtype=float,
                ).reshape(3, 3),
                delta_rotation=np.asarray(
                    getattr(robot, f"{arm}_delta_matrix", np.eye(3)),
                    dtype=float,
                ).reshape(3, 3),
                urdf_path=urdf_path,
            )

        if not self._specs:
            raise ValueError("RoboTwin robot does not expose usable URDF kinematics metadata")
        try:
            from yourdfpy import URDF
        except ImportError as exc:
            raise RuntimeError("YourDFPy is required for RoboTwin planned-path FK") from exc

        for arm, spec in self._specs.items():
            model = self._models.get(spec.urdf_path)
            if model is None:
                model = URDF.load(
                    str(spec.urdf_path),
                    build_scene_graph=True,
                    build_collision_scene_graph=False,
                    load_meshes=False,
                    load_collision_meshes=False,
                )
                self._models[spec.urdf_path] = model
            self._model_by_arm[arm] = model

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(self._specs)

    @property
    def shared_dual_urdf_path(self) -> Path | None:
        if set(self._specs) != {"left", "right"}:
            return None
        left_path = self._specs["left"].urdf_path
        return left_path if self._specs["right"].urdf_path == left_path else None

    def root_pose(self, arm: str = "left") -> tuple[tuple[float, ...], tuple[float, ...]]:
        spec = self._specs[arm]
        return spec.root_position, spec.root_wxyz

    def pose(self, arm: str, joint_positions: Sequence[float]) -> EndEffectorPose | None:
        spec = self._specs.get(arm)
        model = self._model_by_arm.get(arm)
        if spec is None or model is None:
            return None
        qpos = np.asarray(joint_positions, dtype=float).reshape(-1)
        if len(qpos) != len(spec.joint_names) or not np.all(np.isfinite(qpos)):
            return None
        with self._lock:
            model.update_cfg(
                {
                    name: float(value)
                    for name, value in zip(spec.joint_names, qpos, strict=True)
                }
            )
            joint = model.joint_map.get(spec.ee_frame)
            ee_link = joint.child if joint is not None else spec.ee_frame
            base_to_ee = np.asarray(model.get_transform(ee_link), dtype=float)

        world_from_base = np.eye(4, dtype=float)
        world_from_base[:3, :3] = _rotation_matrix_from_wxyz(spec.root_wxyz)
        world_from_base[:3, 3] = spec.root_position
        world_from_ee = world_from_base @ base_to_ee
        # YourDFPy returns the URDF child-link frame.  RoboTwin applies
        # ``global_trans_matrix`` because SAPIEN ``Joint.global_pose`` is in the
        # joint frame; applying it again here would double the fixed joint-to-
        # child transform (Rx(pi) for ALOHA's joint6).  ``delta_matrix`` is the
        # remaining embodiment-specific tool-frame adjustment.
        ee_rotation = world_from_ee[:3, :3] @ spec.delta_rotation
        ee_position = world_from_ee[:3, 3] + ee_rotation @ np.asarray(
            [spec.gripper_bias - 0.12, 0.0, 0.0]
        )
        return EndEffectorPose(
            position=tuple(float(item) for item in ee_position),
            wxyz=_wxyz_from_rotation_matrix(ee_rotation),
        )


class RoboTwinTrajectoryInstrumentation:
    """Record native RoboTwin control layers while keeping hooks recoverable."""

    def __init__(
        self,
        *,
        metadata: Mapping[str, Any],
        max_samples_per_layer: int = 2_000,
    ) -> None:
        self.recorder = TrajectoryRecorder(
            max_samples_per_layer=max_samples_per_layer,
            metadata=metadata,
        )
        self.renderer: Any | None = None
        self.kinematics: RoboTwinForwardKinematics | None = None
        self._task: Any | None = None
        self._joint_names: dict[str, tuple[str, ...]] = {"left": (), "right": ()}
        self._high_level_step = 0
        self._planned_step = 0
        self._physics_step = 0
        self._warnings: set[str] = set()
        self._viser_urdf: Any | None = None
        self._viser_robot_base: Any | None = None
        self._viser_joint_names: tuple[str, ...] = ()
        self._viser_configuration: dict[str, float] = {}
        self._viser_gripper_joints: dict[str, str] = {}
        self._viser_gripper_scales: dict[str, tuple[float, float]] = {}
        self._viser_actor_handles: dict[str, Any] = {}

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(sorted(self._warnings))

    @property
    def physics_step(self) -> int:
        return self._physics_step

    def reset(self, task: Any, *, metadata: Mapping[str, Any]) -> None:
        self._warnings.clear()
        self._task = task
        self._joint_names = self._discover_joint_names(getattr(task, "robot", None))
        try:
            self.kinematics = RoboTwinForwardKinematics(task)
        except Exception as exc:
            self.kinematics = None
            self._warn("planned-path FK", exc)
        enriched_metadata = dict(metadata)
        enriched_metadata["arms"] = {
            arm: {"joint_names": list(names)}
            for arm, names in self._joint_names.items()
            if names
        }
        robot = getattr(task, "robot", None)
        if robot is not None:
            enriched_metadata["robot"] = {
                "class": type(robot).__name__,
                "dual_arm": bool(getattr(robot, "is_dual_arm", True)),
                "planned_fk": self.kinematics is not None,
                "shared_dual_urdf": bool(
                    self.kinematics is not None
                    and self.kinematics.shared_dual_urdf_path is not None
                ),
            }
        self.recorder.reset(metadata=enriched_metadata)
        self._high_level_step = 0
        self._planned_step = 0
        self._physics_step = 0

    def connect_renderer(self, server: Any, *, timeline_max: int | None = None) -> bool:
        if server is None:
            return False
        self.close_renderer()
        robot_state_setters: dict[str, Any] = {}
        try:
            robot_state_setters = self._setup_viser_robot(server)
        except Exception as exc:
            self._warn("RoboTwin URDF visualization", exc)
            self._close_viser_robot()
        try:
            from capx.visualization import ViserTrajectoryRenderer

            self.renderer = ViserTrajectoryRenderer(
                server,
                source=self.recorder,
                root="/trajectory/robotwin",
                robot_state_setters=robot_state_setters,
                auto_subscribe=False,
                timeline_max=timeline_max,
            )
        except Exception as exc:
            self.renderer = None
            self._warn("renderer", exc)
            self._close_viser_robot()
            return False
        return True

    def close_renderer(self) -> None:
        renderer, self.renderer = self.renderer, None
        if renderer is None:
            self._close_viser_robot()
            return
        with suppress(Exception):
            renderer.close()
        self._close_viser_robot()

    def _setup_viser_robot(self, server: Any) -> dict[str, Any]:
        self._setup_viser_actor_markers(server)
        kinematics = self.kinematics
        path = None if kinematics is None else kinematics.shared_dual_urdf_path
        if path is None:
            return {}
        from viser.extras import ViserUrdf

        robot = getattr(self._task, "robot", None)
        position, wxyz = kinematics.root_pose("left")
        self._viser_robot_base = server.scene.add_frame(
            "/robotwin/embodiment",
            position=position,
            wxyz=wxyz,
            show_axes=False,
        )
        self._viser_urdf = ViserUrdf(
            server,
            path,
            root_node_name="/robotwin/embodiment",
            load_meshes=True,
        )
        self._viser_joint_names = tuple(self._viser_urdf.get_actuated_joint_names())
        self._viser_configuration = dict.fromkeys(self._viser_joint_names, 0.0)
        self._viser_gripper_joints = {}
        self._viser_gripper_scales = {}
        for arm in ("left", "right"):
            gripper_spec = getattr(robot, f"{arm}_gripper_name", None)
            base_joint = (
                gripper_spec.get("base") if isinstance(gripper_spec, Mapping) else None
            )
            if base_joint is not None:
                self._viser_gripper_joints[arm] = str(base_joint)
            scale = getattr(robot, f"{arm}_gripper_scale", None)
            vector = _as_finite_vector(scale)
            if vector is not None and len(vector) >= 2:
                self._viser_gripper_scales[arm] = (
                    float(vector[0]),
                    float(vector[1]),
                )
        return dict.fromkeys(kinematics.arms, self._set_viser_robot_state)

    def _setup_viser_actor_markers(self, server: Any) -> None:
        task = self._task
        if task is None:
            return
        for name, actor in vars(task).items():
            if name.startswith("_") or not callable(getattr(actor, "get_pose", None)):
                continue
            try:
                pose = actor.get_pose()
                position = tuple(float(item) for item in np.asarray(pose.p).reshape(3))
                wxyz = tuple(float(item) for item in np.asarray(pose.q).reshape(4))
                component = name.replace("/", "_").replace(" ", "_")
                self._viser_actor_handles[name] = server.scene.add_frame(
                    f"/robotwin/actors/{component}",
                    position=position,
                    wxyz=wxyz,
                    axes_length=0.04,
                    axes_radius=0.002,
                )
            except Exception as exc:
                self._warn(f"actor marker {name}", exc)

    def _update_viser_actor_markers(self) -> None:
        task = self._task
        if task is None:
            return
        for name, handle in self._viser_actor_handles.items():
            actor = getattr(task, name, None)
            get_pose = getattr(actor, "get_pose", None)
            if not callable(get_pose):
                continue
            try:
                pose = get_pose()
                handle.position = tuple(
                    float(item) for item in np.asarray(pose.p).reshape(3)
                )
                handle.wxyz = tuple(float(item) for item in np.asarray(pose.q).reshape(4))
            except Exception as exc:
                self._warn(f"actor marker update {name}", exc)

    def _set_viser_robot_state(self, state: ArmState) -> None:
        urdf = self._viser_urdf
        if urdf is None:
            return
        for name, position in zip(
            state.joint_names, state.joint_positions, strict=True
        ):
            if name in self._viser_configuration:
                self._viser_configuration[name] = float(position)
        gripper_joint = self._viser_gripper_joints.get(state.arm)
        if state.gripper is not None and gripper_joint in self._viser_configuration:
            low, high = self._viser_gripper_scales.get(state.arm, (0.0, 1.0))
            self._viser_configuration[gripper_joint] = low + float(state.gripper) * (
                high - low
            )
        configuration = np.asarray(
            [self._viser_configuration[name] for name in self._viser_joint_names],
            dtype=float,
        )
        urdf.update_cfg(configuration)

    def _close_viser_robot(self) -> None:
        urdf, self._viser_urdf = self._viser_urdf, None
        if urdf is not None:
            with suppress(Exception):
                urdf.remove()
        base, self._viser_robot_base = self._viser_robot_base, None
        if base is not None:
            with suppress(Exception):
                base.remove()
        self._viser_joint_names = ()
        self._viser_configuration.clear()
        self._viser_gripper_joints.clear()
        self._viser_gripper_scales.clear()
        for handle in self._viser_actor_handles.values():
            with suppress(Exception):
                handle.remove()
        self._viser_actor_handles.clear()

    def install(self, task: Any, patches: RestorableAttributePatches) -> None:
        move = getattr(task, "move", None)
        if callable(move):

            @functools.wraps(move)
            def wrapped_move(*args: Any, **kwargs: Any) -> Any:
                self._record_high_level_move(args, kwargs)
                result = move(*args, **kwargs)
                self._update_renderer()
                return result

            if not patches.patch(task, "move", wrapped_move):
                self._warnings.add("could not instrument task.move")
        else:
            self._warnings.add("task.move is unavailable")

        take_dense_action = getattr(task, "take_dense_action", None)
        if callable(take_dense_action):

            @functools.wraps(take_dense_action)
            def wrapped_dense_action(*args: Any, **kwargs: Any) -> Any:
                control_seq = kwargs.get("control_seq", args[0] if args else None)
                self._record_dense_plan(control_seq)
                self._update_renderer()
                result = take_dense_action(*args, **kwargs)
                self._update_renderer()
                return result

            if not patches.patch(task, "take_dense_action", wrapped_dense_action):
                self._warnings.add("could not instrument task.take_dense_action")
        else:
            self._warnings.add("task.take_dense_action is unavailable")

        scene = getattr(task, "scene", None)
        scene_step = getattr(scene, "step", None)
        if callable(scene_step):

            @functools.wraps(scene_step)
            def wrapped_scene_step(*args: Any, **kwargs: Any) -> Any:
                result = scene_step(*args, **kwargs)
                self._physics_step += 1
                self._record_drive_and_measured_state()
                return result

            if not patches.patch(scene, "step", wrapped_scene_step):
                self._warnings.add("could not instrument task.scene.step")
        else:
            self._warnings.add("task.scene.step is unavailable")

    def summary(self) -> dict[str, Any]:
        return self.recorder.summary()

    def snapshot(self) -> TrajectoryArtifact:
        return self.recorder.snapshot()

    def record_current_state(self) -> None:
        """Capture the current drive targets and measured state without stepping."""

        self._record_drive_and_measured_state()
        self._update_renderer()

    def _update_renderer(self) -> None:
        renderer = self.renderer
        if renderer is None:
            return
        try:
            renderer.update()
        except Exception as exc:
            self._warn("renderer update", exc)
            self.close_renderer()

    def _record_high_level_move(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> None:
        try:
            groups = self._move_action_groups(args, kwargs)
            max_actions = max((len(actions) for _, actions in groups), default=0)
            if max_actions == 0:
                self.recorder.append_raw(
                    step=self._high_level_step,
                    payload={"args": _jsonable(args), "kwargs": _jsonable(kwargs)},
                    metadata={"source": "task.move"},
                )
                self._high_level_step += 1
                return

            for action_index in range(max_actions):
                arm_states: dict[str, ArmState] = {}
                raw_actions: list[dict[str, Any]] = []
                for default_arm, actions in groups:
                    if action_index >= len(actions):
                        continue
                    action = actions[action_index]
                    arm = _arm_name(_mapping_value(action, "arm_tag", default_arm))
                    raw_actions.append(
                        {
                            "arm": arm,
                            "action": _jsonable(action),
                        }
                    )
                    if arm is not None:
                        state = self._high_level_arm_state(action)
                        if state is not None:
                            arm_states[arm] = state

                sample_step = self._high_level_step
                payload = {
                    "actions": raw_actions,
                    "save_freq": _jsonable(kwargs.get("save_freq")),
                }
                feasibility = self._planner_feasibility()
                self.recorder.append_raw(
                    step=sample_step,
                    arms=arm_states,
                    payload=payload,
                    feasibility=feasibility,
                    metadata={
                        "source": "task.move",
                        "action_index": action_index,
                    },
                )
                self.recorder.append_commanded(
                    step=sample_step,
                    arms=arm_states,
                    payload={"normalized_from": "task.move"},
                    feasibility=feasibility,
                    metadata={
                        "source": "high_level_target",
                        "action_index": action_index,
                    },
                )
                self._high_level_step += 1
        except Exception as exc:
            self._warn("task.move recording", exc)

    @staticmethod
    def _move_action_groups(
        args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> list[tuple[str | None, list[Any]]]:
        first = kwargs.get("actions_by_arm1", args[0] if args else None)
        second = kwargs.get("actions_by_arm2", args[1] if len(args) > 1 else None)
        groups: list[tuple[str | None, list[Any]]] = []
        for value in (first, second):
            if value is None:
                continue
            if isinstance(value, (tuple, list)) and len(value) == 2:
                default_arm = _arm_name(value[0])
                raw_actions = value[1]
                if raw_actions is None:
                    actions: list[Any] = []
                elif isinstance(raw_actions, (tuple, list)):
                    actions = list(raw_actions)
                else:
                    actions = [raw_actions]
            else:
                default_arm = _arm_name(_mapping_value(value, "arm_tag"))
                actions = [value]
            groups.append((default_arm, actions))
        return groups

    @staticmethod
    def _high_level_arm_state(action: Any) -> ArmState | None:
        action_kind = str(_mapping_value(action, "action", "")).lower()
        pose = _as_pose(_mapping_value(action, "target_pose"))
        gripper_value = _mapping_value(action, "target_gripper_pos")
        gripper: float | None = None
        if gripper_value is not None:
            try:
                gripper = float(gripper_value)
            except (TypeError, ValueError):
                gripper = None
        if pose is None and gripper is None:
            return None
        return ArmState(
            ee_pose=pose,
            gripper=gripper,
            metadata={"action": action_kind or "unknown", "frame": "world"},
        )

    def _record_dense_plan(self, control_seq: Any) -> None:
        if not isinstance(control_seq, Mapping):
            self._warnings.add("take_dense_action control_seq is not a mapping")
            return
        try:
            arm_controls: dict[str, Mapping[str, Any]] = {}
            gripper_controls: dict[str, Mapping[str, Any]] = {}
            max_length = 0
            for arm in ("left", "right"):
                arm_value = control_seq.get(f"{arm}_arm")
                if isinstance(arm_value, Mapping):
                    arm_controls[arm] = arm_value
                    positions = self._matrix(arm_value.get("position"))
                    if positions is not None:
                        max_length = max(max_length, len(positions))
                gripper_value = control_seq.get(f"{arm}_gripper")
                if isinstance(gripper_value, Mapping):
                    gripper_controls[arm] = gripper_value
                    max_length = max(
                        max_length,
                        int(gripper_value.get("num_step", 0) or 0),
                    )

            for control_index in range(max_length):
                states: dict[str, ArmState] = {}
                velocity_payload: dict[str, list[float]] = {}
                for arm in ("left", "right"):
                    positions = self._matrix(arm_controls.get(arm, {}).get("position"))
                    velocities = self._matrix(arm_controls.get(arm, {}).get("velocity"))
                    qpos: tuple[float, ...] = ()
                    names: tuple[str, ...] = ()
                    if positions is not None and control_index < len(positions):
                        row = positions[control_index]
                        names = self._names_for_width(arm, len(row))
                        qpos = tuple(float(item) for item in row)
                    if velocities is not None and control_index < len(velocities):
                        velocity_payload[arm] = [
                            float(item) for item in velocities[control_index]
                        ]
                    gripper = self._gripper_plan_value(
                        gripper_controls.get(arm), control_index
                    )
                    if qpos or gripper is not None:
                        ee_pose = self._forward_kinematics(arm, qpos)
                        states[arm] = ArmState(
                            joint_names=names,
                            joint_positions=qpos,
                            ee_pose=ee_pose,
                            gripper=gripper,
                        )
                if states:
                    self.recorder.append_planned(
                        step=self._planned_step,
                        arms=states,
                        feasibility=self._planner_feasibility(),
                        metadata={
                            "source": "take_dense_action",
                            "control_index": control_index,
                            "joint_velocities": velocity_payload,
                        },
                    )
                    self._planned_step += 1
        except Exception as exc:
            self._warn("dense plan recording", exc)

    @staticmethod
    def _matrix(value: Any) -> np.ndarray | None:
        try:
            matrix = np.asarray(value, dtype=float)
        except Exception:
            return None
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
            return None
        return matrix

    @staticmethod
    def _gripper_plan_value(control: Mapping[str, Any] | None, index: int) -> float | None:
        if control is None:
            return None
        result = _as_finite_vector(control.get("result"))
        if result is None or index >= len(result):
            return None
        return float(result[index])

    def _record_drive_and_measured_state(self) -> None:
        try:
            drive_states = self._collect_robot_states(real=False)
            if drive_states:
                self.recorder.append_commanded(
                    step=self._physics_step,
                    arms=drive_states,
                    feasibility=self._planner_feasibility(),
                    metadata={"source": "drive_target"},
                )
            measured_states = self._collect_robot_states(real=True)
            if measured_states:
                self.recorder.append_executed(
                    step=self._physics_step,
                    arms=measured_states,
                    feasibility=self._planner_feasibility(include_task=True),
                    metadata={"source": "real_qpos", "frame": "world"},
                )
            self._update_viser_actor_markers()
        except Exception as exc:
            self._warn("scene.step recording", exc)

    def _collect_robot_states(self, *, real: bool) -> dict[str, ArmState]:
        task = self._task
        robot = getattr(task, "robot", None)
        if robot is None:
            return {}
        states: dict[str, ArmState] = {}
        for arm in ("left", "right"):
            method_name = (
                f"get_{arm}_arm_real_jointState"
                if real
                else f"get_{arm}_arm_jointState"
            )
            state_method = getattr(robot, method_name, None)
            values: np.ndarray | None = None
            if callable(state_method):
                try:
                    values = _as_finite_vector(state_method())
                except Exception as exc:
                    self._warn(method_name, exc)

            names = self._joint_names[arm]
            if values is not None and not names:
                inferred_width = max(0, len(values) - 1)
                names = self._names_for_width(arm, inferred_width)
            qpos: tuple[float, ...] = ()
            if values is not None and names:
                width = min(len(values), len(names))
                names = names[:width]
                qpos = tuple(float(item) for item in values[:width])

            gripper: float | None = None
            gripper_method = getattr(robot, f"get_{arm}_gripper_val", None)
            if callable(gripper_method):
                try:
                    gripper = float(gripper_method())
                except Exception as exc:
                    self._warn(f"get_{arm}_gripper_val", exc)
            elif values is not None and len(values) > len(qpos):
                gripper = float(values[len(qpos)])

            ee_pose: EndEffectorPose | None = None
            if real:
                ee_method = getattr(robot, f"get_{arm}_ee_pose", None)
                if callable(ee_method):
                    try:
                        ee_pose = _as_pose(ee_method())
                    except Exception as exc:
                        self._warn(f"get_{arm}_ee_pose", exc)
            elif qpos:
                ee_pose = self._forward_kinematics(arm, qpos)

            if qpos or gripper is not None or ee_pose is not None:
                states[arm] = ArmState(
                    joint_names=names,
                    joint_positions=qpos,
                    ee_pose=ee_pose,
                    gripper=gripper,
                    metadata={
                        "state_source": "real_qpos" if real else "drive_target"
                    },
                )
        return states

    def _forward_kinematics(
        self, arm: str, joint_positions: Sequence[float]
    ) -> EndEffectorPose | None:
        if not joint_positions or self.kinematics is None:
            return None
        try:
            return self.kinematics.pose(arm, joint_positions)
        except Exception as exc:
            self._warn(f"{arm} planned-path FK", exc)
            return None

    @staticmethod
    def _discover_joint_names(robot: Any) -> dict[str, tuple[str, ...]]:
        discovered: dict[str, tuple[str, ...]] = {"left": (), "right": ()}
        if robot is None:
            return discovered
        for arm in ("left", "right"):
            configured = getattr(robot, f"{arm}_arm_joints_name", None)
            if configured is not None:
                try:
                    names = tuple(str(name) for name in configured)
                except TypeError:
                    names = ()
                if names:
                    discovered[arm] = names
                    continue
            joints = getattr(robot, f"{arm}_arm_joints", None)
            if joints is None:
                continue
            names_list: list[str] = []
            try:
                for index, joint in enumerate(joints):
                    get_name = getattr(joint, "get_name", None)
                    name = get_name() if callable(get_name) else getattr(joint, "name", None)
                    names_list.append(str(name or f"{arm}_joint_{index}"))
            except Exception:
                names_list = []
            discovered[arm] = tuple(names_list)
        return discovered

    def _names_for_width(self, arm: str, width: int) -> tuple[str, ...]:
        names = self._joint_names[arm]
        if len(names) >= width:
            return names[:width]
        return names + tuple(f"{arm}_joint_{index}" for index in range(len(names), width))

    def _planner_feasibility(self, *, include_task: bool = False) -> FeasibilityMetrics:
        task = self._task
        planner_success: bool | None = None
        if task is not None and hasattr(task, "plan_success"):
            with suppress(Exception):
                planner_success = bool(task.plan_success)
        task_success: bool | None = None
        if include_task:
            check_success = getattr(task, "check_success", None)
            if callable(check_success):
                with suppress(Exception):
                    task_success = bool(check_success())
        return FeasibilityMetrics(
            planner_success=planner_success,
            task_success=task_success,
        )

    def _warn(self, context: str, exc: Exception) -> None:
        self._warnings.add(f"{context}: {type(exc).__name__}: {exc}")


__all__ = [
    "RestorableAttributePatches",
    "RoboTwinForwardKinematics",
    "RoboTwinTrajectoryInstrumentation",
]

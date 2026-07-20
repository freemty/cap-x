"""Thread-safe, simulator-agnostic trajectory recording.

The data model in this module intentionally has no dependency on Viser or a
physics simulator.  Simulator adapters record what they know (joint state,
end-effector pose, or just a raw high-level action) and visualization/export
code consumes an immutable snapshot.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

TRAJECTORY_SCHEMA_VERSION = 1


class TrajectoryLayer(str, Enum):  # noqa: UP042 - package supports Python 3.10
    """Semantic stages of a policy action on its way to the robot."""

    RAW = "raw"
    PLANNED = "planned"
    COMMANDED = "commanded"
    EXECUTED = "executed"


TRAJECTORY_LAYERS: tuple[TrajectoryLayer, ...] = tuple(TrajectoryLayer)


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _float_tuple(values: Sequence[Any], length: int, name: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, name) for value in values)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(result)}")
    return result


def _json_safe(value: Any, path: str = "value") -> Any:
    """Copy a value into the JSON data model, including common NumPy inputs."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_float(value, path)
    if isinstance(value, Enum):
        return _json_safe(value.value, path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {type(key).__name__}")
            result[key] = _json_safe(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[]") for item in value]
    # NumPy arrays and scalar values expose tolist()/item() without requiring
    # NumPy to be imported by this core module.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist(), path)
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item(), path)
    raise TypeError(f"{path} is not JSON serializable: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class EndEffectorPose:
    """A framed end-effector pose using a scalar-first quaternion."""

    position: tuple[float, float, float]
    wxyz: tuple[float, float, float, float]
    frame_id: str = "world"

    def __post_init__(self) -> None:
        position = _float_tuple(self.position, 3, "position")
        wxyz = _float_tuple(self.wxyz, 4, "wxyz")
        norm = math.sqrt(sum(component * component for component in wxyz))
        if norm <= 1e-12:
            raise ValueError("wxyz quaternion must have non-zero norm")
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "wxyz", tuple(component / norm for component in wxyz))
        object.__setattr__(self, "frame_id", self.frame_id.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "wxyz": list(self.wxyz),
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        default_frame_id: str = "world",
    ) -> EndEffectorPose:
        return cls(
            position=tuple(data["position"]),
            wxyz=tuple(data["wxyz"]),
            frame_id=data.get("frame_id", default_frame_id),
        )


@dataclass(frozen=True, slots=True)
class ArmState:
    """Joint and/or end-effector state for one named arm.

    ``arm`` may be omitted when the state is supplied through an ``arms``
    mapping; the recorder fills it from the mapping key.  Empty joint fields
    are valid for EEF-only and high-level actions.
    """

    joint_names: tuple[str, ...] = ()
    joint_positions: tuple[float, ...] = ()
    ee_pose: EndEffectorPose | None = None
    gripper: float | None = None
    arm: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        joint_names = tuple(str(name) for name in self.joint_names)
        joint_positions = tuple(
            _finite_float(value, "joint_positions") for value in self.joint_positions
        )
        if len(joint_names) != len(joint_positions):
            raise ValueError(
                "joint_names and joint_positions must have equal length "
                f"({len(joint_names)} != {len(joint_positions)})"
            )
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names must be unique within an arm")
        arm = str(self.arm)
        if arm and not arm.strip():
            raise ValueError("arm must not be whitespace")
        gripper = None if self.gripper is None else _finite_float(self.gripper, "gripper")
        metadata = _json_safe(self.metadata, "arm.metadata")
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "joint_positions", joint_positions)
        object.__setattr__(self, "gripper", gripper)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "joint_names": list(self.joint_names),
            "joint_positions": list(self.joint_positions),
            "ee_pose": None if self.ee_pose is None else self.ee_pose.to_dict(),
            "gripper": self.gripper,
            "metadata": _json_safe(self.metadata, "arm.metadata"),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        default_frame_id: str = "world",
    ) -> ArmState:
        pose_data = data.get("ee_pose")
        return cls(
            arm=str(data.get("arm", "")),
            joint_names=tuple(data.get("joint_names", ())),
            joint_positions=tuple(data.get("joint_positions", ())),
            ee_pose=(
                None
                if pose_data is None
                else EndEffectorPose.from_dict(pose_data, default_frame_id=default_frame_id)
            ),
            gripper=data.get("gripper"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class FeasibilityMetrics:
    """Planner/simulator checks attached to a trajectory sample.

    ``None`` means that a check was not evaluated; it must not be interpreted
    as success.  Extra simulator-specific checks belong in ``metadata``.
    """

    ik_ok: bool | None = None
    joint_limit_ok: bool | None = None
    collision_free: bool | None = None
    planner_success: bool | None = None
    task_success: bool | None = None
    min_clearance_m: float | None = None
    tracking_error_m: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "ik_ok",
            "joint_limit_ok",
            "collision_free",
            "planner_success",
            "task_success",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None")
        for name in ("min_clearance_m", "tracking_error_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_float(value, name))
        object.__setattr__(self, "metadata", _json_safe(self.metadata, "feasibility.metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ik_ok": self.ik_ok,
            "joint_limit_ok": self.joint_limit_ok,
            "collision_free": self.collision_free,
            "planner_success": self.planner_success,
            "task_success": self.task_success,
            "min_clearance_m": self.min_clearance_m,
            "tracking_error_m": self.tracking_error_m,
            "metadata": _json_safe(self.metadata, "feasibility.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeasibilityMetrics:
        return cls(
            ik_ok=data.get("ik_ok"),
            joint_limit_ok=data.get("joint_limit_ok"),
            collision_free=data.get("collision_free"),
            planner_success=data.get("planner_success"),
            task_success=data.get("task_success"),
            min_clearance_m=data.get("min_clearance_m"),
            tracking_error_m=data.get("tracking_error_m"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    """One timestamped sample in one semantic trajectory layer."""

    sequence: int
    step: int
    timestamp_s: float
    layer: TrajectoryLayer
    arms: tuple[ArmState, ...] = ()
    feasibility: FeasibilityMetrics = field(default_factory=FeasibilityMetrics)
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        object.__setattr__(self, "timestamp_s", _finite_float(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "layer", TrajectoryLayer(self.layer))
        arms = tuple(self.arms)
        arm_names = [state.arm for state in arms]
        if any(not name for name in arm_names):
            raise ValueError("all stored ArmState values must have an arm name")
        if len(set(arm_names)) != len(arm_names):
            raise ValueError("arm names must be unique within a sample")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "payload", _json_safe(self.payload, "payload"))
        object.__setattr__(self, "metadata", _json_safe(self.metadata, "sample.metadata"))

    def arm_state(self, arm: str) -> ArmState | None:
        return next((state for state in self.arms if state.arm == arm), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "step": self.step,
            "timestamp_s": self.timestamp_s,
            "layer": self.layer.value,
            "arms": [state.to_dict() for state in self.arms],
            "feasibility": self.feasibility.to_dict(),
            "payload": _json_safe(self.payload, "payload"),
            "metadata": _json_safe(self.metadata, "sample.metadata"),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        default_frame_id: str = "world",
    ) -> TrajectorySample:
        return cls(
            sequence=int(data["sequence"]),
            step=int(data["step"]),
            timestamp_s=float(data["timestamp_s"]),
            layer=TrajectoryLayer(data["layer"]),
            arms=tuple(
                ArmState.from_dict(item, default_frame_id=default_frame_id)
                for item in data.get("arms", ())
            ),
            feasibility=FeasibilityMetrics.from_dict(data.get("feasibility", {})),
            payload=data.get("payload"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryArtifact:
    """Serializable point-in-time snapshot of a trajectory recorder."""

    samples: tuple[TrajectorySample, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(
        # Keep Python 3.10 compatibility; datetime.UTC was added in 3.11.
        default_factory=lambda: datetime.now(timezone.utc).isoformat()  # noqa: UP017
    )
    schema_version: int = TRAJECTORY_SCHEMA_VERSION
    max_samples_per_layer: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported trajectory schema version {self.schema_version}; "
                f"expected {TRAJECTORY_SCHEMA_VERSION}"
            )
        samples = tuple(sorted(self.samples, key=lambda sample: sample.sequence))
        sequences = [sample.sequence for sample in samples]
        if len(set(sequences)) != len(sequences):
            raise ValueError("sample sequence numbers must be unique")
        if self.max_samples_per_layer is not None and self.max_samples_per_layer <= 0:
            raise ValueError("max_samples_per_layer must be positive")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "metadata", _json_safe(self.metadata, "artifact.metadata"))

    def samples_for(
        self,
        layer: TrajectoryLayer | str | None = None,
        arm: str | None = None,
        through_step: int | None = None,
    ) -> tuple[TrajectorySample, ...]:
        normalized_layer = None if layer is None else TrajectoryLayer(layer)
        return tuple(
            sample
            for sample in self.samples
            if (normalized_layer is None or sample.layer == normalized_layer)
            and (arm is None or sample.arm_state(arm) is not None)
            and (through_step is None or sample.step <= through_step)
        )

    def summary(self) -> dict[str, Any]:
        layer_counts = {
            layer.value: sum(sample.layer == layer for sample in self.samples)
            for layer in TRAJECTORY_LAYERS
        }
        arms = sorted({state.arm for sample in self.samples for state in sample.arms})
        frames = sorted(
            {
                state.ee_pose.frame_id
                for sample in self.samples
                for state in sample.arms
                if state.ee_pose is not None
            }
        )
        steps = [sample.step for sample in self.samples]
        timestamps = [sample.timestamp_s for sample in self.samples]
        bool_metrics: dict[str, dict[str, int]] = {}
        for name in (
            "ik_ok",
            "joint_limit_ok",
            "collision_free",
            "planner_success",
            "task_success",
        ):
            values = [getattr(sample.feasibility, name) for sample in self.samples]
            bool_metrics[name] = {
                "true": sum(value is True for value in values),
                "false": sum(value is False for value in values),
                "unknown": sum(value is None for value in values),
            }
        numeric_metrics: dict[str, dict[str, float] | None] = {}
        for name in ("min_clearance_m", "tracking_error_m"):
            values = [
                value
                for sample in self.samples
                if (value := getattr(sample.feasibility, name)) is not None
            ]
            numeric_metrics[name] = (
                None
                if not values
                else {"min": min(values), "max": max(values), "latest": values[-1]}
            )
        return {
            "schema_version": self.schema_version,
            "num_samples": len(self.samples),
            "layers": layer_counts,
            "arms": arms,
            "frames": frames,
            "step_range": None if not steps else [min(steps), max(steps)],
            "duration_s": 0.0 if not timestamps else max(timestamps) - min(timestamps),
            "feasibility": {**bool_metrics, **numeric_metrics},
            "metadata": _json_safe(self.metadata, "artifact.metadata"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "max_samples_per_layer": self.max_samples_per_layer,
            "metadata": _json_safe(self.metadata, "artifact.metadata"),
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrajectoryArtifact:
        metadata = data.get("metadata", {})
        default_frame_id = "world"
        if isinstance(metadata, Mapping):
            configured_frame = metadata.get("eef_frame_id")
            if isinstance(configured_frame, str) and configured_frame.strip():
                default_frame_id = configured_frame.strip()
            elif str(metadata.get("simulator", "")).strip().lower() == "libero":
                # Historical LIBERO artifacts stored controller-frame EEF poses
                # relative to the MuJoCo robot base, despite lacking a frame ID.
                default_frame_id = "robot0_base"
        return cls(
            schema_version=int(data.get("schema_version", TRAJECTORY_SCHEMA_VERSION)),
            created_at=str(
                data.get(
                    "created_at",
                    datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                )
            ),
            max_samples_per_layer=data.get("max_samples_per_layer"),
            metadata=metadata,
            samples=tuple(
                TrajectorySample.from_dict(sample, default_frame_id=default_frame_id)
                for sample in data.get("samples", ())
            ),
        )

    def dumps(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def loads(cls, payload: str) -> TrajectoryArtifact:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise TypeError("trajectory artifact JSON root must be an object")
        return cls.from_dict(data)

    def save_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        destination = Path(path)
        destination.write_text(self.dumps(indent=indent) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load_json(cls, path: str | Path) -> TrajectoryArtifact:
        return cls.loads(Path(path).read_text(encoding="utf-8"))


ArmInput = Mapping[str, ArmState] | Iterable[ArmState] | None
SampleCallback = Callable[[TrajectorySample], None]


class TrajectoryRecorder:
    """Bounded, thread-safe recorder with optional append subscribers."""

    def __init__(
        self,
        max_samples_per_layer: int = 2_000,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if max_samples_per_layer <= 0:
            raise ValueError("max_samples_per_layer must be positive")
        self.max_samples_per_layer = int(max_samples_per_layer)
        self._lock = threading.RLock()
        self._samples: dict[TrajectoryLayer, deque[TrajectorySample]] = {
            layer: deque(maxlen=self.max_samples_per_layer) for layer in TRAJECTORY_LAYERS
        }
        self._metadata: dict[str, Any] = _json_safe(metadata or {}, "recorder.metadata")
        self._created_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017
        self._start_time = time.monotonic()
        self._next_sequence = 0
        self._subscribers: list[SampleCallback] = []

    def __len__(self) -> int:
        with self._lock:
            return sum(len(samples) for samples in self._samples.values())

    @property
    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return _json_safe(self._metadata, "recorder.metadata")

    def reset(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        """Clear all layers and restart sequence/time counters.

        Metadata is preserved when omitted and replaced when supplied.
        Subscribers remain attached, allowing live renderers to survive episode
        resets.
        """

        with self._lock:
            for samples in self._samples.values():
                samples.clear()
            if metadata is not None:
                self._metadata = _json_safe(metadata, "recorder.metadata")
            self._created_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017
            self._start_time = time.monotonic()
            self._next_sequence = 0

    @staticmethod
    def _normalize_arms(arms: ArmInput) -> tuple[ArmState, ...]:
        if arms is None:
            return ()
        if isinstance(arms, Mapping):
            states: list[ArmState] = []
            for arm_name, state in arms.items():
                if not isinstance(state, ArmState):
                    raise TypeError("arms mapping values must be ArmState instances")
                normalized_name = str(arm_name)
                if not normalized_name:
                    raise ValueError("arm mapping keys must not be empty")
                if state.arm and state.arm != normalized_name:
                    raise ValueError(
                        f"ArmState arm {state.arm!r} conflicts with mapping key {normalized_name!r}"
                    )
                states.append(
                    state if state.arm == normalized_name else replace(state, arm=normalized_name)
                )
            return tuple(sorted(states, key=lambda state: state.arm))
        states = tuple(arms)
        if any(not isinstance(state, ArmState) for state in states):
            raise TypeError("arms iterable values must be ArmState instances")
        if any(not state.arm for state in states):
            raise ValueError("ArmState.arm is required when arms is not a mapping")
        return tuple(sorted(states, key=lambda state: state.arm))

    def append(
        self,
        layer: TrajectoryLayer | str,
        *,
        step: int | None = None,
        timestamp_s: float | None = None,
        arms: ArmInput = None,
        feasibility: FeasibilityMetrics | None = None,
        payload: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrajectorySample:
        normalized_layer = TrajectoryLayer(layer)
        normalized_arms = self._normalize_arms(arms)
        normalized_feasibility = feasibility or FeasibilityMetrics()
        if not isinstance(normalized_feasibility, FeasibilityMetrics):
            raise TypeError("feasibility must be a FeasibilityMetrics instance")

        with self._lock:
            sequence = self._next_sequence
            normalized_step = sequence if step is None else int(step)
            if normalized_step < 0:
                raise ValueError("step must be non-negative")
            normalized_timestamp = (
                time.monotonic() - self._start_time
                if timestamp_s is None
                else _finite_float(timestamp_s, "timestamp_s")
            )
            sample = TrajectorySample(
                sequence=sequence,
                step=normalized_step,
                timestamp_s=normalized_timestamp,
                layer=normalized_layer,
                arms=normalized_arms,
                feasibility=normalized_feasibility,
                payload=payload,
                metadata=metadata or {},
            )
            self._samples[normalized_layer].append(sample)
            self._next_sequence += 1
            subscribers = tuple(self._subscribers)

        # Rendering or telemetry must never make simulator stepping fail.
        for callback in subscribers:
            try:
                callback(sample)
            except Exception:  # pragma: no cover - defensive integration boundary
                LOGGER.exception("trajectory sample subscriber failed")
        return sample

    def append_raw(self, **kwargs: Any) -> TrajectorySample:
        return self.append(TrajectoryLayer.RAW, **kwargs)

    def append_planned(self, **kwargs: Any) -> TrajectorySample:
        return self.append(TrajectoryLayer.PLANNED, **kwargs)

    def append_commanded(self, **kwargs: Any) -> TrajectorySample:
        return self.append(TrajectoryLayer.COMMANDED, **kwargs)

    def append_executed(self, **kwargs: Any) -> TrajectorySample:
        return self.append(TrajectoryLayer.EXECUTED, **kwargs)

    def subscribe(self, callback: SampleCallback) -> Callable[[], None]:
        """Subscribe to new samples and return an idempotent unsubscribe hook."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._subscribers.append(callback)
        unsubscribed = False

        def unsubscribe() -> None:
            nonlocal unsubscribed
            with self._lock:
                if unsubscribed:
                    return
                unsubscribed = True
                with suppress(ValueError):
                    self._subscribers.remove(callback)

        return unsubscribe

    def snapshot(self) -> TrajectoryArtifact:
        with self._lock:
            samples = tuple(
                sorted(
                    (sample for layer in TRAJECTORY_LAYERS for sample in self._samples[layer]),
                    key=lambda sample: sample.sequence,
                )
            )
            return TrajectoryArtifact(
                samples=samples,
                metadata=_json_safe(self._metadata, "recorder.metadata"),
                created_at=self._created_at,
                max_samples_per_layer=self.max_samples_per_layer,
            )

    def summary(self) -> dict[str, Any]:
        return self.snapshot().summary()

    def save_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        return self.snapshot().save_json(path, indent=indent)

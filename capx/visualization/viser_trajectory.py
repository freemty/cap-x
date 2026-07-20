"""Viser renderer for live and recorded CaP-X trajectories.

Viser is deliberately imported only by :func:`create_viser_server`.  Passing a
real or fake server to :class:`ViserTrajectoryRenderer` therefore works even in
an environment where the optional package is absent.  The renderer uses APIs
available in Viser 0.2.15 through 1.x: batched line segments, batched axes, and
basic GUI handles.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

from .trajectory import (
    TRAJECTORY_LAYERS,
    ArmState,
    TrajectoryArtifact,
    TrajectoryLayer,
    TrajectoryRecorder,
    TrajectorySample,
)

LOGGER = logging.getLogger(__name__)

LayerColor = tuple[int, int, int]

DEFAULT_LAYER_COLORS: Mapping[TrajectoryLayer, LayerColor] = {
    TrajectoryLayer.RAW: (125, 125, 125),
    TrajectoryLayer.PLANNED: (45, 110, 235),
    TrajectoryLayer.COMMANDED: (235, 145, 35),
    TrajectoryLayer.EXECUTED: (40, 165, 85),
}
INFEASIBLE_COLOR: LayerColor = (225, 45, 45)

RobotStateSetter = Callable[[ArmState], None]


@dataclass(frozen=True, slots=True)
class _TraceRun:
    """One contiguous, single-frame, single-plan EEF trace."""

    frame_id: str
    segment_label: str
    samples: tuple[TrajectorySample, ...]
    positions: np.ndarray
    wxyzs: np.ndarray


def create_viser_server(**kwargs: Any) -> Any:
    """Create a Viser server without making Viser a core import dependency."""

    try:
        import viser
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "Viser trajectory rendering requires `viser>=0.2.15`; "
            "install the visualization dependency before starting a server."
        ) from exc
    return viser.ViserServer(**kwargs)


def _safe_remove(handle: Any) -> None:
    if handle is None:
        return
    try:
        handle.remove()
    except Exception:  # pragma: no cover - defensive third-party boundary
        LOGGER.debug("failed to remove Viser handle", exc_info=True)


def _is_infeasible(sample: TrajectorySample) -> bool:
    metrics = sample.feasibility
    checks = (
        metrics.ik_ok,
        metrics.joint_limit_ok,
        metrics.collision_free,
        metrics.planner_success,
    )
    return any(check is False for check in checks) or (
        metrics.min_clearance_m is not None and metrics.min_clearance_m < 0.0
    )


def _clean_path_component(value: str) -> str:
    return value.strip().replace("/", "_").replace(" ", "_") or "unnamed"


def _segment_label(sample: TrajectorySample) -> str:
    """Return the strongest available continuity identifier for a sample."""

    for key in ("segment_id", "plan_id"):
        if key in sample.metadata and sample.metadata[key] is not None:
            encoded = json.dumps(sample.metadata[key], sort_keys=True, separators=(",", ":"))
            return f"{key}={encoded}"
    return "continuous"


class ViserTrajectoryRenderer:
    """Render a recorder or artifact as batched paths with a scrubber.

    ``robot_state_setters`` can connect existing ``ViserUrdf`` instances to the
    time slider.  Each callback receives the latest state for its arm at the
    selected timestep.  FK is intentionally not performed here because joint
    conventions are embodiment-specific.
    """

    def __init__(
        self,
        server: Any,
        source: TrajectoryRecorder | TrajectoryArtifact,
        *,
        root: str = "/trajectory",
        line_width: float = 3.0,
        axes_length: float = 0.06,
        axes_radius: float = 0.002,
        max_axes_per_trace: int = 64,
        layer_colors: Mapping[TrajectoryLayer | str, LayerColor] | None = None,
        robot_state_setters: Mapping[str, RobotStateSetter] | None = None,
        frame_id: str | None = None,
        connect_raw_targets: bool = False,
        auto_subscribe: bool = True,
        update_every_n: int = 1,
        timeline_max: int | None = None,
    ) -> None:
        if not isinstance(source, (TrajectoryRecorder, TrajectoryArtifact)):
            raise TypeError("source must be a TrajectoryRecorder or TrajectoryArtifact")
        if line_width <= 0 or axes_length <= 0 or axes_radius <= 0:
            raise ValueError("line and axes dimensions must be positive")
        if max_axes_per_trace <= 0:
            raise ValueError("max_axes_per_trace must be positive")
        if update_every_n <= 0:
            raise ValueError("update_every_n must be positive")
        if not hasattr(server, "scene") or not hasattr(server, "gui"):
            raise TypeError("server must expose scene and gui APIs")
        if not hasattr(server.scene, "add_line_segments"):
            raise RuntimeError(
                "This renderer requires Viser's add_line_segments API (available in viser>=0.2.15)."
            )
        if not hasattr(server.scene, "add_batched_axes"):
            raise RuntimeError("This renderer requires Viser's add_batched_axes API")

        self.server = server
        self.source = source
        root_component = root.strip("/")
        self.root = f"/{root_component}" if root_component else ""
        self.line_width = float(line_width)
        self.axes_length = float(axes_length)
        self.axes_radius = float(axes_radius)
        self.max_axes_per_trace = int(max_axes_per_trace)
        self.robot_state_setters = dict(robot_state_setters or {})
        if frame_id is not None and (not isinstance(frame_id, str) or not frame_id.strip()):
            raise ValueError("frame_id must be a non-empty string or None")
        self._requested_frame_id = None if frame_id is None else frame_id.strip()
        self._active_frame_id: str | None = None
        self.connect_raw_targets = bool(connect_raw_targets)
        self.update_every_n = int(update_every_n)
        self._lock = threading.RLock()
        self._closed = False
        self._append_count = 0
        self._follow_latest = True
        self._selected_step: int | None = None
        self._scene_handles: dict[str, Any] = {}
        self._gui_handles: list[Any] = []
        self._unsubscribe: Callable[[], None] | None = None
        self._layer_visibility = dict.fromkeys(TRAJECTORY_LAYERS, True)
        self._show_axes = True
        self._colors = dict(DEFAULT_LAYER_COLORS)
        if layer_colors is not None:
            for layer, color in layer_colors.items():
                normalized = tuple(int(component) for component in color)
                if len(normalized) != 3 or any(not 0 <= value <= 255 for value in normalized):
                    raise ValueError("layer colors must be RGB triples in [0, 255]")
                self._colors[TrajectoryLayer(layer)] = normalized  # type: ignore[assignment]

        artifact = self._artifact()
        latest_step = max((sample.step for sample in artifact.samples), default=0)
        if timeline_max is None:
            if isinstance(source, TrajectoryRecorder):
                timeline_max = max(1, source.max_samples_per_layer - 1, latest_step)
            else:
                timeline_max = max(1, latest_step)
        if timeline_max < 1:
            raise ValueError("timeline_max must be at least 1")
        self._timeline_max = int(timeline_max)

        self._build_gui(initial_step=latest_step)
        if isinstance(source, TrajectoryRecorder) and auto_subscribe:
            self._unsubscribe = source.subscribe(self._on_sample)
        self.update(artifact)

    def _artifact(self) -> TrajectoryArtifact:
        return (
            self.source.snapshot() if isinstance(self.source, TrajectoryRecorder) else self.source
        )

    def _build_gui(self, *, initial_step: int) -> None:
        gui = self.server.gui
        slider = gui.add_slider(
            "Trajectory timestep",
            min=0,
            max=self._timeline_max,
            step=1,
            initial_value=min(initial_step, self._timeline_max),
        )
        slider.on_update(lambda _event: self._on_timeline_changed())
        self.timeline_handle = slider
        self._gui_handles.append(slider)

        self.layer_handles: dict[TrajectoryLayer, Any] = {}
        for layer in TRAJECTORY_LAYERS:
            handle = gui.add_checkbox(
                f"Show {layer.value}", initial_value=self._layer_visibility[layer]
            )
            handle.on_update(
                lambda _event, selected_layer=layer: self._on_layer_changed(selected_layer)
            )
            self.layer_handles[layer] = handle
            self._gui_handles.append(handle)

        axes_handle = gui.add_checkbox("Show EEF axes", initial_value=True)
        axes_handle.on_update(lambda _event: self._on_axes_changed())
        self.axes_handle = axes_handle
        self._gui_handles.append(axes_handle)

        self.status_handle = gui.add_markdown("**Trajectory:** no samples")
        self._gui_handles.append(self.status_handle)

    def _grow_timeline(self, latest_step: int) -> None:
        """Occasionally rebuild the slider when a live run outgrows its bound."""

        if latest_step <= self._timeline_max:
            return
        old_handle = self.timeline_handle
        self._timeline_max = max(latest_step, self._timeline_max * 2)
        slider = self.server.gui.add_slider(
            "Trajectory timestep",
            min=0,
            max=self._timeline_max,
            step=1,
            initial_value=latest_step,
        )
        slider.on_update(lambda _event: self._on_timeline_changed())
        self.timeline_handle = slider
        with suppress(ValueError):
            self._gui_handles.remove(old_handle)
        self._gui_handles.append(slider)
        _safe_remove(old_handle)

    def _on_timeline_changed(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._follow_latest = False
            self._selected_step = int(self.timeline_handle.value)
        self.update()

    def _on_layer_changed(self, layer: TrajectoryLayer) -> None:
        with self._lock:
            if self._closed:
                return
            self._layer_visibility[layer] = bool(self.layer_handles[layer].value)
        self.update()

    def _on_axes_changed(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._show_axes = bool(self.axes_handle.value)
        self.update()

    def _on_sample(self, _sample: TrajectorySample) -> None:
        with self._lock:
            if self._closed:
                return
            self._append_count += 1
            should_update = self._append_count % self.update_every_n == 0
        if should_update:
            self.update()

    def set_timestep(self, step: int | None) -> None:
        """Select a simulation step, or pass ``None`` to follow the latest."""

        if step is not None and step < 0:
            raise ValueError("step must be non-negative")
        with self._lock:
            if self._closed:
                return
            self._follow_latest = step is None
            self._selected_step = step
            if step is not None:
                self.timeline_handle.value = min(int(step), self._timeline_max)
        self.update()

    def set_layer_visible(self, layer: TrajectoryLayer | str, visible: bool) -> None:
        normalized = TrajectoryLayer(layer)
        with self._lock:
            if self._closed:
                return
            self._layer_visibility[normalized] = bool(visible)
            self.layer_handles[normalized].value = bool(visible)
        self.update()

    def set_robot_state_setter(self, arm: str, setter: RobotStateSetter | None) -> None:
        """Add, replace, or remove an embodiment-specific robot playback hook."""

        normalized_arm = str(arm).strip()
        if not normalized_arm:
            raise ValueError("arm must not be empty")
        if setter is not None and not callable(setter):
            raise TypeError("setter must be callable or None")
        with self._lock:
            if self._closed:
                return
            if setter is None:
                self.robot_state_setters.pop(normalized_arm, None)
            else:
                self.robot_state_setters[normalized_arm] = setter
        self.update()

    def set_frame_id(self, frame_id: str | None) -> None:
        """Show one coordinate frame, or pass ``None`` for automatic selection.

        Coordinates with different frame IDs are never overlaid.  A simulator
        must transform poses into a common frame before recording if it wants
        them to appear together.
        """

        if frame_id is not None and (not isinstance(frame_id, str) or not frame_id.strip()):
            raise ValueError("frame_id must be a non-empty string or None")
        with self._lock:
            if self._closed:
                return
            self._requested_frame_id = None if frame_id is None else frame_id.strip()
        self.update()

    def _atomic(self):  # type: ignore[no-untyped-def]
        atomic = getattr(self.server, "atomic", None)
        return atomic() if callable(atomic) else nullcontext()

    def _remove_scene_key(self, key: str) -> None:
        handle = self._scene_handles.pop(key, None)
        _safe_remove(handle)

    def _trace_runs(
        self,
        artifact: TrajectoryArtifact,
        *,
        layer: TrajectoryLayer,
        arm: str,
        through_step: int | None,
        frame_id: str,
    ) -> tuple[_TraceRun, ...]:
        runs: list[_TraceRun] = []
        samples: list[TrajectorySample] = []
        positions: list[tuple[float, float, float]] = []
        wxyzs: list[tuple[float, float, float, float]] = []
        current_key: tuple[str, str] | None = None

        def flush() -> None:
            nonlocal samples, positions, wxyzs, current_key
            if samples and current_key is not None:
                runs.append(
                    _TraceRun(
                        frame_id=current_key[0],
                        segment_label=current_key[1],
                        samples=tuple(samples),
                        positions=np.asarray(positions, dtype=np.float32).reshape((-1, 3)),
                        wxyzs=np.asarray(wxyzs, dtype=np.float32).reshape((-1, 4)),
                    )
                )
            samples = []
            positions = []
            wxyzs = []
            current_key = None

        for sample in artifact.samples_for(layer=layer, arm=arm, through_step=through_step):
            state = sample.arm_state(arm)
            if state is None or state.ee_pose is None:
                # Missing FK is an unknown interval, not permission to draw a
                # straight shortcut between surrounding samples.
                flush()
                continue
            if state.ee_pose.frame_id != frame_id:
                flush()
                continue
            key = (state.ee_pose.frame_id, _segment_label(sample))
            if current_key is not None and key != current_key:
                flush()
            current_key = key
            samples.append(sample)
            positions.append(state.ee_pose.position)
            wxyzs.append(state.ee_pose.wxyz)
        flush()
        return tuple(runs)

    def _select_frame(self, frames: list[str]) -> str | None:
        if self._requested_frame_id is not None:
            return self._requested_frame_id
        if "world" in frames:
            return "world"
        return frames[0] if frames else None

    def _update_robot_states(self, artifact: TrajectoryArtifact, through_step: int | None) -> None:
        # Prefer the state closest to physical execution.
        priority = (
            TrajectoryLayer.EXECUTED,
            TrajectoryLayer.COMMANDED,
            TrajectoryLayer.PLANNED,
            TrajectoryLayer.RAW,
        )
        for arm, setter in self.robot_state_setters.items():
            selected: ArmState | None = None
            for layer in priority:
                if not self._layer_visibility[layer]:
                    continue
                candidates = artifact.samples_for(layer=layer, arm=arm, through_step=through_step)
                if candidates:
                    selected = candidates[-1].arm_state(arm)
                    if selected is not None:
                        break
            if selected is None:
                continue
            try:
                setter(selected)
            except Exception:  # pragma: no cover - external URDF integration
                LOGGER.exception("robot state setter failed for arm %s", arm)

    def update(self, artifact: TrajectoryArtifact | None = None) -> None:
        """Refresh all batched scene nodes from a consistent snapshot."""

        with self._lock:
            if self._closed:
                return
            artifact = self._artifact() if artifact is None else artifact
            if not isinstance(artifact, TrajectoryArtifact):
                raise TypeError("artifact must be a TrajectoryArtifact")
            latest_step = max((sample.step for sample in artifact.samples), default=0)
            self._grow_timeline(latest_step)
            through_step = None if self._follow_latest else self._selected_step
            if self._follow_latest:
                self.timeline_handle.value = min(latest_step, self._timeline_max)

            summary = artifact.summary()
            frames = list(summary["frames"])
            active_frame = self._select_frame(frames)
            self._active_frame_id = active_frame
            arms = sorted({state.arm for sample in artifact.samples for state in sample.arms})
            expected_scene_keys: set[str] = set()
            with self._atomic():
                for layer in TRAJECTORY_LAYERS:
                    if not self._layer_visibility[layer]:
                        continue
                    for arm in arms:
                        if active_frame is None:
                            continue
                        runs = self._trace_runs(
                            artifact,
                            layer=layer,
                            arm=arm,
                            through_step=through_step,
                            frame_id=active_frame,
                        )
                        component = _clean_path_component(arm)
                        frame_component = _clean_path_component(active_frame)
                        for run_index, run in enumerate(runs):
                            trace_root = (
                                f"{self.root}/{layer.value}/{component}/"
                                f"{frame_component}/run_{run_index}"
                            )
                            line_key = f"{trace_root}/path"
                            axes_key = f"{trace_root}/axes"
                            if len(run.positions) >= 2 and (
                                layer is not TrajectoryLayer.RAW or self.connect_raw_targets
                            ):
                                segments = np.stack((run.positions[:-1], run.positions[1:]), axis=1)
                                colors = np.empty(segments.shape, dtype=np.uint8)
                                for index in range(len(segments)):
                                    color = (
                                        INFEASIBLE_COLOR
                                        if _is_infeasible(run.samples[index])
                                        or _is_infeasible(run.samples[index + 1])
                                        else self._colors[layer]
                                    )
                                    colors[index, :, :] = color
                                self._scene_handles[line_key] = self.server.scene.add_line_segments(
                                    line_key,
                                    points=segments,
                                    colors=colors,
                                    line_width=self.line_width,
                                    visible=True,
                                )
                                expected_scene_keys.add(line_key)
                            if self._show_axes and len(run.positions) > 0:
                                if len(run.positions) > self.max_axes_per_trace:
                                    indices = np.linspace(
                                        0,
                                        len(run.positions) - 1,
                                        self.max_axes_per_trace,
                                        dtype=int,
                                    )
                                    axes_positions = run.positions[indices]
                                    axes_wxyzs = run.wxyzs[indices]
                                else:
                                    axes_positions = run.positions
                                    axes_wxyzs = run.wxyzs
                                self._scene_handles[axes_key] = self.server.scene.add_batched_axes(
                                    axes_key,
                                    batched_wxyzs=axes_wxyzs,
                                    batched_positions=axes_positions,
                                    axes_length=self.axes_length,
                                    axes_radius=self.axes_radius,
                                    visible=True,
                                )
                                expected_scene_keys.add(axes_key)

                for key in tuple(self._scene_handles):
                    if key not in expected_scene_keys:
                        self._remove_scene_key(key)

                self._update_robot_states(artifact, through_step)
                failed = sum(_is_infeasible(sample) for sample in artifact.samples)
                selected_label = "latest" if through_step is None else str(through_step)
                hidden_frames = [frame for frame in frames if frame != active_frame]
                frame_label = active_frame or "none"
                if hidden_frames:
                    frame_label += f" (hidden: {', '.join(hidden_frames)})"
                elif active_frame is not None and active_frame not in frames:
                    frame_label += " (not present)"
                self.status_handle.content = (
                    f"**Trajectory:** {summary['num_samples']} samples  \n"
                    f"**Timestep:** {selected_label} / {latest_step}  \n"
                    f"**Arms:** {', '.join(summary['arms']) or 'none'}  \n"
                    f"**Frame:** {frame_label}  \n"
                    f"**Flagged infeasible:** {failed}"
                )

    def close(self) -> None:
        """Detach callbacks and remove every scene/GUI handle owned here."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            unsubscribe = self._unsubscribe
            self._unsubscribe = None
            scene_handles = tuple(self._scene_handles.values())
            gui_handles = tuple(self._gui_handles)
            self._scene_handles.clear()
            self._gui_handles.clear()
        if unsubscribe is not None:
            unsubscribe()
        for handle in scene_handles:
            _safe_remove(handle)
        for handle in reversed(gui_handles):
            _safe_remove(handle)

    def __enter__(self) -> ViserTrajectoryRenderer:
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()

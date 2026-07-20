"""Trajectory recording and optional Viser rendering utilities."""

from __future__ import annotations

from .trajectory import (
    TRAJECTORY_LAYERS,
    TRAJECTORY_SCHEMA_VERSION,
    ArmState,
    EndEffectorPose,
    FeasibilityMetrics,
    TrajectoryArtifact,
    TrajectoryLayer,
    TrajectoryRecorder,
    TrajectorySample,
)

__all__ = [
    "TRAJECTORY_LAYERS",
    "TRAJECTORY_SCHEMA_VERSION",
    "ArmState",
    "EndEffectorPose",
    "FeasibilityMetrics",
    "TrajectoryArtifact",
    "TrajectoryLayer",
    "TrajectoryRecorder",
    "TrajectorySample",
    "ViserTrajectoryRenderer",
    "create_viser_server",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    # Keep the core recorder importable in minimal simulator environments.  The
    # renderer module itself also imports Viser lazily, but deferring it here
    # keeps package import time and optional NumPy use minimal.
    if name in {"ViserTrajectoryRenderer", "create_viser_server"}:
        from .viser_trajectory import ViserTrajectoryRenderer, create_viser_server

        return {
            "ViserTrajectoryRenderer": ViserTrajectoryRenderer,
            "create_viser_server": create_viser_server,
        }[name]
    raise AttributeError(name)

"""Offline trajectory artifact playback with Viser.

Usage::

    python -m capx.visualization.replay trajectory.json --port 8080
    python -m capx.visualization.replay trajectory.json --urdf panda.urdf
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory import ArmState, TrajectoryArtifact
from .viser_trajectory import ViserTrajectoryRenderer, create_viser_server

LOGGER = logging.getLogger(__name__)

_CANONICAL_PANDA_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
_PANDA_JOINT_ALIASES = {
    **{name: name for name in _CANONICAL_PANDA_JOINTS},
    **{f"robot0_joint{index}": f"panda_joint{index}" for index in range(1, 8)},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a CaP-X trajectory in Viser")
    parser.add_argument("artifact", type=Path, help="Trajectory JSON artifact")
    parser.add_argument("--host", default="0.0.0.0", help="Viser bind host")
    parser.add_argument("--port", type=int, default=8080, help="Viser bind port")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Optional URDF for joint-state playback (first recorded arm)",
    )
    parser.add_argument(
        "--root", default="/trajectory", help="Viser scene root for trajectory nodes"
    )
    return parser


def _canonical_panda_names(names: tuple[str, ...]) -> tuple[str, ...] | None:
    """Canonicalize only a complete, known seven-joint Panda name set."""

    if len(names) != len(_CANONICAL_PANDA_JOINTS):
        return None
    try:
        canonical = tuple(_PANDA_JOINT_ALIASES[name] for name in names)
    except KeyError:
        return None
    if set(canonical) != set(_CANONICAL_PANDA_JOINTS):
        return None
    return canonical


def _joint_configuration_for_urdf(
    state: ArmState, urdf_joint_names: tuple[str, ...]
) -> np.ndarray | None:
    """Map recorded joints to URDF order without positional guesswork.

    Arbitrary embodiments require an exact joint-name set.  The only alias
    mapping is the complete canonical Panda set used by LIBERO
    (``robot0_joint1..7`` <-> ``panda_joint1..7``).  Every path requires strict
    dimensional equality.
    """

    source_names = tuple(state.joint_names)
    target_names = tuple(str(name) for name in urdf_joint_names)
    values = tuple(state.joint_positions)
    if (
        not values
        or len(source_names) != len(values)
        or len(target_names) != len(values)
        or len(set(target_names)) != len(target_names)
    ):
        return None

    if set(source_names) == set(target_names):
        by_name = dict(zip(source_names, values, strict=True))
        return np.asarray([by_name[name] for name in target_names], dtype=float)

    canonical_source = _canonical_panda_names(source_names)
    canonical_target = _canonical_panda_names(target_names)
    if canonical_source is None or canonical_target is None:
        return None
    by_canonical_name = dict(zip(canonical_source, values, strict=True))
    return np.asarray([by_canonical_name[name] for name in canonical_target], dtype=float)


def _make_urdf_setter(server: Any, urdf_path: Path) -> tuple[Any, Any]:
    try:
        from viser.extras import ViserUrdf
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("--urdf requires viser with viser.extras.ViserUrdf") from exc
    urdf = ViserUrdf(server, urdf_or_path=urdf_path, load_meshes=True)
    joint_names = tuple(urdf.get_actuated_joint_names())

    def set_state(state: ArmState) -> None:
        configuration = _joint_configuration_for_urdf(state, joint_names)
        if configuration is None:
            LOGGER.warning(
                "URDF playback skipped: recorded joints %s do not safely map to %s",
                state.joint_names,
                joint_names,
            )
            return
        urdf.update_cfg(configuration)

    return urdf, set_state


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = TrajectoryArtifact.load_json(args.artifact)
    server = create_viser_server(host=args.host, port=args.port)
    urdf = None
    setters = None
    if args.urdf is not None:
        arms = artifact.summary()["arms"]
        if not arms:
            raise ValueError("artifact contains no named arms for --urdf playback")
        urdf, setter = _make_urdf_setter(server, args.urdf)
        setters = {arms[0]: setter}
    renderer = ViserTrajectoryRenderer(
        server,
        source=artifact,
        root=args.root,
        robot_state_setters=setters,
    )
    try:
        sleep_forever = getattr(server, "sleep_forever", None)
        if callable(sleep_forever):
            sleep_forever()
        else:  # pragma: no cover - compatibility with very old Viser
            import time

            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    finally:
        renderer.close()
        if urdf is not None:
            remove = getattr(urdf, "remove", None)
            if callable(remove):
                remove()
        stop = getattr(server, "stop", None)
        if callable(stop):
            stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

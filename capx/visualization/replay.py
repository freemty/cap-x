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


def _make_urdf_setter(server: Any, urdf_path: Path) -> tuple[Any, Any]:
    try:
        from viser.extras import ViserUrdf
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("--urdf requires viser with viser.extras.ViserUrdf") from exc
    urdf = ViserUrdf(server, urdf_or_path=urdf_path, load_meshes=True)
    joint_names = tuple(urdf.get_actuated_joint_names())

    def set_state(state: ArmState) -> None:
        if not state.joint_positions:
            return
        by_name = dict(zip(state.joint_names, state.joint_positions, strict=True))
        missing = [name for name in joint_names if name not in by_name]
        if missing:
            LOGGER.debug("URDF playback skipped; missing joints: %s", missing)
            return
        urdf.update_cfg(np.asarray([by_name[name] for name in joint_names], dtype=float))

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

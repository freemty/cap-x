"""Helpers for wiring simulator-side Viser servers into the web UI."""

from __future__ import annotations

import socket
from collections.abc import Mapping, MutableMapping
from typing import Any
from urllib.request import urlopen


DEFAULT_VISER_PORTS = tuple(range(8080, 8100))


def probe_viser_port(
    *,
    preferred: int | None = None,
    cached: int | None = None,
    fallback_ports: tuple[int, ...] = DEFAULT_VISER_PORTS,
) -> int | None:
    """Find a reachable Viser HTTP server without crossing session boundaries.

    A session-owned port is authoritative. Legacy fallback probing is used only
    when the environment could not report its selected port.
    """

    candidates: list[int] = []
    possible_ports = (preferred,) if preferred is not None else (cached, *fallback_ports)
    for candidate in possible_ports:
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)

    for port in candidates:
        try:
            # Most fallback ports are closed. A short TCP probe avoids waiting
            # for a full HTTP timeout on every candidate.
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                pass
            with urlopen(f"http://localhost:{port}/", timeout=0.5):
                pass
            return port
        except (OSError, TimeoutError):
            continue
    return None


def enable_web_visualization(env_factory: MutableMapping[str, Any]) -> int:
    """Enable rendering and Viser on execution and low-level environment configs.

    Configuration objects are recursively instantiated, so setting flags only on
    the outer execution environment is too late for nested simulator objects.  We
    update both layers before instantiation and return the number of mappings that
    were changed.  The function intentionally ignores unrelated mappings.
    """

    changed = 0

    def visit(value: Any) -> None:
        nonlocal changed
        if isinstance(value, MutableMapping):
            target = str(value.get("_target_", ""))
            is_execution_config = target.endswith("CodeExecEnvConfig")
            is_simulator = ".envs.simulators." in target
            if is_execution_config or is_simulator:
                if value.get("enable_render") is not True:
                    value["enable_render"] = True
                if value.get("viser_debug") is not True:
                    value["viser_debug"] = True
                changed += 1
            for child in tuple(value.values()):
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(env_factory)

    # Older configs may omit the CodeExecEnvConfig target while still using the
    # conventional top-level ``cfg`` mapping.
    outer_cfg = env_factory.get("cfg")
    if isinstance(outer_cfg, MutableMapping):
        outer_cfg["enable_render"] = True
        outer_cfg["viser_debug"] = True

    return changed


def find_low_level_env(env: Any) -> Any | None:
    """Find a wrapped low-level environment without assuming a wrapper class."""

    pending = [env]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "viser_server") or hasattr(candidate, "trajectory_recorder"):
            return candidate
        for attribute in ("low_level_env", "_env", "env"):
            child = getattr(candidate, attribute, None)
            if child is not None and child is not candidate:
                pending.append(child)
    return None


def get_viser_port(env: Any) -> int | None:
    """Return the exact port of the Viser server owned by ``env``, if any."""

    low_level = find_low_level_env(env)
    server = getattr(low_level, "viser_server", None) if low_level is not None else None
    get_port = getattr(server, "get_port", None)
    if not callable(get_port):
        return None
    try:
        port = int(get_port())
    except (TypeError, ValueError, RuntimeError):
        return None
    return port if 0 < port < 65536 else None


def trajectory_summary(env: Any) -> Mapping[str, Any] | None:
    """Read a small JSON-safe trajectory summary from a wrapped environment."""

    low_level = find_low_level_env(env)
    if low_level is None:
        return None
    summary_fn = getattr(low_level, "trajectory_summary", None)
    if not callable(summary_fn):
        return None
    summary = summary_fn()
    return summary if isinstance(summary, Mapping) else None

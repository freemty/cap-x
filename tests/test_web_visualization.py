from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any

from capx.web.models import SessionState
from capx.web.session_manager import SessionManager
from capx.web.visualization import (
    enable_web_visualization,
    find_low_level_env,
    get_viser_port,
    probe_viser_port,
    trajectory_summary,
)


def test_enable_web_visualization_reaches_nested_simulator() -> None:
    factory = {
        "_target_": "capx.envs.tasks.robotwin.robotwin_env.RoboTwinCodeEnv",
        "cfg": {
            "_target_": "capx.envs.tasks.base.CodeExecEnvConfig",
            "low_level": {
                "_target_": "capx.envs.simulators.robotwin.RoboTwinEnv",
                "task_name": "stack_blocks_two",
            },
        },
    }

    changed = enable_web_visualization(factory)

    assert changed == 2
    assert factory["cfg"]["viser_debug"] is True
    assert factory["cfg"]["enable_render"] is True
    assert factory["cfg"]["low_level"]["viser_debug"] is True
    assert factory["cfg"]["low_level"]["enable_render"] is True


def test_enable_web_visualization_ignores_unrelated_mappings() -> None:
    factory = {"cfg": {"model": {"viser_debug": False}}}

    assert enable_web_visualization(factory) == 0
    assert factory["cfg"]["viser_debug"] is True
    assert factory["cfg"]["model"]["viser_debug"] is False


def test_find_low_level_viser_port_and_summary() -> None:
    class Server:
        def get_port(self) -> int:
            return 8087

    class LowLevel:
        viser_server = Server()
        trajectory_recorder = object()

        def trajectory_summary(self) -> dict[str, int]:
            return {"executed": 12}

    class Wrapper:
        low_level_env = LowLevel()

    wrapper = Wrapper()
    assert find_low_level_env(wrapper) is wrapper.low_level_env
    assert get_viser_port(wrapper) == 8087
    assert trajectory_summary(wrapper) == {"executed": 12}


def test_session_cleanup_closes_env_on_owning_executor() -> None:
    async def scenario() -> None:
        manager = SessionManager()
        session = await manager.create_session()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        owner_thread = await asyncio.get_running_loop().run_in_executor(
            executor, threading.get_ident
        )

        class Env:
            close_thread: int | None = None

            def close(self) -> None:
                self.close_thread = threading.get_ident()

        env = Env()
        session.env = env
        session.env_executor = executor
        session.viser_port = 8084
        session.state = SessionState.COMPLETE

        assert manager.get_active_session() is session
        await manager.remove_session(session.session_id)

        assert env.close_thread == owner_thread
        assert await manager.get_session(session.session_id) is None

    asyncio.run(scenario())


def test_stop_completed_session_closes_retained_visualization() -> None:
    async def scenario() -> None:
        manager = SessionManager()
        session = await manager.create_session()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        owner_thread = await asyncio.get_running_loop().run_in_executor(
            executor, threading.get_ident
        )

        class Env:
            close_thread: int | None = None

            def close(self) -> None:
                self.close_thread = threading.get_ident()

        env = Env()
        session.env = env
        session.env_executor = executor
        session.viser_port = 8088
        session.state = SessionState.COMPLETE

        assert await manager.stop_session(session.session_id) is True
        assert env.close_thread == owner_thread
        assert session.env is None
        assert session.viser_port is None
        assert session.state is SessionState.IDLE

    asyncio.run(scenario())


def test_session_owned_viser_port_never_falls_back(monkeypatch: Any) -> None:
    from capx.web import visualization

    attempted: list[int] = []

    def refuse_connection(address: tuple[str, int], timeout: float) -> None:
        del timeout
        attempted.append(address[1])
        raise OSError("closed")

    monkeypatch.setattr(visualization.socket, "create_connection", refuse_connection)

    assert probe_viser_port(preferred=8097, cached=8081) is None
    assert attempted == [8097]

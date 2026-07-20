from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from typing import Any
from urllib.parse import urlparse

import pytest

from capx.web.models import SessionState
from capx.web.session_manager import (
    SessionManager,
    interrupt_owner_call,
    run_on_owner_executor,
)
from capx.web.visualization import (
    enable_web_visualization,
    fetch_viser_http,
    find_low_level_env,
    get_viser_port,
    probe_viser_port,
    reset_render_and_viser_port,
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


def test_cleanup_interrupts_blocked_owner_call_and_reclaims_executor() -> None:
    async def scenario() -> None:
        manager = SessionManager()
        session = await manager.create_session()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        session.env_executor = executor

        started = threading.Event()
        stopped = threading.Event()
        owner_threads: list[int] = []

        class Env:
            close_thread: int | None = None

            def close(self) -> None:
                self.close_thread = threading.get_ident()

        env = Env()
        session.env = env

        def blocked_python_loop() -> None:
            owner_threads.append(threading.get_ident())
            started.set()
            try:
                while True:
                    time.sleep(0.005)
            finally:
                stopped.set()

        session.task = asyncio.create_task(
            run_on_owner_executor(session, blocked_python_loop)
        )
        assert await asyncio.to_thread(started.wait, 1.0)

        # Cleanup injects before task cancellation, waits for the worker, and
        # closes on that same owner thread rather than queueing behind the loop.
        await asyncio.wait_for(manager.remove_session(session.session_id), timeout=1.0)

        assert stopped.is_set()
        assert env.close_thread == owner_threads[0]
        assert session.env_executor_poisoned is False
        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            executor.submit(lambda: None)

    asyncio.run(scenario())


def test_cleanup_retains_cancelled_initialization_future_and_closes_env() -> None:
    async def scenario() -> None:
        manager = SessionManager()
        session = await manager.create_session()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        session.env_executor = executor
        started = threading.Event()

        class Env:
            close_thread: int | None = None

            def close(self) -> None:
                self.close_thread = threading.get_ident()

        env = Env()

        def instantiate_and_register() -> Env:
            started.set()
            try:
                while True:
                    time.sleep(0.005)
            except KeyboardInterrupt:
                # Model a constructor that completes at the cancellation edge.
                session.env = env
                return env

        session.task = asyncio.create_task(
            run_on_owner_executor(session, instantiate_and_register)
        )
        assert await asyncio.to_thread(started.wait, 1.0)

        await asyncio.wait_for(manager.remove_session(session.session_id), timeout=1.0)

        assert env.close_thread is not None
        assert session.owner_call_future is None
        assert session.env_executor is None

    asyncio.run(scenario())


def test_unreleased_owner_call_is_explicitly_poisoned() -> None:
    async def scenario() -> None:
        manager = SessionManager()
        session = await manager.create_session()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        session.env_executor = executor
        started = threading.Event()
        release = threading.Event()

        def ignores_interrupt_until_released() -> None:
            started.set()
            while not release.is_set():
                try:
                    time.sleep(0.005)
                except KeyboardInterrupt:
                    continue

        owner_task = asyncio.create_task(
            run_on_owner_executor(session, ignores_interrupt_until_released)
        )
        assert await asyncio.to_thread(started.wait, 1.0)

        assert await interrupt_owner_call(session, timeout=0.03) is False
        assert session.env_executor_poisoned is True

        release.set()
        await asyncio.wait_for(owner_task, timeout=1.0)
        executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(scenario())


def test_reset_refreshes_reset_created_viser_port() -> None:
    class Server:
        def get_port(self) -> int:
            return 8093

    class LowLevel:
        viser_server: Server | None = None

    class Env:
        low_level_env = LowLevel()

        def reset(self) -> tuple[dict[str, str], dict[str, str]]:
            self.low_level_env.viser_server = Server()
            return {"state": "ready"}, {"reset": "done"}

        def render(self) -> str:
            return "frame"

    env = Env()
    assert get_viser_port(env) is None

    obs, info, frame, port = reset_render_and_viser_port(env)

    assert obs == {"state": "ready"}
    assert info == {"reset": "done"}
    assert frame == "frame"
    assert port == 8093


def test_viser_head_fetch_does_not_read_application_bundle(monkeypatch: Any) -> None:
    from capx.web import visualization

    requests: list[Any] = []

    class Response:
        status = 200
        headers = {
            "Content-Type": "text/html",
            "Content-Length": "2900000",
        }
        closed = False
        read_called = False

        def read(self) -> bytes:
            self.read_called = True
            raise AssertionError("HEAD proxy must not read the Viser bundle")

        def close(self) -> None:
            self.closed = True

    response = Response()

    def fake_urlopen(request: Any, timeout: float) -> Response:
        assert timeout == 5
        requests.append(request)
        return response

    monkeypatch.setattr(visualization, "urlopen", fake_urlopen)

    status, content, content_type, content_length = fetch_viser_http(
        "http://localhost:8093/", include_body=False
    )

    assert status == 200
    assert content == b""
    assert content_type == "text/html"
    assert content_length == "2900000"
    assert requests[0].get_method() == "GET"
    assert response.read_called is False
    assert response.closed is True


def test_active_session_without_viser_port_never_uses_legacy_fallback(
    monkeypatch: Any,
) -> None:
    from capx.web import server

    class ActiveSession:
        viser_port = None

    class Manager:
        def get_active_session(self) -> ActiveSession:
            return ActiveSession()

    monkeypatch.setattr(server, "get_session_manager", lambda: Manager())

    def forbidden_fallback(preferred: int | None = None) -> int | None:
        pytest.fail(f"active-session isolation violated: probed {preferred=}")

    monkeypatch.setattr(server, "_find_viser_port", forbidden_fallback)

    assert server._active_viser_port() is None


def test_session_owned_viser_port_never_falls_back(monkeypatch: Any) -> None:
    from capx.web import visualization

    attempted: list[int] = []

    def refuse_request(request: Any, timeout: float) -> None:
        assert timeout == 0.5
        assert request.get_method() == "GET"
        assert request.get_header("Range") == "bytes=0-0"
        port = urlparse(request.full_url).port
        assert port is not None
        attempted.append(port)
        raise OSError("closed")

    monkeypatch.setattr(visualization, "urlopen", refuse_request)

    assert probe_viser_port(preferred=8097, cached=8081) is None
    assert attempted == [8097]

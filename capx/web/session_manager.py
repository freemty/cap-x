"""Session manager for tracking active trial sessions."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import threading
import uuid
from concurrent.futures import Executor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from fastapi import WebSocket

from capx.web.models import SessionState, WSEventBase

if TYPE_CHECKING:
    from capx.web.async_trial_runner import TrialContext

logger = logging.getLogger(__name__)

OWNER_INTERRUPT_TIMEOUT_SECONDS = 2.0
OWNER_CLOSE_TIMEOUT_SECONDS = 5.0


class OwnerThreadInterrupted(RuntimeError):
    """Normal asyncio-facing form of an injected ``KeyboardInterrupt``."""


async def run_on_owner_executor(
    session: "Session",
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run one environment operation on its single owner executor.

    The executor future is shielded and retained on the session.  Cancelling the
    coroutine therefore cannot lose a still-running constructor/step call, and
    cleanup can interrupt and wait for the exact call before closing the env.
    """

    executor = session.env_executor
    if executor is None:
        raise RuntimeError("Environment owner executor is not available")
    if session.env_executor_poisoned:
        raise RuntimeError("Environment owner executor is poisoned")

    loop = asyncio.get_running_loop()

    def wrapper() -> Any:
        thread_id = threading.get_ident()
        try:
            try:
                session.execution_thread_id = thread_id
                return func(*args, **kwargs)
            finally:
                if session.execution_thread_id == thread_id:
                    session.execution_thread_id = None
        except KeyboardInterrupt as exc:
            # Never let BaseException escape an executor Future into asyncio's
            # event loop.  Cleanup treats this normal exception as a released
            # owner thread.
            raise OwnerThreadInterrupted("Environment owner call interrupted") from exc

    future = loop.run_in_executor(executor, wrapper)
    session.owner_call_future = future
    try:
        return await asyncio.shield(future)
    finally:
        if future.done() and session.owner_call_future is future:
            session.owner_call_future = None


def request_owner_interrupt(session: "Session") -> bool:
    """Inject an interrupt into the current owner call, if it has started."""

    future = session.owner_call_future
    thread_id = session.execution_thread_id
    if future is None or future.done() or thread_id is None:
        return False
    logger.info("Interrupting environment owner thread %s", thread_id)
    sent = _raise_exception_in_thread(thread_id, KeyboardInterrupt)
    if not sent:
        logger.warning("Environment owner-thread interrupt failed")
    return sent


async def interrupt_owner_call(
    session: "Session",
    *,
    timeout: float = OWNER_INTERRUPT_TIMEOUT_SECONDS,
) -> bool:
    """Interrupt and wait for the retained owner call to release its worker.

    Returns ``True`` only when the call is no longer occupying the executor.
    A call stuck in native code may not observe ``PyThreadState_SetAsyncExc``;
    in that case the executor is explicitly poisoned so no reset/close is ever
    queued behind it.
    """

    future = session.owner_call_future
    if future is None:
        return True

    def consume_done_future() -> None:
        try:
            future.result()
        except BaseException:
            pass
        if session.owner_call_future is future:
            session.owner_call_future = None

    if future.done():
        consume_done_future()
        return True

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(timeout, 0.0)
    interrupt_sent = False

    while not future.done():
        if not interrupt_sent and session.execution_thread_id is not None:
            interrupt_sent = request_owner_interrupt(session)

        remaining = deadline - loop.time()
        if remaining <= 0:
            session.env_executor_poisoned = True
            logger.error(
                "Environment owner executor for session %s did not release; marking it poisoned",
                session.session_id,
            )
            return False

        try:
            await asyncio.wait_for(
                asyncio.shield(future),
                timeout=min(0.05, remaining),
            )
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            if not future.cancelled():
                raise
        except BaseException:
            # The owner operation failed or observed the injected interrupt;
            # either way the single worker has been released.
            pass

    consume_done_future()
    return True


async def run_blocking_with_interrupt(
    session: "Session",
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Backward-compatible wrapper around the environment owner executor."""

    return await run_on_owner_executor(session, func, *args, **kwargs)


def _raise_exception_in_thread(thread_id: int, exception_type: type) -> bool:
    """Raise an exception in another thread.

    This is a safety mechanism to interrupt code execution.
    Returns True if successful, False otherwise.
    """
    try:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id),
            ctypes.py_object(exception_type)
        )
        if res == 0:
            logger.warning(f"Thread {thread_id} not found")
            return False
        elif res > 1:
            # If more than one thread was affected, reset
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
            logger.error(f"Multiple threads affected when interrupting {thread_id}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to raise exception in thread: {e}")
        return False


@dataclass
class Session:
    """Represents an active trial session."""

    session_id: str
    state: SessionState = SessionState.IDLE
    config_path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    env_factory: dict[str, Any] | None = None

    # Settings that can be changed during a trial
    await_user_input_each_turn: bool = False
    execution_timeout: int = 180  # seconds per code block

    # Event history for replay on reconnect
    event_history: list[str] = field(default_factory=list)

    # Async coordination
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    user_injection_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)

    # Running task reference
    task: asyncio.Task | None = None

    # Environment reference for forced shutdown
    env: Any = None
    env_executor: Executor | None = None
    viser_port: int | None = None

    # Thread tracking for interruption
    execution_thread_id: int | None = None
    owner_call_future: asyncio.Future[Any] | None = None
    env_executor_poisoned: bool = False

    # Connected WebSocket clients
    websockets: list[WebSocket] = field(default_factory=list)

    # Execution state
    current_block_index: int = 0
    total_code_blocks: int = 0
    num_regenerations: int = 0

    # Timestamps
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None

    async def emit(self, event: WSEventBase) -> None:
        """Broadcast event to all connected WebSocket clients and store for replay."""
        message = event.model_dump_json()
        # Store for replay on reconnect (skip high-frequency streaming deltas)
        if event.type != "model_streaming_delta":
            self.event_history.append(message)
        disconnected = []

        for ws in self.websockets:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            self.websockets.remove(ws)

    def reset(self) -> None:
        """Reset session state for a new trial."""
        self.state = SessionState.IDLE
        self.cancel_event = asyncio.Event()
        self.user_injection_queue = asyncio.Queue()
        self.event_history = []
        self.task = None
        self.env = None
        self.env_executor = None
        self.viser_port = None
        self.execution_thread_id = None
        self.owner_call_future = None
        self.env_executor_poisoned = False
        self.current_block_index = 0
        self.total_code_blocks = 0
        self.num_regenerations = 0
        self.started_at = None
        self.completed_at = None


class SessionManager:
    """Manages all active trial sessions.

    Only one session can be active at a time. Creating a new session
    will automatically stop and clean up any existing sessions.
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> Session:
        """Create a new session, stopping any existing sessions first."""
        async with self._lock:
            # Stop and clean up ALL existing sessions first
            for session_id in list(self._sessions.keys()):
                await self._cleanup_session_unlocked(session_id)

            session_id = str(uuid.uuid4())
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            logger.info(f"Created session: {session_id}")
            return session

    async def _cleanup_session_unlocked(self, session_id: str) -> None:
        """Clean up a session (must be called with lock held)."""
        if session_id not in self._sessions:
            return

        session = self._sessions[session_id]
        logger.info(f"Cleaning up session: {session_id}")

        # Cancel any running task
        if session.task and not session.task.done():
            session.cancel_event.set()
            # Inject before cancellation so the retained executor call has a
            # chance to unwind rather than being hidden by task cancellation.
            request_owner_interrupt(session)
            session.task.cancel()
            try:
                await asyncio.wait_for(session.task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        await self._close_environment(session)

        # Close all WebSocket connections
        for ws in session.websockets:
            try:
                await ws.close(code=4001, reason="Session replaced")
            except Exception:
                pass

        del self._sessions[session_id]
        logger.info(f"Session cleaned up: {session_id}")

    async def _close_environment(self, session: Session) -> None:
        """Close a simulator on the thread that owns its rendering context."""

        executor = session.env_executor

        # Initialization or step may still own the only worker even after its
        # awaiting task was cancelled.  Never enqueue close behind that call.
        released = await interrupt_owner_call(session)
        env = session.env  # Initialization may have registered it while waiting.

        def close() -> None:
            if env is None:
                return
            if hasattr(env, "close"):
                env.close()
            elif hasattr(env, "shutdown"):
                env.shutdown()

        close_task: asyncio.Task[Any] | None = None
        try:
            if env is not None and executor is not None and released:
                close_task = asyncio.create_task(run_on_owner_executor(session, close))
                try:
                    await asyncio.wait_for(
                        asyncio.shield(close_task),
                        timeout=OWNER_CLOSE_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timed out while closing environment for session %s",
                        session.session_id,
                    )
                    released = await interrupt_owner_call(session)
                    if not released:
                        session.env_executor_poisoned = True
                finally:
                    if close_task.done():
                        await asyncio.gather(close_task, return_exceptions=True)
                    elif not released:
                        close_task.cancel()
                        await asyncio.gather(close_task, return_exceptions=True)
            elif env is not None and executor is None:
                await asyncio.wait_for(
                    asyncio.to_thread(close), timeout=OWNER_CLOSE_TIMEOUT_SECONDS
                )
            elif env is not None and not released:
                logger.error(
                    "Skipping environment close for poisoned session %s; owner worker is still occupied",
                    session.session_id,
                )
        except asyncio.TimeoutError:
            logger.warning("Timed out while closing environment for session %s", session.session_id)
        except Exception as exc:
            logger.warning("Error closing environment for session %s: %s", session.session_id, exc)
        finally:
            session.env = None
            session.env_executor = None
            session.viser_port = None
            session.execution_thread_id = None
            session.owner_call_future = None
            if executor is not None:
                if session.env_executor_poisoned:
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    # The owner call and close have both completed, so this join
                    # is bounded and proves the worker was reclaimed.
                    await asyncio.to_thread(
                        executor.shutdown, wait=True, cancel_futures=True
                    )

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        async with self._lock:
            await self._cleanup_session_unlocked(session_id)

    async def stop_session(self, session_id: str) -> bool:
        """Stop a running session immediately.

        This is a safety-critical operation that should interrupt code execution
        as quickly as possible.
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        if session.task and not session.task.done():
            logger.info(f"STOPPING session (safety interrupt): {session_id}")
            session.cancel_event.set()

            # Interrupt before cancelling the asyncio task; the executor future
            # remains retained for _close_environment to await safely.
            request_owner_interrupt(session)

            # Cancel the task immediately (don't wait for graceful shutdown)
            session.task.cancel()
            try:
                await asyncio.wait_for(session.task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        had_environment = session.env is not None or session.env_executor is not None
        if had_environment:
            await self._close_environment(session)

        if session.task is not None or had_environment:
            session.state = SessionState.IDLE
            logger.info(f"Session {session_id} stopped")
            return True

        return False

    async def inject_prompt(self, session_id: str, text: str) -> bool:
        """Inject user prompt text into a session."""
        session = await self.get_session(session_id)
        if not session:
            return False

        if session.state == SessionState.AWAITING_USER_INPUT:
            await session.user_injection_queue.put(text)
            logger.info(f"Injected prompt into session {session_id}: {text[:50]}...")
            return True

        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with their status."""
        return [
            {
                "session_id": s.session_id,
                "state": s.state.value,
                "config_path": s.config_path,
                "created_at": s.created_at.isoformat(),
            }
            for s in self._sessions.values()
        ]

    def get_active_session(self) -> Session | None:
        """Get the currently active session (if any).

        Since only one session is allowed at a time, this returns
        the single session if it exists and is still running.
        """
        for session in self._sessions.values():
            if session.state in (
                SessionState.RUNNING,
                SessionState.AWAITING_USER_INPUT,
                SessionState.LOADING_CONFIG,
                SessionState.COMPLETE,
            ):
                return session
        return None

    async def on_websocket_disconnect(self, session_id: str) -> None:
        """Handle WebSocket disconnection.

        If the session has no more connected WebSockets and is still running,
        we'll keep it alive briefly in case of reconnection. If it's complete
        or errored, clean it up.
        """
        session = await self.get_session(session_id)
        if not session:
            return

        # If session is complete/error and no WebSockets, clean up
        if session.state in (SessionState.COMPLETE, SessionState.ERROR, SessionState.IDLE):
            if not session.websockets:
                logger.info(f"Session {session_id} has no connections and is {session.state}, cleaning up")
                await self.remove_session(session_id)


# Global singleton
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager

"""CaP-X Interactive Web UI Backend.

This module provides a FastAPI-based web server with WebSocket support
for real-time interactive robot code execution demos.
"""

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Import the optional web stack only when an app is requested."""

    from capx.web.server import create_app as _create_app

    return _create_app(*args, **kwargs)

__all__ = ["create_app"]

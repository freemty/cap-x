"""Relay CaP-X Chat Completions requests to a network-reachable upstream.

The simulator hosts used by CaP-X may not have outbound access to model
providers. This small relay keeps the provider credential on a reachable host
and is exposed to the simulator through an SSH reverse tunnel.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


class ChatCompletionRequest(BaseModel):
    """Chat Completions request with provider-specific fields preserved."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False


def create_app(
    *,
    upstream_api_key: str,
    upstream_url: str,
    timeout_seconds: float = 300.0,
) -> FastAPI:
    """Create a non-streaming authenticated Chat Completions relay."""

    app = FastAPI(title="CaP-X Chat Completions Relay", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported by this relay")

        headers = {
            "Authorization": f"Bearer {upstream_api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                upstream_url,
                headers=headers,
                json=request.model_dump(exclude_none=True),
                timeout=timeout_seconds,
            )
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

        if not response.ok:
            raise HTTPException(status_code=response.status_code, detail=response.text[:2000])
        try:
            return response.json()
        except requests.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail="Upstream returned invalid JSON") from exc

    return app


def main(
    upstream_url: str,
    host: str = "127.0.0.1",
    port: int = 18110,
    api_key_env: str = "GLM_API_KEY",
    api_key_file: Path | None = None,
    timeout_seconds: float = 300.0,
) -> None:
    """Run the relay without placing credentials in command-line arguments."""

    api_key = os.environ.get(api_key_env)
    if not api_key and api_key_file is not None:
        api_key = api_key_file.read_text().strip()
    if not api_key:
        raise RuntimeError(
            f"Set environment variable {api_key_env!r} or provide --api-key-file"
        )
    app = create_app(
        upstream_api_key=api_key,
        upstream_url=upstream_url,
        timeout_seconds=timeout_seconds,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)

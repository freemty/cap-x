"""Expose an OpenAI-compatible chat endpoint backed by the Responses API.

CaP-X currently sends Chat Completions-shaped requests.  This proxy keeps that
stable local contract while allowing a Responses-only upstream to be used for
models such as GPT-5.5.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import requests
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


class ChatCompletionRequest(BaseModel):
    """Subset of Chat Completions fields used by CaP-X."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    stream: bool = False


def _responses_content(content: Any) -> Any:
    if isinstance(content, str) or content is None:
        return content
    converted: list[dict[str, Any]] = []
    for item in content:
        item = dict(item)
        if item.get("type") == "text":
            converted.append({"type": "input_text", "text": item.get("text", "")})
        elif item.get("type") == "image_url":
            image = item.get("image_url", {})
            url = image.get("url") if isinstance(image, dict) else image
            converted.append({"type": "input_image", "image_url": url})
        else:
            converted.append(item)
    return converted


def _to_responses_payload(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise ValueError("Streaming is not supported by this compatibility proxy")

    input_messages = []
    for message in request.messages:
        role = "developer" if message.get("role") == "system" else message.get("role")
        input_messages.append(
            {
                "role": role,
                "content": _responses_content(message.get("content")),
            }
        )

    payload: dict[str, Any] = {
        "model": request.model,
        "input": input_messages,
        "store": False,
    }
    max_output_tokens = request.max_completion_tokens or request.max_tokens
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if request.reasoning_effort is not None:
        payload["reasoning"] = {"effort": request.reasoning_effort}
    return payload


def _extract_output_text(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str):
        return output_text

    chunks: list[str] = []
    for output in body.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if not chunks:
        raise ValueError("Responses payload did not contain output text")
    return "".join(chunks)


def create_app(
    *,
    upstream_api_key: str,
    upstream_url: str,
    auth_header: str = "Authorization",
    auth_scheme: str = "Bearer",
    timeout_seconds: float = 300.0,
) -> FastAPI:
    app = FastAPI(title="CaP-X Responses Compatibility Proxy", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        try:
            payload = _to_responses_payload(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        credential = f"{auth_scheme} {upstream_api_key}".strip()
        headers = {"Content-Type": "application/json", auth_header: credential}
        try:
            response = requests.post(
                upstream_url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

        if not response.ok:
            detail = response.text[:2000]
            raise HTTPException(status_code=response.status_code, detail=detail)

        body = response.json()
        try:
            content = _extract_output_text(body)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return {
            "id": body.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion",
            "created": body.get("created_at", int(time.time())),
            "model": body.get("model", request.model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": body.get("usage"),
        }

    return app


def main(
    upstream_url: str,
    host: str = "127.0.0.1",
    port: int = 18110,
    api_key_env: str = "UPSTREAM_API_KEY",
    auth_header: str = "Authorization",
    auth_scheme: str = "Bearer",
    timeout_seconds: float = 300.0,
) -> None:
    """Run the compatibility proxy without placing credentials in CLI args."""

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Required environment variable {api_key_env!r} is not set")
    app = create_app(
        upstream_api_key=api_key,
        upstream_url=upstream_url,
        auth_header=auth_header,
        auth_scheme=auth_scheme,
        timeout_seconds=timeout_seconds,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)

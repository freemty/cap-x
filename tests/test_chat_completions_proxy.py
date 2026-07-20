from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from capx.serving.chat_completions_proxy import create_app


class _Response:
    ok = True

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}]}


def test_proxy_injects_credential_and_preserves_glm_fields(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return _Response()

    monkeypatch.setattr("capx.serving.chat_completions_proxy.requests.post", fake_post)
    app = create_app(
        upstream_api_key="test-only-key",
        upstream_url="https://example.test/chat/completions",
    )

    response = TestClient(app).post(
        "/chat/completions",
        json={
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "ping"}],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
        },
    )

    assert response.status_code == 200
    assert captured["headers"]["Authorization"] == "Bearer test-only-key"
    assert captured["payload"]["thinking"] == {"type": "enabled"}
    assert captured["payload"]["reasoning_effort"] == "max"

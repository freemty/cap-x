from __future__ import annotations

import json
from typing import Any

from capx.llm.client import GLM_MODELS, ModelQueryArgs, query_model


class _Response:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": "```python\nRESULT = True\n```",
                        "reasoning_content": "planned",
                    }
                }
            ]
        }


def test_glm_request_uses_provider_contract(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url, *, headers, data, timeout):
        captured.update(url=url, headers=headers, payload=json.loads(data), timeout=timeout)
        return _Response()

    monkeypatch.setenv("GLM_API_KEY", "test-only-key")
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    args = ModelQueryArgs(
        model="glm-5.2",
        server_url="https://example.test/chat/completions",
        temperature=1.0,
        max_tokens=4096,
        reasoning_effort="max",
    )

    result = query_model(args, [{"role": "user", "content": "write robot code"}])

    assert "glm-5.2" in GLM_MODELS
    assert captured["headers"]["Authorization"] == "Bearer test-only-key"
    assert captured["payload"] == {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "write robot code"}],
        "temperature": 1.0,
        "max_tokens": 4096,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    assert result == {
        "content": "```python\nRESULT = True\n```",
        "reasoning": "planned",
    }

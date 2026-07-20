from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from capx.serving.llm_api_smoke import (
    MARKER,
    PROVIDERS,
    _load_credential,
    _payload,
    _text_from_anthropic,
    _text_from_chat,
    _text_from_responses,
    run_provider,
)


class _Response:
    status = 200

    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self._body


class LlmApiSmokeTests(unittest.TestCase):
    def test_provider_payloads_cover_all_supported_formats(self) -> None:
        glm_chat = _payload(PROVIDERS["glm_coding_chat"], "glm-5.2")
        self.assertEqual(glm_chat["thinking"], {"type": "enabled"})
        self.assertEqual(glm_chat["reasoning_effort"], "max")
        self.assertIn(MARKER, glm_chat["messages"][0]["content"])

        responses = _payload(PROVIDERS["openai_responses"], "gpt-test")
        self.assertFalse(responses["store"])
        self.assertIn(MARKER, responses["input"])

        anthropic = _payload(PROVIDERS["glm_coding_anthropic"], "glm-5.2")
        self.assertEqual(anthropic["max_tokens"], 256)

    def test_response_extractors(self) -> None:
        self.assertEqual(
            _text_from_chat({"choices": [{"message": {"content": MARKER}}]}), MARKER
        )
        self.assertEqual(_text_from_responses({"output_text": MARKER}), MARKER)
        self.assertEqual(
            _text_from_responses(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": MARKER}],
                        }
                    ]
                }
            ),
            MARKER,
        )
        self.assertEqual(
            _text_from_anthropic({"content": [{"type": "text", "text": MARKER}]}),
            MARKER,
        )

    def test_credential_loader_prefers_environment_and_warns_on_open_file(self) -> None:
        spec = PROVIDERS["glm_coding_chat"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / ".glmkey"
            path.write_text("file-key\n")
            path.chmod(0o644)
            with patch.dict(os.environ, {"GLM_API_KEY": "env-key"}, clear=True):
                key, source, warning = _load_credential(spec, root)
                self.assertEqual((key, source, warning), ("env-key", "env:GLM_API_KEY", None))
            with patch.dict(os.environ, {}, clear=True):
                key, source, warning = _load_credential(spec, root)
                self.assertEqual(key, "file-key")
                self.assertEqual(source, "file:.glmkey")
                self.assertIn("0600", warning or "")

    def test_live_probe_validates_marker_without_exposing_key(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured["authorization"] = request.headers.get("Authorization")
            captured["timeout"] = timeout
            return _Response({"choices": [{"message": {"content": MARKER}}]})

        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"GLM_API_KEY": "secret-test-key"}, clear=True):
                result = run_provider(
                    PROVIDERS["glm_coding_chat"],
                    root=Path(temporary),
                    timeout_seconds=3,
                    opener=opener,
                )
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.response_chars, len(MARKER))
        self.assertEqual(captured["authorization"], "Bearer secret-test-key")
        self.assertNotIn("secret-test-key", json.dumps(result.__dict__))

    def test_http_error_reports_code_not_response_message(self) -> None:
        # HTTPError.read() needs a file-like body, so use a small subclass for this fixture.
        class ErrorWithBody(urllib.error.HTTPError):
            def read(self, _limit):
                return json.dumps(
                    {"error": {"code": "invalid_api_key", "message": "secret server detail"}}
                ).encode()

        def opener_with_body(request, *, timeout):
            del request, timeout
            raise ErrorWithBody("https://example.test", 401, "no", {}, None)

        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"GLM_API_KEY": "secret-test-key"}, clear=True):
                result = run_provider(
                    PROVIDERS["glm_coding_chat"],
                    root=Path(temporary),
                    timeout_seconds=3,
                    opener=opener_with_body,
                )
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.error, "HTTP 401 (invalid_api_key)")
        self.assertNotIn("secret", json.dumps(result.__dict__))


if __name__ == "__main__":
    unittest.main()

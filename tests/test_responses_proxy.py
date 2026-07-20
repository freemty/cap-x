from __future__ import annotations

from capx.serving.responses_proxy import (
    ChatCompletionRequest,
    _extract_output_text,
    _to_responses_payload,
)
from capx.llm.client import GPT_MODELS


def test_chat_request_converts_to_responses_payload() -> None:
    assert "gpt-5.5" in GPT_MODELS

    request = ChatCompletionRequest(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": "Generate robot code."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Stack the blocks."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ],
        max_completion_tokens=4096,
        reasoning_effort="medium",
    )

    payload = _to_responses_payload(request)

    assert payload == {
        "model": "gpt-5.5",
        "input": [
            {"role": "developer", "content": "Generate robot code."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Stack the blocks."},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                ],
            },
        ],
        "store": False,
        "max_output_tokens": 4096,
        "reasoning": {"effort": "medium"},
    }


def test_extract_output_text_from_responses_items() -> None:
    body = {
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "```python\n"},
                    {"type": "output_text", "text": "RESULT = True\n```"},
                ],
            },
        ]
    }

    assert _extract_output_text(body) == "```python\nRESULT = True\n```"

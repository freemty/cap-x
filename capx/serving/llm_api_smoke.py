"""Safe, reusable live smoke tests for CaP-X language-model providers.

The runner never prints credentials or full model responses. Providers without
configured credentials are skipped by default, while every configured provider
must return the requested marker for the command to succeed.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER = "CAPX_API_OK"
DEFAULT_OUTPUT = Path("outputs/api-smoke/llm-api-smoke-latest.json")


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    api_format: str
    endpoint: str
    endpoint_env: str
    model: str | None
    model_env: str
    key_envs: tuple[str, ...] = ()
    key_files: tuple[str, ...] = ()
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    credential_required: bool = True


@dataclass
class SmokeResult:
    provider: str
    status: str
    api_format: str
    endpoint: str
    model: str | None
    credential_source: str | None = None
    credential_warning: str | None = None
    http_status: int | None = None
    latency_ms: int | None = None
    response_chars: int = 0
    validation: str | None = None
    error: str | None = None


PROVIDERS: dict[str, ProviderSpec] = {
    "glm_coding_chat": ProviderSpec(
        name="glm_coding_chat",
        api_format="chat_completions",
        endpoint="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
        endpoint_env="GLM_CODING_CHAT_URL",
        model="glm-5.2",
        model_env="GLM_SMOKE_MODEL",
        key_envs=("GLM_CODING_API_KEY", "GLM_API_KEY"),
        key_files=(".glmkey",),
    ),
    "glm_coding_anthropic": ProviderSpec(
        name="glm_coding_anthropic",
        api_format="anthropic_messages",
        endpoint="https://open.bigmodel.cn/api/anthropic/v1/messages",
        endpoint_env="GLM_ANTHROPIC_URL",
        model="glm-5.2",
        model_env="GLM_SMOKE_MODEL",
        key_envs=("GLM_CODING_API_KEY", "GLM_API_KEY"),
        key_files=(".glmkey",),
        auth_header="x-api-key",
        auth_scheme="",
    ),
    "glm_standard_chat": ProviderSpec(
        name="glm_standard_chat",
        api_format="chat_completions",
        endpoint="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        endpoint_env="BIGMODEL_CHAT_URL",
        model="glm-5.2",
        model_env="BIGMODEL_SMOKE_MODEL",
        key_envs=("BIGMODEL_API_KEY",),
        key_files=(".bigmodelkey",),
    ),
    "openai_responses": ProviderSpec(
        name="openai_responses",
        api_format="responses",
        endpoint="https://api.openai.com/v1/responses",
        endpoint_env="OPENAI_RESPONSES_URL",
        model="gpt-5.5",
        model_env="OPENAI_SMOKE_MODEL",
        key_envs=("OPENAI_API_KEY", "UPSTREAM_API_KEY"),
        key_files=(".openaikey",),
    ),
    "openrouter_chat": ProviderSpec(
        name="openrouter_chat",
        api_format="chat_completions",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        endpoint_env="OPENROUTER_CHAT_URL",
        model="google/gemini-2.5-flash-lite",
        model_env="OPENROUTER_SMOKE_MODEL",
        key_envs=("OPENROUTER_API_KEY",),
        key_files=(".openrouterkey",),
    ),
    "capx_local_chat": ProviderSpec(
        name="capx_local_chat",
        api_format="chat_completions",
        endpoint="http://127.0.0.1:8110/chat/completions",
        endpoint_env="CAPX_LLM_URL",
        model=None,
        model_env="CAPX_LLM_MODEL",
        credential_required=False,
    ),
}


def _load_credential(
    spec: ProviderSpec, root: Path
) -> tuple[str | None, str | None, str | None]:
    for env_name in spec.key_envs:
        value = os.environ.get(env_name)
        if value:
            return value.strip(), f"env:{env_name}", None

    for relative_path in spec.key_files:
        path = root / relative_path
        if not path.is_file():
            continue
        value = path.read_text().strip()
        if not value:
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        warning = None
        if mode & 0o077:
            warning = f"credential file permissions are {mode:04o}; use 0600"
        return value, f"file:{relative_path}", warning
    return None, None, None


def _payload(spec: ProviderSpec, model: str) -> dict[str, Any]:
    prompt = f"Reply with exactly {MARKER} and nothing else."
    if spec.api_format == "chat_completions":
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
        }
        if spec.name.startswith("glm_"):
            body.update(
                {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "max",
                }
            )
        return body
    if spec.api_format == "responses":
        return {
            "model": model,
            "input": prompt,
            "store": False,
            "max_output_tokens": 512,
            "reasoning": {"effort": "low"},
        }
    if spec.api_format == "anthropic_messages":
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0,
        }
    raise ValueError(f"Unsupported API format: {spec.api_format}")


def _text_from_chat(body: dict[str, Any]) -> str:
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    raise TypeError("Chat Completions content is neither text nor a content list")


def _text_from_responses(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    chunks: list[str] = []
    for output in body.get("output", []):
        if not isinstance(output, dict) or output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks)


def _text_from_anthropic(body: dict[str, Any]) -> str:
    return "".join(
        item.get("text", "")
        for item in body.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    )


EXTRACTORS: dict[str, Callable[[dict[str, Any]], str]] = {
    "chat_completions": _text_from_chat,
    "responses": _text_from_responses,
    "anthropic_messages": _text_from_anthropic,
}


def _http_error_code(body: bytes) -> str | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        return str(code)[:100] if code else None
    return None


def run_provider(
    spec: ProviderSpec,
    *,
    root: Path,
    timeout_seconds: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> SmokeResult:
    endpoint = os.environ.get(spec.endpoint_env, spec.endpoint)
    model = os.environ.get(spec.model_env, spec.model or "").strip() or None
    result = SmokeResult(
        provider=spec.name,
        status="skip",
        api_format=spec.api_format,
        endpoint=endpoint,
        model=model,
    )
    if model is None:
        result.error = f"set {spec.model_env} to enable this probe"
        return result

    credential, source, warning = _load_credential(spec, root)
    result.credential_source = source
    result.credential_warning = warning
    if spec.credential_required and not credential:
        choices = [*spec.key_envs, *spec.key_files]
        result.error = "credential not configured; expected one of: " + ", ".join(choices)
        return result

    headers = {"Content-Type": "application/json", "User-Agent": "CaP-X-API-Smoke/1.0"}
    if credential:
        value = f"{spec.auth_scheme} {credential}".strip()
        headers[spec.auth_header] = value
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(_payload(spec, model)).encode(),
        headers=headers,
        method="POST",
    )

    started = time.monotonic()
    try:
        with opener(request, timeout=timeout_seconds) as response:
            raw = response.read(4 * 1024 * 1024)
            result.http_status = int(response.status)
    except urllib.error.HTTPError as exc:
        result.http_status = exc.code
        result.latency_ms = round((time.monotonic() - started) * 1000)
        try:
            code = _http_error_code(exc.read(64 * 1024))
        finally:
            exc.close()
        result.status = "fail"
        result.error = f"HTTP {exc.code}" + (f" ({code})" if code else "")
        return result
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result.latency_ms = round((time.monotonic() - started) * 1000)
        result.status = "fail"
        result.error = f"network error: {type(exc).__name__}"
        return result

    result.latency_ms = round((time.monotonic() - started) * 1000)
    try:
        body = json.loads(raw)
        text = EXTRACTORS[spec.api_format](body)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
        result.status = "fail"
        result.error = f"invalid {spec.api_format} response: {type(exc).__name__}"
        return result

    result.response_chars = len(text)
    if MARKER not in text:
        result.status = "fail"
        result.validation = "nonempty response but expected marker was absent"
        return result
    result.status = "pass"
    result.validation = f"response contained {MARKER}"
    return result


def _print_results(results: list[SmokeResult]) -> None:
    print("Provider                    Status  HTTP  Latency  Model")
    print("--------------------------  ------  ----  -------  ------------------------------")
    for row in results:
        http_status = str(row.http_status) if row.http_status is not None else "-"
        latency = f"{row.latency_ms}ms" if row.latency_ms is not None else "-"
        print(
            f"{row.provider:<26}  {row.status.upper():<6}  {http_status:<4}  "
            f"{latency:<7}  {row.model or '-'}"
        )
        if row.error:
            print(f"  {row.error}")
        if row.credential_warning:
            print(f"  WARNING: {row.credential_warning}")


def _write_report(path: Path, results: list[SmokeResult]) -> None:
    report = {
        "schema": "capx.llm-api-smoke.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "summary": {
            "pass": sum(row.status == "pass" for row in results),
            "fail": sum(row.status == "fail" for row in results),
            "skip": sum(row.status == "skip" for row in results),
        },
        "results": [asdict(row) for row in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        action="append",
        choices=sorted(PROVIDERS),
        help="Provider to test; repeat as needed. The default tests every profile.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero when a provider is skipped as well as when it fails.",
    )
    parser.add_argument("--list", action="store_true", help="List provider profiles and exit.")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    names = args.provider or list(PROVIDERS)
    if args.list:
        for name in names:
            spec = PROVIDERS[name]
            _, source, _ = _load_credential(spec, root)
            model = os.environ.get(spec.model_env, spec.model or "").strip() or "not configured"
            print(f"{name}: {spec.api_format}, model={model}, credential={source or 'not configured'}")
        return 0

    results = [
        run_provider(PROVIDERS[name], root=root, timeout_seconds=args.timeout) for name in names
    ]
    _print_results(results)
    _write_report(root / args.output if not args.output.is_absolute() else args.output, results)
    print(f"\nMachine-readable report: {args.output}")

    failures = sum(row.status == "fail" for row in results)
    skipped = sum(row.status == "skip" for row in results)
    if failures or (args.strict and skipped):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

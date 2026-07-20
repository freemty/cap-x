#!/usr/bin/env bash
# Run offline contract tests, then live-smoke every configured LLM provider.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$REPO_ROOT"

echo "== Offline LLM API contract tests =="
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_llm_api_smoke.py'

echo
echo "== Live LLM API smoke tests =="
exec "$PYTHON_BIN" -m capx.serving.llm_api_smoke "$@"

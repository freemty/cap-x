# LLM API smoke-test guide

Use this guide to distinguish a reachable provider endpoint from a genuinely
usable language-model API before launching a CaP-X simulator experiment.

## Current verified state

"The API works" has two different meanings:

1. **Contract or route verified** means CaP-X built the expected payload, parsed
   a fixture response, or reached the provider authentication boundary.
2. **Live generation verified** means a real credential returned non-empty model
   output containing the deterministic `CAPX_API_OK` marker.

As of 2026-07-20, the repository has no retained evidence of a successful live
generation from an external LLM provider. The following lower layers have been
verified:

| Profile | Contract | Network/route | Authenticated generation |
|---|---|---|---|
| `glm_coding_chat` | Pass | Local provider endpoint reached `401`; local relay and SSH reverse tunnel pass | Awaiting a rotated Coding Plan key |
| `glm_coding_anthropic` | Pass | Local provider endpoint reached `401` | Awaiting a rotated Coding Plan key |
| `glm_standard_chat` | Pass | Local provider endpoint reached `401` | Awaiting a regular BigModel API key |
| `openai_responses` | Pass | Compatibility proxy passes; local upstream endpoint reached `401` | No successful upstream model response retained |
| `openrouter_chat` | Pass through the shared Chat Completions contract | Local provider endpoint reached `401` | Not currently verified |
| `capx_local_chat` | Pass when a local proxy is running | Relay `/health` and remote tunnel were verified | Depends on the selected upstream |

The LIBERO and RoboTwin oracle results in `exp01b` are real simulator/API
executions, but they deliberately bypass external model generation and therefore
do not count as LLM API success.

## Prerequisites

- Python 3.10 or newer.
- Network access to the selected provider from the machine running the smoke
  test.
- A valid provider credential for live generation. Do not reuse a credential
  that has appeared in a screenshot, terminal transcript, or chat message.
- Credential files must be git-ignored and should have permission `0600`.

Environment variables take precedence over credential files:

| Profile | Credential | Optional model override |
|---|---|---|
| `glm_coding_chat`, `glm_coding_anthropic` | `GLM_CODING_API_KEY`, `GLM_API_KEY`, or `.glmkey` | `GLM_SMOKE_MODEL` |
| `glm_standard_chat` | `BIGMODEL_API_KEY` or `.bigmodelkey` | `BIGMODEL_SMOKE_MODEL` |
| `openai_responses` | `OPENAI_API_KEY`, `UPSTREAM_API_KEY`, or `.openaikey` | `OPENAI_SMOKE_MODEL` |
| `openrouter_chat` | `OPENROUTER_API_KEY` or `.openrouterkey` | `OPENROUTER_SMOKE_MODEL` |
| `capx_local_chat` | None | `CAPX_LLM_MODEL` (required) |

For example, after rotating the exposed GLM key, store the replacement locally
and restrict its permissions:

```bash
chmod 600 .glmkey
```

## Run all configured providers

From the repository root:

```bash
./scripts/test_llm_apis.sh
```

The command performs two stages:

1. Run dependency-light offline tests for request payloads, response parsing,
   credential precedence, error redaction, and marker validation.
2. Send one minimal marker prompt to every provider whose credential or local
   model is configured.

Missing credentials are reported as `SKIP`. A configured provider that cannot
return `CAPX_API_OK` is `FAIL` and makes the command return a nonzero exit code.

Use strict mode in CI when every profile is required:

```bash
./scripts/test_llm_apis.sh --strict
```

## Run selected providers

Repeat `--provider` to select one or more profiles:

```bash
./scripts/test_llm_apis.sh --provider glm_coding_chat

./scripts/test_llm_apis.sh \
  --provider glm_coding_chat \
  --provider glm_coding_anthropic
```

List the built-in profiles, resolved models, and credential sources without
making a network request:

```bash
python3 -m capx.serving.llm_api_smoke --list
```

## Validate the final CaP-X endpoint

After starting a provider proxy, test the same local Chat Completions endpoint
that the simulator will call:

```bash
CAPX_LLM_MODEL=glm-5.2 \
./scripts/test_llm_apis.sh --provider capx_local_chat
```

For exp00a, the simulator host cannot reach BigModel directly. Start the local
GLM relay and expose it with the reverse tunnel documented in
[`docs/configuration.md`](../configuration.md), then run the local-endpoint test
before starting the 180-task evaluation.

## Results

The command prints a compact table and writes the latest machine-readable report
to:

```text
outputs/api-smoke/llm-api-smoke-latest.json
```

The report uses schema `capx.llm-api-smoke.v1` and records provider, API format,
model, endpoint, credential source name, HTTP status, latency, response length,
validation result, and sanitized error category. It never stores credential
values or the complete model response.

Interpret statuses as follows:

- `PASS`: the provider returned a valid response containing `CAPX_API_OK`.
- `FAIL`: the provider was configured but authentication, networking, response
  parsing, or marker validation failed.
- `SKIP`: the required credential or local model was not configured. This is not
  evidence that the API works.

## Troubleshooting

### `HTTP 401`

The endpoint is reachable, but the credential is missing, expired, revoked, or
belongs to a different product. A GLM Coding Plan key and a regular BigModel API
key use separate smoke profiles and endpoints.

### `HTTP 403`

Check account permissions, subscription scope, and whether the calling tool is
allowed by the provider plan. A technically compatible endpoint does not expand
the subscription's permitted use.

### `HTTP 404` or model-not-found error

Override the smoke model without editing the runner:

```bash
OPENAI_SMOKE_MODEL=<available-model> ./scripts/test_llm_apis.sh --provider openai_responses
OPENROUTER_SMOKE_MODEL=<available-model> ./scripts/test_llm_apis.sh --provider openrouter_chat
```

### Network timeout on the simulator host

Do not copy provider credentials to the remote simulator solely to work around
network egress. Keep the credential on the reachable machine, start the local
relay, and use an SSH reverse tunnel to expose loopback port `8110` remotely.

### Credential-file permission warning

Restrict the reported file and rerun:

```bash
chmod 600 .glmkey .bigmodelkey .openaikey .openrouterkey 2>/dev/null || true
```

The runner reports insecure permissions but never prints file contents.

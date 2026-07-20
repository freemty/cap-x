# exp00a: GLM-5.2 single-episode baselines on LIBERO and RoboTwin

## Motivation

Measure the first real language-model baseline on both integrated simulators.
Every benchmark task receives one independent GLM-5.2 code-generation call and
one environment episode; repository oracle programs are disabled.

## Relation to Previous

This is the baseline experiment and has no parent. `exp01a` validated one
RoboTwin repository oracle outside CaP-X. `exp01b` validated one privileged
repository oracle per simulator through CaP-X. Neither measured model-generated
code.

## Settings

See `config.yaml`. The primary scope is 180 tasks:

- LIBERO standard benchmark: 130 tasks (`libero_spatial`, `libero_object`,
  `libero_goal`, `libero_10`, and `libero_90`).
- RoboTwin at revision `c3ddfa8`: 50 task modules.

The first pass uses privileged textual state and high-level APIs so both
simulators share a language-model/code-policy evaluation layer. It is not a
pure-RGB VLM evaluation. Each task uses trial 1, which CaP-X also passes as the
environment seed.

## Completion Criteria

- The task manifest contains exactly 130 LIBERO and 50 RoboTwin tasks.
- Each task has one ledger row from GLM-5.2 with oracle code disabled.
- Code-execution rate, task-success rate, infrastructure failures, and elapsed
  time are reported separately by benchmark and suite.

## Findings

(populated by `analyze.py`)

## Pitfalls

- The remote simulator host cannot reach the configured model provider
  directly. Run `capx.serving.chat_completions_proxy` on a reachable host and
  use an SSH reverse tunnel to expose it as remote `127.0.0.1:8110`.
- The Coding Plan endpoint in `config.yaml` must only be used when the account's
  subscription terms permit the calling tool. Otherwise use a regular BigModel
  API key and `https://open.bigmodel.cn/api/paas/v4/chat/completions`.
- Do not interpret this privileged-state baseline as visual generalization.
- RoboTwin's current generic API covers grasp, place, gripper, displacement,
  and home motions. Tasks requiring specialized contact motions may expose API
  coverage failures rather than model reasoning failures.

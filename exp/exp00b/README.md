# exp00b: GLM-5.2 on ten RoboTwin tasks with frontend ingestion

## Motivation

Run one real GLM-5.2/CaP-X episode on ten deterministic RoboTwin tasks, report
code-execution and task-success rates, retain videos and generated programs, and
verify that the normalized results appear correctly in the existing viewer.

## Relation to Previous

This is a ten-task RoboTwin subset of `exp00a`. `exp00a` defines the eventual
180-task LIBERO plus RoboTwin baseline; `exp00b` is the smaller integration gate
before paying the cost of that full run. `exp01b` validated the RoboTwin adapter
with repository oracle code, whereas this experiment disables oracle code and
uses one independent GLM-5.2 generation per task.

## Settings

See `config.yaml`. The fixed tasks are:

1. `adjust_bottle`
2. `beat_block_hammer`
3. `blocks_ranking_rgb`
4. `blocks_ranking_size`
5. `click_alarmclock`
6. `click_bell`
7. `dump_bin_bigbin`
8. `grab_roller`
9. `handover_block`
10. `handover_mic`

All runs use privileged textual state, the generic `RoboTwinPrivilegedApi`, the
`aloha-agilex` embodiment, seed/trial 1, one GLM-5.2 generation, and no oracle
program. GPU 6 is used because GPU 5 was already occupied when this subset was
launched. This is a code-policy baseline rather than a pure-RGB VLM evaluation.

## Frontend contract

`analyze.py` converts the append-only ledger and CaP-X output folders into
`results/manifest.json` using schema `capx.result.v1`. Every attempted task is
represented, including infrastructure failures. The viewer discovers this
manifest automatically and exposes videos, generated code, summaries, events,
and model responses through its artifact allow-list. The runner refreshes the
manifest after every durable ledger row, and the frontend polls every five
seconds, so completed tasks appear without restarting the viewer.

## Findings

(populated after the live run)

## Pitfalls

- A `401` reachability result is not a successful model call. Run only with a
  rotated, unexposed GLM credential.
- The simulator host cannot reach BigModel directly. Keep the credential on the
  reachable local machine and use the authenticated relay plus SSH reverse
  tunnel.
- Generic RoboTwin APIs may be insufficient for contact-heavy tasks; separate
  code execution, planning, task success, and infrastructure errors.

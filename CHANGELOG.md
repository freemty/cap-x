# Changelog

## Unreleased

### Added

- Added layered RoboTwin and LIBERO trajectory instrumentation, replay artifacts,
  Viser integration, and live Web UI status for planner and task success.
- Added OpenAI Responses and OpenAI-compatible chat proxies, reusable provider
  smoke tests, and native GLM-5.2 client support.
- Added `exp00a` and `exp00b` frozen code-policy evaluation scaffolds for LIBERO
  and RoboTwin.

### Changed

- Separated agent finish decisions, motion-planner success, and simulator task
  success in trial and viewer results.
- Let the RoboTwin batch runner accept an explicit actual simulator seed and
  task-config name, and keep seed-specific generated configs and outputs
  separate.
- Allow the outer benchmark to override the RoboTwin root, Python, and GPU
  without committing host-specific edits to experiment configs.
- Harden RoboTwin sweeps with configured-seed precedence, bounded process-group
  timeouts, unstable-scene seed fallback, classified infrastructure failures,
  and portable discovery of result configs copied from remote workers.
- Expanded the PhysicalAgent comparison to distinguish the legacy privileged
  oracle from the `dev-local-bringup` file-handoff runtime and to define a
  reproducible cross-system benchmark boundary.

### Experimental results

- Completed the ten-task `exp00b` GLM-5.2/CaP-X RoboTwin baseline: 10/10 tasks
  attempted, 9/10 produced trials, 70% code-execution rate, 0% task-success
  rate, and one infrastructure failure.

### Verification

- `55 passed, 3 subtests passed` across the changed Python test surface.
- `29 passed` across the exp00b runner, RoboTwin adapter, and viewer regression
  surface.
- `npm run build` completed for `web-ui`.
- `git diff --check` completed successfully.

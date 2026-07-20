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
- Expanded the PhysicalAgent comparison to distinguish the legacy privileged
  oracle from the `dev-local-bringup` file-handoff runtime and to define a
  reproducible cross-system benchmark boundary.

### Verification

- `55 passed, 3 subtests passed` across the changed Python test surface.
- `npm run build` completed for `web-ui`.
- `git diff --check` completed successfully.

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
- Made trajectory rendering frame- and segment-aware, with strict joint-name
  replay mapping and explicit `unknown` semantics for checks that were not run.
- Kept simulator, rendering, and shutdown operations on one owner thread;
  completed and failed Web sessions now retain their live scene across refreshes.
- Documented the observer architecture, policy-rollout diagnosis workflow,
  simulator-runtime isolation, and real acceptance baselines.
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

### Fixed

- Calibrated LIBERO Panda URDF hand FK into the MuJoCo `robot0_base` control
  convention so planned and executed EEF paths are directly comparable.
- Preserved the last trajectory artifact on policy timeout and forced a
  truncated failure instead of resetting the scene or leaking prior success.
- Prevented Viser port fallback, cross-session proxying, and config-load races
  from replacing or hiding the active session's final visualization.

### Experimental results

- Completed the ten-task `exp00b` GLM-5.2/CaP-X RoboTwin baseline: 10/10 tasks
  attempted, 9/10 produced trials, 70% code-execution rate, 0% task-success
  rate, and one infrastructure failure.

### Verification

- `64 passed, 3 subtests passed` across the trajectory, simulator adapter, Web
  visualization, artifact-saving, and viewer regression surface.
- `29 passed` across the exp00b runner, RoboTwin adapter, and viewer regression
  surface.
- A real LIBERO oracle completed with 580 commanded and 580 executed samples;
  calibrated dynamic EEF error was `6.65e-6 m`.
- A real RoboTwin ALOHA oracle completed planner and task success with 4,139
  physics steps; all 16 plan lineages and 82 expected renderer ticks were present.
- Offline Viser returned HTTP 200 and distinct planner/task status for both a
  successful RoboTwin artifact and a retained planner-failure artifact.
- `npm --prefix web-ui run build` completed successfully.
- `git diff --check` completed successfully.

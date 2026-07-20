# Changelog

## Unreleased

### Added

- Added layered RoboTwin and LIBERO trajectory instrumentation, replay artifacts,
  Viser integration, and live Web UI status for planner and task success.
- Added OpenAI Responses and OpenAI-compatible chat proxies, reusable provider
  smoke tests, and native GLM-5.2 client support.

### Changed

- Separated agent finish decisions, motion-planner success, and simulator task
  success in trial artifacts and the native Web UI.
- Made trajectory rendering frame- and segment-aware, with strict joint-name
  replay mapping and explicit `unknown` semantics for checks that were not run.
- Kept simulator, rendering, and shutdown operations on one owner thread;
  completed and failed Web sessions now retain their live scene across refreshes.
- Documented the observer architecture, policy-rollout diagnosis workflow,
  simulator-runtime isolation, and real acceptance baselines.
- Moved matched experiment scaffolds, result normalization, PhysicalAgent
  reviews, analysis slides, and cross-system visualizations to the separate
  Agent as Policy control-plane repository. CaP-X now retains only runtime,
  native configuration, implementation documentation, and system tests.

### Fixed

- Calibrated LIBERO Panda URDF hand FK into the MuJoCo `robot0_base` control
  convention so planned and executed EEF paths are directly comparable.
- Preserved the last trajectory artifact on policy timeout and forced a
  truncated failure instead of resetting the scene or leaking prior success.
- Prevented Viser port fallback, cross-session proxying, and config-load races
  from replacing or hiding the active session's final visualization.

### Verification

- `64 passed, 3 subtests passed` across the trajectory, simulator adapter, Web
  visualization, and artifact-saving regression surface before the repository
  boundary migration.
- A real LIBERO oracle completed with 580 commanded and 580 executed samples;
  calibrated dynamic EEF error was `6.65e-6 m`.
- A real RoboTwin ALOHA oracle completed planner and task success with 4,139
  physics steps; all 16 plan lineages and 82 expected renderer ticks were present.
- Offline Viser returned HTTP 200 and distinct planner/task status for both a
  successful RoboTwin artifact and a retained planner-failure artifact.
- `npm --prefix web-ui run build` completed successfully.
- `git diff --check` completed successfully.

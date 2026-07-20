# exp01b: CaP-X on RoboTwin and LIBERO

## Motivation

Run both simulators through the CaP-X code-execution harness on xdlab23, retain
replayable videos and machine-readable success semantics, and expose them in a
single read-only results viewer.

## Relation to Previous

`exp01a` established that RoboTwin's official `play_once()` oracle can solve
`stack_blocks_two`. It did not exercise CaP-X. This experiment adds a real
RoboTwin `BaseEnv` adapter and API, then uses the same CaP-X trial machinery as
the existing LIBERO integration.

## Settings

See `config.yaml`. The initial integration checks use repository oracle code so
simulator/API failures are separated from model-generation failures. A local
Qwen3-4B model smoke is recorded separately when available.

## Completion Criteria

- RoboTwin and LIBERO each complete at least one CaP-X trial on xdlab23.
- Each result reports code execution, planning (where available), and task
  success separately.
- Each successful benchmark has a playable MP4.
- The results viewer lists both benchmarks and safely serves their artifacts.

## Findings

Both benchmarks completed one real `CodeExecutionEnvBase` trial on xdlab23:

| Benchmark | Elapsed | Sandbox | Reward | Plan | Task success |
|---|---:|---:|---:|---:|---:|
| LIBERO `libero_object_swap[7]` | 42.33 s | 0 | 1.0 | N/A | True |
| RoboTwin `stack_blocks_two` | 77.28 s | 0 | 1.0 | True | True |

Each trial produced a replayable combined MP4 and a normalized
`capx.result.v1` manifest consumed by the viewer. This closes the integration
loop from simulator reset, API injection and sandboxed Python execution through
low-level motion, the benchmark verifier, result normalization and replay.

This is an execution-substrate result, not a learned-policy or RL result. Both
configs deliberately use privileged repository oracle programs, and each row
contains one deterministic trial. There is no rollout group, advantage
estimate, gradient update or checkpoint in exp01b. In particular, this result
does not yet establish model-generated-code reliability, visual perception,
random-seed robustness or zero-shot policy transfer between embodiments.

The next controlled sequence is: (1) disable oracle code and evaluate a frozen
Qwen model on fixed seeds, reporting code execution, planning and task success
separately; (2) ablate privileged state toward RGB-D observations; (3) factor a
shared `grasp / transport / place` skill IR; and only then (4) compare
LIBERO-only, RoboTwin-only and joint GRPO, including train-on-A/test-on-B
cross-embodiment evaluation.

Quantitative rows and artifact paths are recorded in `results/summary.md` and
`results/manifest.json`.

## Pitfalls

- Do not count `exp01a` as a CaP-X run; it bypasses the code-execution harness.
- Keep all new environments and outputs under `/data1`; the server home volume
  is nearly full.
- RoboTwin and LIBERO require different Python runtimes. `run.py` invokes each
  interpreter explicitly instead of trying to merge their simulator stacks.
- The vendored robosuite normally opens the global `/tmp/robosuite.log`. On a
  shared host that file can belong to another user. `run.py` creates the
  submodule-ignored `macros_private.py` from the tracked template only when it
  is absent; it never overwrites an existing host override.

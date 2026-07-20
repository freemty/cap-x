# PhysicalAgent versus CaP-X

This audit compares `freemty/PhysicalAgent` at
`2f6bfc424524e06e4fcb0d171c8cfbc8b6a475f0` with CaP-X and the exp01b
RoboTwin adapter. PhysicalAgent is a private, rapidly changing evaluation
workspace, so all claims below are commit-specific.

## Verdict

PhysicalAgent is not an alternative CaP-RL implementation. Its useful core is
a RoboTwin/SAPIEN runtime, imitation-learning baselines, and closed-loop policy
evaluation clients. CaP-X should remain the owner of the common environment,
reward, evaluation, result-viewer, and GRPO layers. The reusable portion of
PhysicalAgent is its observation/action/embodiment/evaluation contract.

The two policy families should coexist behind separate adapters:

```text
CodePolicyAdapter                         DirectPolicyAdapter
task + API docs + images                  task + RGB-D + state
        |                                         |
        v                                         v
LLM-generated Python                       VLA action chunk
        |                                         |
        v                                         v
named planner primitives                 qpos / absolute EE / delta EE
        +--------------------+--------------------+
                             |
                             v
               common simulator + verifier
                             |
                             v
                 reward + normalized artifacts
```

## Implementation differences

| Layer | PhysicalAgent | CaP-X | Consequence |
|---|---|---|---|
| Policy action | A direct VLA returns qpos or 14D/16D dual-arm EEF chunks | An LLM emits Python that invokes documented tools | Keep two versioned action schemas; do not disguise one as the other |
| Environment | `Base_Task.setup_demo/get_obs/take_action/check_success`; not a standard Gym reset/step contract | `BaseEnv` plus `CodeExecutionEnvBase` | Wrap RoboTwin with CaP-X rather than copying its runtime tree |
| Tool layer | Oracle planner methods and learned-policy actions live on the same task object | `ApiBase` separately registers and documents exposed functions | CaP-X has the cleaner privilege and prompt boundary |
| Training | Successful oracle demonstrations followed by ACT/DP3/pi0-style BC, diffusion, or flow-matching losses | VeRL GRPO with simulator reward is present for existing CaP-X tasks | PhysicalAgent adds an offline baseline, not an RL learner |
| Simulator scope | Core is RoboTwin/SAPIEN; LIBERO scripts are separate evaluators | LIBERO/robosuite, BEHAVIOR, real Franka, and the new RoboTwin adapter | CaP-X remains the cross-simulator layer |
| Cross embodiment | URDF/controller configuration is swappable, but the VA decoder is hard-coded to dual-arm 14D/16D | Simulator/API integration can vary by embodiment, but there is no universal learned policy yet | Reuse the embodiment schema; do not claim policy transfer |
| Evaluation | Several incompatible in-process, TCP, and WebSocket clients | One trial harness plus normalized manifest and viewer | Port the protocol metadata, not the launchers |

## Exact PhysicalAgent dataflow

The main VA client (`script/eval_policy_client_va.py`) does the following:

1. Run privileged `play_once()` and retain only seeds for which planning and the
   task predicate both succeed.
2. Reset the same seed and generate task text from oracle metadata.
3. Send three RGB views, joint state, optional depth/calibration, 16D EEF state,
   and task text to an external policy server.
4. Read an action tensor shaped as action-dimension by columns by temporal steps.
5. Convert a 14D dual-arm Euler action to 16D quaternion form, or normalize and
   execute a 16D absolute EEF action.
6. Call `Base_Task.take_action(..., "ee")` and stop when the task predicate or
   step limit fires.

This differs fundamentally from CaP-X, where the online action is a Python
program, the execution namespace contains functions exported by `ApiBase`, and
the simulator computes task reward after the program executes.

PhysicalAgent's `code_gen/` directory does not change that conclusion. It
generates an offline privileged `play_once()` subclass into `envs_gen/`; it is
not an online sandboxed policy and is not connected to RL.

## What PhysicalAgent does not currently provide

- No PPO/GRPO loop, advantage estimate, critic, online learner, or RL
  checkpoint. The committed trainers are imitation-learning objectives.
- No unified LIBERO/RoboTwin environment interface. LIBERO is reached through
  separate deployment scripts.
- No learned cross-embodiment policy: there is no robot-ID conditioning,
  canonical action tokenizer, retargeter, or embodiment-mixture training.
- Several advertised Wan, V-JEPA, and Claude policy adapters are stubs whose
  real server and parser live in an external workspace.
- The committed checkout is not fresh-clone reproducible: task configuration is
  ignored, the asset symlink is host-absolute, and one VA client imports a
  missing communication utility.

Its reported evaluation success is also conditional on an expert-solvable seed
filter. That is a valid controlled protocol, but it must not be reported as
unconditional success over the requested seed interval.

## Merge plan

Port these contracts into CaP-X:

1. `RoboTwinObservationV1`: task text, RGB/RGB-D, camera intrinsics/extrinsics,
   EEF and gripper state. Standardize depth in meters; PhysicalAgent currently
   multiplies it by 1000.
2. `RoboTwinActionV1`: `qpos | ee_absolute | ee_delta`, with dimensions derived
   from the embodiment and validation for finite values, normalized
   quaternions, workspace bounds, and gripper ranges.
3. `EmbodimentSpec(left, right, separation)`: URDF/SRDF, joint and move-group
   names, EEF frames, gripper transforms, home pose, and planner parameters.
4. `EvaluationProtocolV1`: requested and accepted seeds, expert-filter mode,
   camera protocol, step limit, code/model hashes, numerator/denominator, and
   skip reasons.
5. `DirectPolicyAdapter`: a versioned transport schema for direct VLA action
   chunks, parallel to the existing code-program policy.
6. A fresh-observation contract after every primitive or chunk, including the
   reached EEF state, rather than hidden server cache state.

Do not port the policy stubs, vendored baseline trees, host-specific launchers,
manual `action.json` handoff, hard-coded episode counts, or external benchmark
claims without their artifacts.

## Current CaP-X boundary

exp01b already validates the shared execution substrate on both LIBERO and
RoboTwin. It remains privileged-oracle integration evidence, not learned-policy
or RL evidence. The RoboTwin adapter now also gates actor ground-truth poses and
task status behind `privileged=True`, so future visual-policy runs do not
silently inherit those fields.

Before RoboTwin GRPO, two additional CaP-X changes are required:

- register a task-specific RoboTwin code environment and reset it before the
  dataset builder reads the first observation;
- keep code-execution success, planner success, and task success as distinct
  metrics. Existing reward shaping and `TrialSummary.success` cannot be used as
  aliases for benchmark task success.

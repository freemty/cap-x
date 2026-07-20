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
| Tool layer | Oracle planner methods and learned-policy actions live on the same task object | `ApiBase` separately registers and documents exposed functions, but generated code also receives the raw low-level `env` | Neither path currently enforces a strict privilege boundary; use a restricted capability interface for fair evaluation |
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

## What actually solves a task

PhysicalAgent contains two different solvers that must not be conflated.

The expert solver is a privileged scripted program:

```text
SAPIEN actor handle and exact pose
        |
        v
annotated contact/functional frames
        |
        v
grasp_actor / place_actor / move
        |
        v
MPLib path planning and dense joint control
```

The learned-policy solver does not call `grasp_actor()` in the built-in
adapters inspected at the audited commit:

```text
allowed observation + task instruction
        |
        v
ACT / DP3 / pi0 / OpenVLA / RDT / DexVLA / external VA server
        |
        v
qpos or absolute/delta EEF action chunk
        |
        v
Base_Task.take_action()
        |
        v
joint interpolation or planner-assisted EEF execution
```

The built-in adapters use the following model inputs:

| Adapter | Model-visible input | Model output |
|---|---|---|
| ACT | Three RGB views and qpos; language is commented out | qpos chunks |
| DP3 | Point cloud and joint vector | qpos chunks |
| pi0/OpenVLA/RDT/DexVLA | RGB/state plus the generated task instruction | qpos chunks |
| Custom VA client | Three RGB views, joint state, 16D EEF state, task text, and optional head depth/calibration | 14D Euler or 16D quaternion EEF chunks |

The standard adapter function nevertheless receives the whole `TASK_ENV`
object as well as the observation. Existing built-in adapters only use
`get_instruction()`, `get_obs()`, and `take_action()`, but the interface does
not prevent a custom adapter from reading task actors or calling oracle
methods. This is a capability leak even when existing baselines do not exploit
it.

## What `grasp_actor()` really does

`Base_Task.grasp_actor(actor, arm_tag, pre_grasp_dis, grasp_dis,
gripper_pos, contact_point_id)` is an oracle motion primitive, not a visual
grasp model. It returns an arm tag plus a list of internal actions:

```text
move to pre-grasp 7D pose
move to final grasp 7D pose
close the normalized gripper
```

For mesh and URDF objects, `assets/objects/<name>/model_data*.json` stores
object-local contact and functional frames. The runtime computes the world
contact frame from the exact SAPIEN actor transform, applies a fixed gripper
frame rotation and TCP offset, tests rotated candidates with the motion
planner, and then converts the selected gripper pose into the embodiment's
planner end-link frame. MPLib produces joint position/velocity trajectories;
SAPIEN drive targets close the base and mimic gripper joints.

This path does not estimate the object pose from RGB or depth. Its apparent
precision comes from exact simulator state, pre-annotated affordance frames,
perfect robot calibration, exact collision geometry, and high-rate
deterministic control. Depth is optional observation data for learned policies
such as DP3 or the VA client. The current oracle `grasp_actor()` and MPLib path
do not consume the rendered depth map; point-cloud planner arguments exist but
are not wired through in the audited implementation.

The CaP-X RoboTwin wrapper further simplifies this to
`grasp_actor(name, arm="auto", pre_grasp_dis=...) -> bool`. It resolves the
name to the SAPIEN actor, invokes the PhysicalAgent primitive, and executes the
returned action list. The boolean reports code/planning execution, not proof
that the object is stably grasped; task completion must be checked separately.

## Evaluation leakage audit

The learned model is not directly given actor poses or annotated contact
points by the inspected built-in adapters. The surrounding evaluation
protocol is nevertheless not a clean end-to-end benchmark.

| Channel | Status | Effect |
|---|---|---|
| Exact actor pose/contact point in built-in model input | Not found | The learned action predictor still has to infer targets from its allowed observations |
| Oracle trajectory copied into policy rollout | Not found | The oracle scene is closed and the same seed is reset; trajectory lists and counters are reinitialized |
| Expert-solvable seed filter | Confirmed | Reports `P(policy success | oracle success)`, not success over all requested random seeds |
| Oracle-derived task instruction | Confirmed | `episode_info["info"]` can include the oracle's left/right arm assignment; language-conditioned policies can receive this privileged decomposition hint |
| Whole `TASK_ENV` passed to policy adapter | Confirmed interface weakness | A custom adapter can read actor state or call oracle methods even though inspected baselines do not |
| Exact-scene EEF motion planner | Confirmed privileged assistance | EEF policies predict a target pose but receive perfect-calibration IK/path execution against simulator geometry |
| Ground-truth task predicate | Confirmed and acceptable for evaluation | Used to terminate and score the episode; it is not part of the inspected model observation |

The seed filter occurs before every accepted evaluation episode: the harness
runs `play_once()`, retains only seeds satisfying both planner success and the
task predicate, closes the scene, and resets the same seed for the learned
policy. Evaluation seeds start in a high numbered range, which helps separate
them from ordinary training seeds, but does not remove the conditional
denominator bias.

The instruction leak is task and template dependent. For example,
`stack_blocks_two.play_once()` records which arm the oracle chose for each
block. Some instruction templates replace `{a}` and `{b}` with phrases such as
"the left arm" and "the right arm". ACT and DP3 ignore that text, while pi0,
OpenVLA, RDT, DexVLA, and the custom VA path can consume it.

The correct interpretation is therefore:

- the inspected built-in learned policies are not simply calling the
  privileged `grasp_actor()` oracle;
- their reported success is still affected by oracle selection, occasional
  oracle-derived language, and planner assistance;
- results must be labeled by observation privilege, controller privilege, and
  denominator rather than summarized as one undifferentiated success rate.

## Clean evaluation protocol

A fair non-privileged track should enforce the following contract:

1. Freeze and publish a requested seed manifest. Evaluate every stable scene,
   or report both unconditional success and the separately defined
   oracle-solvable conditional success.
2. Generate instructions from task semantics only. Do not insert an
   oracle-selected arm, contact point, target pose, or execution plan.
3. Replace `eval(TASK_ENV, model, observation)` with a restricted policy
   interface that exposes only a versioned observation and an action sink.
4. Keep actor handles, task attributes, annotated contact frames, raw simulator
   objects, and private environment methods outside the policy process.
5. Report direct-qpos, planner-assisted EEF, modular RGB-D perception, and
   privileged oracle as separate tracks.
6. Record requested seeds, accepted/skipped seeds and reasons, model-visible
   fields, instruction text, controller type, and task-success predicate in
   every result manifest.
7. Add negative tests proving that an unprivileged policy cannot obtain actor
   poses, contact frames, oracle actions, or the low-level environment object.

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
filter. This may be retained as a separately labeled controlled upper-bound
protocol, but it must not be reported as unconditional success over the
requested seed interval. Language-conditioned runs must additionally disclose
whether oracle-derived arm placeholders were eligible for sampling.

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
task status behind `privileged=True`. This observation gating is not yet a hard
security boundary: `CodeExecutionEnvBase` injects the raw low-level `env` into
the generated program namespace, so generated code can bypass documented APIs.
An unprivileged CaP-X track must remove that capability rather than relying on
the observation dictionary alone.

Before RoboTwin GRPO, two additional CaP-X changes are required:

- register a task-specific RoboTwin code environment and reset it before the
  dataset builder reads the first observation;
- keep code-execution success, planner success, and task success as distinct
  metrics. Existing reward shaping and `TrialSummary.success` cannot be used as
  aliases for benchmark task success.

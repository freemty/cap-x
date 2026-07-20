# PhysicalAgent, CaP-X, and Guava

This audit has two commit-pinned PhysicalAgent scopes:

- the older `freemty/PhysicalAgent` snapshot at
  `2f6bfc424524e06e4fcb0d171c8cfbc8b6a475f0`, used below to audit the
  privileged `play_once()` oracle and learned-policy clients; and
- the current default comparison target,
  `cherubicXN/PhysicalAgent:dev-local-bringup` at
  `e57e677584abf529e7cbef18b989eb98eb55cd6d`, used to audit the file-handoff
  planner runtime.

The comparison also covers CaP-X's exp01b RoboTwin adapter and uses Guava arXiv
`2606.18363v1` for the harness-level comparison. PhysicalAgent is a private,
rapidly changing evaluation workspace, so all claims are commit-specific;
never identify an experiment only by the branch name.

The audited PhysicalAgent commit contains no file, class, or symbol named
`RAPPLE`. In this document, "the no-VLA route" refers specifically to the
`play_once()` + `Base_Task` oracle primitives + motion-planner stack.

## Current default: `dev-local-bringup`

The current branch is not the legacy no-VLA oracle route. It is a closed-loop
planner harness in which a general multimodal agent reads three live RGB views,
current absolute end-effector/gripper state, task text, and optional RGB-D
pixel-query results, then writes one absolute Cartesian target to
`action.json`. RoboTwin's CuRobo/MPLib layer plans and executes that target,
and the agent receives a fresh observation before the next primitive.

Its strict protocol explicitly prohibits simulator object-pose queries and
task-source reads. Ground-truth object pose is therefore not part of the
planner-facing interface at the audited commit. The runtime does know the
robot's own end-effector and gripper state, uses calibrated depth/intrinsics/
extrinsics to backproject requested pixels, and relies on mandatory expert
demonstrations plus persistent global/task memory. Those are strong forms of
scaffolding, but they are different from GT object pose leakage and must be
reported separately.

Relative to CaP-X, both systems use a foundation-model planner over explicit
robot tools and a classical motion planner. Their central difference is the
online policy representation:

| `dev-local-bringup` | CaP-X |
|---|---|
| One constrained JSON primitive per model turn | An executable Python program per model turn |
| Fresh observation after every primitive | A program may compose many primitives before another observation |
| Pixel-query RGB-D localization | Modular perception/geometric APIs selected by the environment tier |
| Mandatory demonstrations and persistent task/global memory | Demonstrations and memory are experiment choices, not runtime invariants |
| No task source or raw simulator handle in the declared planner contract | Current code-execution globals include `env`, so the fair track needs a restricted namespace |

Thus `dev-local-bringup` is best treated as another point in the CaP-X harness
design space: approximately CaP-X's modular tool setting with a constrained
one-primitive REPL, mandatory demonstrations, and persistent memory. It is not
an end-to-end VLA, but it is also not the privileged `play_once()` oracle.

## Verdict

PhysicalAgent is not an alternative CaP-RL implementation. Its useful core is
a RoboTwin/SAPIEN runtime, imitation-learning baselines, and closed-loop policy
evaluation clients. CaP-X should remain the owner of the common environment,
reward, evaluation, result-viewer, and GRPO layers. The reusable portion of
PhysicalAgent is its observation/action/embodiment/evaluation contract.

PhysicalAgent's no-VLA `play_once()` route is a privileged scripted
task-and-motion-planning oracle. That is a legitimate expert baseline,
demonstration generator, and simulator/planner validation tool. It is not an
end-to-end visual agent. Describing it as a general non-VLA autonomous agent,
or comparing its success rate directly with an RGB-D-conditioned VLA without
labeling the privilege difference, would be benchmark leakage.

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

## How the no-VLA oracle replaces a learned policy

The no-VLA route does not replace the VLA with another online model. It moves
the missing intelligence into three human-authored layers:

1. **Asset annotations.** `model_data*.json` stores object-local contact and
   functional frames. At runtime, the exact SAPIEN actor pose transforms these
   frames into world coordinates:

   ```text
   T_world_point = T_world_actor @ T_actor_point
   ```

   There is no RGB-D object identification or pose estimation on this oracle
   path.
2. **Generic geometric macros.** `grasp_actor()` selects a reachable annotated
   contact frame and expands it into pre-grasp, grasp, and gripper-close
   actions. `place_actor()` preserves the held-object/gripper transform and
   aligns an object functional frame with a target functional frame before
   opening the gripper.
3. **Task-specific programs.** Each task's `play_once()` decides which actor to
   manipulate, which arm and functional-point IDs to use, the operation order,
   and any task-specific offsets. `stack_blocks_two`, for example, chooses the
   arm from the exact block position, grasps and lifts the block, queries the
   preceding block's functional point, and calls `place_actor()` to align the
   frames.

The geometric intermediate representation contains only
`Action(move, 7D_pose)` and `Action(gripper, value)`. MPLib or CuRobo converts
move actions into joint trajectories, and SAPIEN drive targets execute those
trajectories. The optional `code_gen/` path can use an LLM offline to synthesize
`play_once()`, but it receives actor variables and functional-point guidance;
the resulting runtime remains a privileged program rather than an online
visual policy.

## Three-way tool-abstraction comparison

The important distinction is not just whether a function is called
`grasp`. It is the epistemic contract of its arguments and the work hidden
behind the call.

| Dimension | CaP-X | Guava | PhysicalAgent no-VLA oracle |
|---|---|---|---|
| Object reference | A semantic string such as `"red cube"`; S2 uses visual grounding while S1 may use privileged state | A semantic string grounded from RGB-D with SAM3 | A direct simulator `Actor` handle with exact pose |
| Grasp tool | `sample_grasp_pose(name)` returns a pose; generated code still composes open, move, close, and lift | `grasp(object)` performs segmentation, grasp planning, approach, close, and reports gripper outcome | `grasp_actor(actor)` expands annotated contact frames into pre-grasp, grasp, and close actions |
| Grasp source | SAM3/Molmo, point clouds, Contact-GraspNet, and frame transforms in the non-privileged path | SAM3 plus a learned 6-DoF grasp planner or PCA top-down baseline | Hand-annotated contact frames plus exact simulator state |
| Placement/alignment | Generated code generally computes the target pose, offsets, and sequence | `align(object, direction, clearance)` exposes categorical relative geometry | `place_actor()` performs functional-frame-to-functional-frame SE(3) alignment |
| Task decomposition | A coding agent writes a Python program with branches, loops, and multiple primitive calls | A VLM selects one tool per ReAct step and replans from new observations | A human or offline code generator writes the fixed `play_once()` program |
| Lower layer | Can expose segmentation, point clouds, grasp candidates, transforms, IK, and joint movement | The low-level ablation exposes a fused absolute Cartesian pose and gripper width | Internal `Action(SE3/gripper)` is lowered through a motion planner and dense controller |
| Online recovery | Single-turn execution or multi-turn code regeneration and visual feedback | Fresh perception-reasoning-action loop after each tool | No semantic recovery unless explicitly programmed; planner failure and the final task predicate remain available |
| Dominant prior | Human API design plus the coding model's composition ability | Human semantic-skill design plus the VLM's tool selection | Asset annotations, exact state, task program, and planner |

This produces different orderings along different axes:

- Guava has the most semantically complete `grasp()` call.
- CaP-X gives the coding agent the greatest freedom to compose and inspect
  intermediate perception and geometry.
- PhysicalAgent has the strongest functional-frame placement abstraction, but
  only because those frames and object identities are provided as privileged
  inputs.

## What the abstraction studies do and do not establish

CaP-X makes abstraction level an explicit benchmark axis. Its high-level S1/S2
tiers collapse perception, geometric reasoning, and control behind
human-designed helpers, while S3/S4 expose the constituent perception and
control modules. The reported conclusion is a trade-off: higher abstraction
raises task success by reducing the program search space, but imposes a
generality and expressivity ceiling.

Guava directly compares its semantic tool set with a four-tool geometric
interface consisting of absolute Cartesian pose plus gripper width, object
position/size queries, and home pose. The aggregate harness ablation favors the
semantic interface (`41%` versus `29%`), but the result is not task-universal:
the reported push task favors the lower-level interface (`47%` versus `20%`).
Guava therefore supports the claim that semantic tools help overall, not that
maximal abstraction is always optimal.

PhysicalAgent does not contain a matched abstraction-level ablation. Its
privileged expert and learned-policy paths change observation privilege, object
representation, task decomposition, action space, and planner assistance at
the same time. Their success rates cannot isolate the causal effect of tool
abstraction.

## When the no-VLA route is and is not a fair baseline

The privileged route is valid when it is labeled and used as one of the
following:

- a scripted expert or oracle upper bound;
- a demonstration generator for behavior cloning;
- a simulator, asset, controller, or motion-planner validation test;
- an explicitly conditional `P(policy success | oracle success)` protocol,
  reported alongside unconditional success.

It becomes an invalid comparison when a result:

- describes `play_once()` as a visual or general autonomous agent;
- compares it directly with an RGB-D-conditioned VLA under one undifferentiated
  success-rate column;
- hides actor handles, exact poses, contact/functional frames, or task-specific
  scripts from the privilege declaration;
- filters evaluation seeds through the expert without reporting the requested
  and accepted denominators.

A fair three-track abstraction study should keep the simulator seeds, motion
planner, controller, and `check_success()` predicate fixed while varying only
the agent-facing interface:

1. **PhysicalAgent-Oracle:** actor handles plus contact/functional frames.
2. **Guava-Semantic:** RGB-D plus semantic `grasp` and `align` tools.
3. **CaP-X-Modular:** RGB-D plus segmentation, grasp, transform, IK, and motion
   primitives.

Report grounding, grasp generation, motion planning, execution, and terminal
task success separately. Also report the manual annotation budget for object
frames and task-specific `play_once()` programs; otherwise human-provided task
knowledge is silently counted as agent capability.

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

## What the legacy `2f6bfc4` snapshot does not provide

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

## Repository maintenance for a PhysicalAgent versus CaP-X benchmark

CaP-X should own the benchmark control plane, while PhysicalAgent remains an
external, commit-pinned system under test. Do not copy PhysicalAgent into this
repository or add the private repository as a public submodule: either choice
couples the public checkout to private, fast-moving code and obscures which
system was actually evaluated. Use sibling checkouts discovered through
`PHYSICAL_AGENT_ROOT` and `ROBOTWIN_ROOT`, and fail a launch when their commits
do not match the run lock.

Maintain two primary scoreboards and one diagnostic upper bound:

1. **Native-system performance.** Each system keeps its intended advantages.
   PhysicalAgent uses mandatory demonstrations, persistent memory, one JSON
   primitive per turn, and fresh observations. CaP-X uses its native Python
   code-policy interface and configured perception tools. This answers which
   complete system works better, but not why.
2. **Matched-interface performance.** Fix the task/seed manifest, simulator,
   embodiment, camera fields, controller/motion planner, planner model,
   demonstration and memory allowance, and resource budgets. Vary only the
   policy representation: one-primitive JSON REPL versus restricted Python
   composition over the same primitive library. This is the causal harness
   comparison.
3. **Privileged oracle upper bound.** Keep `play_once()` and actor/frame APIs in
   a separate table. Never mix them into either primary aggregate.

The reusable code and immutable experiment instances should be separated:

```text
capx/benchmarks/physicalagent_vs_capx/
  contracts.py              # versioned observation/action/episode DTOs
  runner.py                 # common episode and budget accounting
  adapters/
    capx.py                  # native and matched CaP-X policies
    physicalagent.py         # external file-handoff process adapter
  normalize.py              # extend capx.result.v1; do not invent a viewer fork
  validate.py               # privilege, schema, and version-lock checks

exp/<next-id>-pa-capx-*/
  README.md                  # hypothesis, exact comparison, findings
  config.yaml                # human-readable run configuration
  versions.lock.yaml         # immutable repositories, assets, model, controller
  seeds.jsonl                # requested seeds; never silently rewrite this file
  results/manifest.json      # small normalized records tracked in Git
```

Large RGB-D traces, HDF5 demonstrations, and videos should live in artifact
storage. Git tracks their URI, size, and checksum. Every run lock should record
at least the CaP-X, PhysicalAgent, and RoboTwin commit SHAs; dirty-worktree
status and diff hash; task config and embodiment; asset checksum; model ID and
serving parameters; planner/controller configuration; and observation/action
schema versions. Recording only `dev-local-bringup` is not reproducible.

Extend the existing `capx.result.v1` manifest rather than creating a separate
PhysicalAgent result format. Each episode needs:

- requested seed, executed seed, skip status, and skip reason;
- official task success, code/protocol validity, and motion-planner success as
  different fields;
- model calls, robot primitives, simulator steps, input/output tokens, wall
  time, and estimated cost;
- exact model-visible observation fields, controller privilege, action schema,
  demo count, memory mode and memory snapshot hash;
- a failure label from localization, invalid action/code, planning, grasp,
  transport, release, predicate, timeout, or infrastructure.

Agent turns alone are not a fair budget: one PhysicalAgent turn executes one
primitive, while one CaP-X program can execute many. Set and report independent
limits for model calls, robot primitives, simulator steps, tokens, and wall
time, then plot success against primitive/token budgets. The primary success
rate must cover all requested stable seeds. An oracle-solvable conditional
rate may be reported only as an additional metric with both denominators.

For the matched track, run the planner out of process with only a serialized
observation DTO and action sink. The current CaP-X
`CodeExecutionEnvBase._init_exec_globals()` exposes `env`, so
`privileged=False` observation gating alone is insufficient. Remove the raw
environment from this track and add negative tests proving that code cannot
read task actors, simulator poses, task source, annotated functional frames,
or the success predicate. Ground-truth success remains evaluator-side only.

The minimum useful rollout sequence is:

1. protocol smoke: three representative tasks, one fixed seed, both adapters;
2. fairness smoke: the same tasks over ten fixed seeds, with privilege and
   budget validation enabled;
3. reportable run: the frozen task suite over at least 100 requested seeds per
   setting, with bootstrap or Wilson confidence intervals;
4. ablations for demos, persistent memory, one-versus-many primitives per
   model call, RGB versus RGB-D, and semantic versus modular tools.

CI should stay cheap: unit-test DTO round trips and action validation, verify
every run lock and seed manifest, reject dirty/unpinned reportable launches,
run one fixed-seed simulator smoke, validate `capx.result.v1`, and include a
negative capability test for raw-environment access. Expensive benchmark runs
remain explicit jobs whose artifacts are normalized by the same code.

## Audited sources

- [PhysicalAgent commit `2f6bfc4`](https://github.com/freemty/PhysicalAgent/tree/2f6bfc424524e06e4fcb0d171c8cfbc8b6a475f0)
- [PhysicalAgent `dev-local-bringup` commit `e57e677`](https://github.com/cherubicXN/PhysicalAgent/tree/e57e677584abf529e7cbef18b989eb98eb55cd6d)
- [CaP-X paper, arXiv:2603.22435](https://arxiv.org/abs/2603.22435)
- [Guava paper, arXiv:2606.18363](https://arxiv.org/abs/2606.18363)

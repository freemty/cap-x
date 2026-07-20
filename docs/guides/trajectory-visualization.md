# Policy trajectory visualization

CaP-X can record and visualize the same four-stage policy trajectory in LIBERO
and RoboTwin. Recording is always enabled in these two adapters. Setting
`viser_debug: true` adds a live Viser view; setting `output_dir` also saves a
replayable `trajectory.json` beside the normal trial artifacts.

## What the view means

| Layer | Color | Meaning |
| --- | --- | --- |
| `raw` | gray | High-level policy or planner request, when the simulator exposes it |
| `planned` | blue | Planner joint waypoints, converted to world/base-frame EEF poses with the embodiment URDF |
| `commanded` | orange | Joint or drive targets sent to the controller |
| `executed` | green | Measured simulator state and EEF pose after stepping physics |
| infeasible segment | red | At least one supplied feasibility check failed |

The GUI has a timestep scrubber, one visibility toggle per layer, EEF-axis
visibility, sample counts, arm names, and the number of samples flagged as
infeasible. The robot follows the selected timestep when the simulator has
provided a matching URDF state setter.

The visualization is diagnostic rather than a collision certificate. A red
segment is meaningful only for checks actually supplied by the planner or
simulator. `null` feasibility fields mean “not evaluated,” not “passed.” The
Web completion card separately reports planner success, the agent's `Finish`
decision, and the simulator's task predicate.

## Live use

RoboTwin has a ready-to-run trajectory config:

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin uv run capx/envs/launch.py \
  --config-path env_configs/robotwin/stack_blocks_two_privileged_oracle_trajectory.yaml
```

For LIBERO, use either existing Viser config:

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_wine_bottle_cabinet_viser.yaml
```

For any other LIBERO or RoboTwin YAML, add `viser_debug: true` to the nested
`cfg.low_level` mapping. Keep `num_workers: 1` for an interactive live view.
Batch workers still record trajectory artifacts when Viser is disabled.
RoboTwin retains 10,000 samples per layer by default; raise
`trajectory_max_samples_per_layer` for tasks with longer native physics traces.

The interactive Web UI enables Viser on both the execution wrapper and nested
simulator automatically:

```bash
uv run capx/envs/launch.py \
  --config-path env_configs/robotwin/stack_blocks_two_privileged_oracle.yaml \
  --web-ui
```

The backend remembers the exact port selected by Viser and proxies both HTTP
and WebSocket traffic through `/viser-proxy`. Simulator construction, stepping,
rendering, and shutdown stay on one owner thread so MuJoCo/SAPIEN graphics
contexts are not closed from the event-loop thread. A completed session keeps
its view alive until it is stopped, replaced, or its final WebSocket disconnects.

## Offline replay

Every saved trial contains:

```text
trial_XX_.../
  code.py
  summary.txt
  trajectory.json
```

Replay paths and EEF frames directly:

```bash
uv run python -m capx.visualization.replay \
  outputs/.../trial_XX_.../trajectory.json --port 8080
```

For an artifact that has joint states but no recorded FK pose, supply a matching
URDF to animate the first recorded arm:

```bash
uv run python -m capx.visualization.replay trajectory.json \
  --urdf /path/to/robot.urdf --port 8080
```

The JSON schema is versioned and simulator-independent. Each sample contains
`sequence`, `step`, `timestamp_s`, `layer`, named arm states, optional EEF pose
in scalar-first `wxyz` with an explicit `frame_id`, feasibility metrics,
simulator payload, and metadata. Version-1 artifacts without `frame_id` load as
world-frame poses for backward compatibility. Use
`TrajectoryArtifact.load_json()` for analysis without starting Viser.

## Simulator coverage

LIBERO records controller targets and measured MuJoCo state on every control
step. Successful CuRobo paths are recorded as planned waypoints and use Panda
URDF FK. Its existing Viser scene also provides RGB-D point-cloud context.

RoboTwin instruments native `task.move`, `take_dense_action`, articulation drive
targets, and `scene.step` without changing the external RoboTwin checkout. It
records both arms, uses the selected embodiment's joint names and URDF transform
chain, and restores all monkey patches on reset or close. Actor markers provide
task-space context; calibrated camera point clouds are intentionally not inferred
when camera intrinsics/extrinsics are unavailable.

## Python interface

Low-level adapters expose:

```python
summary = env.trajectory_summary()       # compact and JSON-safe
artifact = env.trajectory_snapshot()     # immutable full snapshot
artifact.save_json("trajectory.json")
```

New adapters can use `TrajectoryRecorder.append_raw()`, `append_planned()`,
`append_commanded()`, and `append_executed()`, then attach a
`ViserTrajectoryRenderer`. The recorder is bounded and thread-safe; renderer
subscriber failures cannot make simulator stepping fail.

## Troubleshooting

- If live Viser initialization fails, trajectory recording and JSON export keep
  working; inspect the runtime warning and replay later.
- If a planned line is absent but joint samples exist, check that the matching
  URDF assets are installed and that the recorded joint names match its actuated
  joints.
- If the Web iframe says Viser is unavailable, confirm that the current session
  has finished environment initialization; the proxy does not guess another
  session's port before that point.
- Do not interpret missing collision or clearance metrics as successful checks.

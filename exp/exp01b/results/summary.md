# exp01b Results

Both rows below are real CaP-X `CodeExecutionEnvBase` trials on xdlab23;
the earlier `exp01a` direct RoboTwin oracle is not included.

| Benchmark | Task | Code execution | Plan | Task success | Reward | Elapsed | Video |
|---|---|---:|---:|---:|---:|---:|---|
| libero | `libero_object_swap[7]` | True | N/A | True | 1.000 | 42.33 s | `outputs/exp01b/oracle/libero/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined.mp4` |
| robotwin | `stack_blocks_two` | True | True | True | 1.000 | 77.28 s | `outputs/exp01b/oracle/robotwin/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined.mp4` |

## Interpretation

- Both simulators completed one repository-oracle program through the CaP-X trial harness with `sandbox_rc=0`, reward 1.0, and task success.
- RoboTwin additionally reports planner success because its reward is the conjunction of planner and task success. LIBERO has no separate planner predicate in this integration.
- These trials establish adapter and execution correctness. They do not measure learned-policy quality or constitute an RL training result.

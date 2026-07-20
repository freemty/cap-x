# Experiment Summary

| Experiment | Motivation | Status | Result |
|---|---|---|---|
| exp00a | Run one GLM-5.2 code-policy episode on every standard LIBERO and RoboTwin task | In Progress | 180 tasks: 130 LIBERO + 50 RoboTwin; no oracle |
| exp00b | Run GLM-5.2 through CaP-X on ten RoboTwin tasks and verify frontend ingestion | Analyzed | 10/10 attempted, 9 completed, 70% code execution, 0% task success, 1 infrastructure failure; privileged state, no oracle code |
| exp01a | Run a RoboTwin oracle smoke test on xdlab23 | Completed | `stack_blocks_two`, seed 0: plan success and task success (1/1, 48.60 s) |
| exp01b | Run CaP-X end to end on RoboTwin and LIBERO with unified replay UI | Analyzed | Both privileged-oracle CaP-X trials succeeded (`sandbox_rc=0`, reward 1.0, task success); integration milestone, not learned-policy/RL evidence |

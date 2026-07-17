# exp01a: Run a RoboTwin oracle smoke test on xdlab23

## Motivation

Establish a reproducible RoboTwin runtime on xdlab23 before adding the CaP-X code-policy adapter or GRPO training.

## Relation to Previous

First experiment — no predecessor.

## Settings

See `config.yaml` for configuration.

## Findings

- The official RoboTwin oracle completed `stack_blocks_two` with the `aloha-agilex` embodiment and seed 0 on xdlab23 GPU 5.
- The single trial had both `plan_success=true` and `success=true`; wall time was 48.60 seconds.
- Both blocks were handled by the left arm in this seed. The clean scene used no randomized wall or table texture.
- This establishes the simulator-and-planner baseline needed before adding a CaP-X code-policy adapter or RL training loop.

## Pitfalls

- Keep user-site Python packages disabled so the isolated Torch/SAPIEN environment is not polluted by `~/.local`.
- Validate the official oracle task before attributing failures to the future CaP-X adapter.
- PyTorch3D is optional for this RGB-only smoke path; RoboTwin falls back when point-cloud sampling is disabled.
- `CUDA_VISIBLE_DEVICES=5` placed CuRobo compute on GPU 5, but SAPIEN's default Vulkan renderer also opened a graphics context on physical GPU 0. Pin the Vulkan device before shared-server scaling runs.
- `demo_clean` does not access the 10.97 GB randomized-background archive, so only the embodiment and object archives were required.

"""Run the exp00b ten-task RoboTwin GLM-5.2 baseline."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp00a.run import main  # noqa: E402

if __name__ == "__main__":
    main(default_config="exp/exp00b/config.yaml")

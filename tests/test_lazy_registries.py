from __future__ import annotations

import subprocess
import sys


def test_registry_import_does_not_eagerly_load_simulator_stacks() -> None:
    script = r"""
import sys
from capx.envs.base import list_envs
from capx.envs.tasks import list_exec_envs
from capx.integrations.base_api import list_apis

assert "robotwin_low_level" in list_envs()
assert "franka_libero_code_env" in list_exec_envs()
assert "RoboTwinPrivilegedApi" in list_apis()
assert "FrankaLiberoPrivilegedApi" in list_apis()

for module in (
    "capx.envs.simulators.libero",
    "capx.envs.simulators.robotwin",
    "capx.integrations.franka.control",
    "capx.integrations.franka.libero_privileged",
    "capx.integrations.robotwin",
):
    assert module not in sys.modules, module
"""
    subprocess.run([sys.executable, "-c", script], check=True)

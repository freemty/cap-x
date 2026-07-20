"""Dependency-light simulator registry.

Concrete simulator modules are imported only when their registered environment
is constructed. This keeps mutually incompatible stacks such as RoboTwin,
LIBERO, robosuite, and BEHAVIOR isolated in separate Python environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from capx.envs.base import list_envs, register_env


def _lazy_env(
    module: str,
    class_name: str,
    **bound_kwargs: Any,
) -> Callable[..., Any]:
    def factory(*args: Any, **kwargs: Any) -> Any:
        env_class = getattr(import_module(module, package=__name__), class_name)
        return env_class(*args, **bound_kwargs, **kwargs)

    factory.__name__ = f"lazy_{class_name}"
    return factory


_ENV_SPECS: dict[str, tuple[str, str]] = {
    "franka_real_low_level": (".franka_real", "FrankaRealLowLevel"),
    "franka_robosuite_cube_lift_low_level": (".robosuite_cube_lift", "FrankaRobosuiteCubeLiftLowLevel"),
    "franka_robosuite_cubes_low_level": (".robosuite_cubes", "FrankaRobosuiteCubesLowLevel"),
    "franka_robosuite_cubes_restack_low_level": (".robosuite_cubes_restack", "FrankaRobosuiteCubesRestackLowLevel"),
    "franka_robosuite_spill_wipe_low_level": (".robosuite_spill_wipe", "FrankaRobosuiteSpillWipeLowLevel"),
    "franka_robosuite_nut_assembly_low_level": (".robosuite_nut_assembly", "FrankaRobosuiteNutAssembly"),
    "franka_robosuite_nut_assembly_low_level_visual": (".robosuite_nut_assembly", "FrankaRobosuiteNutAssemblyVisual"),
    "two_arm_handover_robosuite": (".robosuite_handover", "RobosuiteHandoverEnv"),
    "two_arm_lift_robosuite": (".robosuite_two_arm_lift", "RobosuiteTwoArmLiftEnv"),
    "franka_libero_pick_place_low_level": (".libero", "FrankaLiberoPickPlace"),
    "franka_libero_open_microwave_low_level": (".libero", "FrankaLiberoOpenMicrowave"),
    "franka_libero_pick_alphabet_soup_low_level": (".libero", "FrankaLiberoPickAlphabetSoup"),
    "r1pro_b1k_low_level": (".r1pro_b1k", "R1ProBehaviourLowLevel"),
    "robotwin_low_level": (".robotwin", "RoboTwinEnv"),
}

for _name, (_module, _class_name) in _ENV_SPECS.items():
    register_env(_name, _lazy_env(_module, _class_name))

# Preserve the generated LIBERO registry without importing LIBERO at package
# import time. Existing suites use at most ten task slots in these configs.
for _suite in (
    "libero_10",
    "libero_90",
    "libero_object",
    "libero_object_swap",
    "libero_spatial",
    "libero_goal",
):
    for _task_id in range(10):
        register_env(
            f"franka_libero_{_suite}_{_task_id}_low_level",
            _lazy_env(
                ".libero",
                "FrankaLiberoTask",
                suite_name=_suite,
                task_id=_task_id,
            ),
        )


__all__ = ["list_envs", "register_env"]

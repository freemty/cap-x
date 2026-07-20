"""Dependency-light code-environment registry."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from .base import (
    CodeExecEnvConfig,
    CodeExecutionEnvBase,
    get_config,
    get_exec_env,
    list_configs,
    list_exec_envs,
    register_config,
    register_exec_env,
)


def _lazy_exec_env(module: str, class_name: str) -> Callable[..., Any]:
    def factory(*args: Any, **kwargs: Any) -> Any:
        env_class = getattr(import_module(module, package=__name__), class_name)
        return env_class(*args, **kwargs)

    factory.__name__ = f"lazy_{class_name}"
    return factory


_EXEC_SPECS: dict[str, tuple[str, str, CodeExecEnvConfig]] = {
    "franka_real_code_env": (
        ".franka.franka_pick_place",
        "FrankaPickPlaceCodeEnv",
        CodeExecEnvConfig(low_level="franka_real_low_level", apis=["FrankaControlApi"]),
    ),
    "franka_robosuite_spill_wipe_code_env": (
        ".franka.franka_spill_wipe",
        "FrankaSpillWipeCodeEnv",
        CodeExecEnvConfig(low_level="franka_robosuite_spill_wipe_low_level", apis=["FrankaControlSpillWipePrivilegedApi"]),
    ),
    "franka_pick_place_code_env": (
        ".franka.franka_pick_place",
        "FrankaPickPlaceCodeEnv",
        CodeExecEnvConfig(low_level="franka_cubes_low_level", apis=["FrankaControlPrivilegedApi"]),
    ),
    "franka_robosuite_pick_place_code_env": (
        ".franka.franka_pick_place",
        "FrankaPickPlaceCodeEnv",
        CodeExecEnvConfig(low_level="franka_robosuite_cubes_low_level", apis=["FrankaControlPrivilegedApi"]),
    ),
    "franka_nut_assembly_code_env": (
        ".franka.franka_nut_assembly",
        "FrankaNutAssemblyCodeEnv",
        CodeExecEnvConfig(low_level="franka_robosuite_nut_assembly_low_level", apis=["FrankaControlNutAssemblyPrivilegedApi"], privileged=True),
    ),
    "franka_nut_assembly_code_env_visual": (
        ".franka.franka_nut_assembly",
        "FrankaNutAssemblyCodeEnv",
        CodeExecEnvConfig(low_level="franka_robosuite_nut_assembly_low_level_visual", apis=["FrankaControlNutAssemblyVisualApi"], privileged=False),
    ),
    "franka_pick_place_multi_code_env": (
        ".franka.franka_pick_place",
        "FrankaPickPlaceCodeEnv",
        CodeExecEnvConfig(low_level="franka_cubes_low_level", apis=["FrankaControlMultiPrivilegedApi"]),
    ),
    "franka_lift_code_env": (
        ".franka.franka_lift",
        "FrankaLiftCodeEnv",
        CodeExecEnvConfig(low_level="franka_robosuite_cube_lift_low_level", apis=["FrankaControlPrivilegedApi"]),
    ),
    "two_arm_handover_code_env": (
        ".franka.two_arm_handover",
        "TwoArmHandoverCodeEnv",
        CodeExecEnvConfig(low_level="two_arm_handover_robosuite", apis=["FrankaHandoverApi"]),
    ),
    "franka_libero_code_env": (
        ".franka.franka_libero_env",
        "FrankaLiberoCodeEnv",
        CodeExecEnvConfig(low_level="franka_libero_low_level", apis=["FrankaLiberoApi"]),
    ),
    "franka_restack_code_env": (
        ".franka.franka_cube_restack",
        "FrankaRestackCodeEnv",
        CodeExecEnvConfig(low_level="franka_robosuite_cubes_restack_low_level", apis=["FrankaControlPrivilegedApi"]),
    ),
    "r1pro_radio_code_env": (
        ".r1pro.r1pro_pickup_radio",
        "R1ProRadioCodeEnv",
        CodeExecEnvConfig(low_level="r1pro_b1k_low_level", apis=["R1ProControlApi"]),
    ),
    "r1pro_trash_code_env": (
        ".r1pro.r1pro_pickup_trash",
        "R1ProTrashCodeEnv",
        CodeExecEnvConfig(low_level="r1pro_b1k_low_level", apis=["R1ProControlApi"]),
    ),
}

for _name, (_module, _class_name, _config) in _EXEC_SPECS.items():
    register_exec_env(_name, _lazy_exec_env(_module, _class_name))
    register_config(_name, _config)


__all__ = [
    "CodeExecEnvConfig",
    "CodeExecutionEnvBase",
    "get_config",
    "get_exec_env",
    "list_configs",
    "list_exec_envs",
    "register_config",
    "register_exec_env",
]

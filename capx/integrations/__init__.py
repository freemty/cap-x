"""Dependency-light API registry.

Simulator stacks in CaP-X are intentionally heterogeneous.  Importing every
Franka, LIBERO, and R1Pro dependency just to construct a RoboTwin API made the
registry itself a cross-embodiment dependency bottleneck.  Registry entries are
therefore lazy: the concrete module is imported only when that API is selected.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from .base_api import list_apis, register_api


def _lazy_api(module: str, class_name: str, **kwargs: Any) -> Callable[[Any], Any]:
    def factory(env: Any) -> Any:
        api_class = getattr(import_module(module, package=__name__), class_name)
        return api_class(env, **kwargs)

    factory.__name__ = f"lazy_{class_name}"
    return factory


_API_SPECS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "FrankaControlPrivilegedApi": (".franka.control_privileged", "FrankaControlPrivilegedApi", {}),
    "FrankaControlApi": (".franka.control", "FrankaControlApi", {"use_sam3": True}),
    "FrankaControlApiReduced": (".franka.control_reduced", "FrankaControlApiReduced", {}),
    "FrankaControlApiReducedBimanual": (".franka.control_reduced", "FrankaControlApiReduced", {"bimanual": True}),
    "FrankaControlApiReducedExamplelessBimanual": (".franka.control_reduced_exampleless", "FrankaControlApiReducedExampleless", {"bimanual": True}),
    "FrankaControlApiReducedBimanualHandover": (".franka.control_reduced", "FrankaControlApiReduced", {"bimanual": True, "is_handover": True}),
    "FrankaControlApiReducedExamplelessBimanualHandover": (".franka.control_reduced_exampleless", "FrankaControlApiReducedExampleless", {"bimanual": True, "is_handover": True}),
    "FrankaControlApiReducedSpillWipe": (".franka.control_reduced", "FrankaControlApiReduced", {"tcp_offset": [0.0, 0.0, -0.0158]}),
    "FrankaControlApiReducedExampleless": (".franka.control_reduced_exampleless", "FrankaControlApiReducedExampleless", {}),
    "FrankaControlApiReducedSkillLibrary": (".franka.control_reduced_skill_library", "FrankaControlApiReducedSkillLibrary", {}),
    "FrankaControlApiReducedSkillLibraryBimanual": (".franka.control_reduced_skill_library", "FrankaControlApiReducedSkillLibrary", {"bimanual": True}),
    "FrankaControlApiReducedSkillLibrarySpillWipe": (".franka.control_reduced_skill_library", "FrankaControlApiReducedSkillLibrary", {"tcp_offset": [0.0, 0.0, -0.0158]}),
    "FrankaControlApiReducedSkillLibraryBimanualHandover": (".franka.control_reduced_skill_library", "FrankaControlApiReducedSkillLibrary", {"bimanual": True, "is_handover": True}),
    "FrankaControlSpillWipeApi": (".franka.spill_wipe", "FrankaControlSpillWipeApi", {"tcp_offset": [0.0, 0.0, -0.0158], "use_sam3": True}),
    "FrankaControlSpillWipeApiReduced": (".franka.control_reduced", "FrankaControlApiReduced", {"tcp_offset": [0.0, 0.0, -0.0158], "is_spill_wipe": True}),
    "FrankaControlSpillWipePrivilegedApi": (".franka.spill_wipe_privileged", "FrankaControlSpillWipePrivilegedApi", {"tcp_offset": [0.0, 0.0, -0.0158]}),
    "FrankaControlSpillWipeApiReducedExampleless": (".franka.control_reduced_exampleless", "FrankaControlApiReducedExampleless", {"tcp_offset": [0.0, 0.0, -0.0158], "is_spill_wipe": True}),
    "FrankaHandoverPrivilegedApi": (".franka.handover_privileged", "FrankaHandoverPrivilegedApi", {}),
    "FrankaHandoverApi": (".franka.handover", "FrankaHandoverApi", {}),
    "FrankaHandoverApiReduced": (".franka.handover_reduced", "FrankaHandoverApiReduced", {}),
    "FrankaHandoverApiReducedExampleless": (".franka.handover_reduced_exampleless", "FrankaHandoverApiReducedExampleless", {}),
    "FrankaTwoArmLiftApi": (".franka.two_arm_lift", "FrankaTwoArmLiftApi", {}),
    "FrankaTwoArmLiftPrivilegedApi": (".franka.two_arm_lift_privileged", "FrankaTwoArmLiftPrivilegedApi", {}),
    "FrankaTwoArmLiftApiReduced": (".franka.control_reduced", "FrankaControlApiReduced", {"bimanual": True, "use_sam3": False}),
    "FrankaTwoArmLiftApiReducedExampleless": (".franka.control_reduced_exampleless", "FrankaControlApiReducedExampleless", {"bimanual": True, "use_sam3": False}),
    "FrankaControlNutAssemblyPrivilegedApi": (".franka.nut_assembly_privileged", "FrankaControlNutAssemblyPrivilegedApi", {}),
    "FrankaControlNutAssemblyVisualApi": (".franka.nut_assembly_visual", "FrankaControlNutAssemblyVisualApi", {}),
    "FrankaControlNutAssemblyApiReduced": (".franka.control_reduced", "FrankaControlApiReduced", {"is_peg_assembly": True}),
    "FrankaControlNutAssemblyApiReducedExampleless": (".franka.control_reduced_exampleless", "FrankaControlApiReducedExampleless", {"is_peg_assembly": True}),
    "FrankaControlMultiPrivilegedApi": (".franka.control_privileged", "FrankaControlPrivilegedApi", {"multi_turn": True}),
    "FrankaRealReducedSkillLibraryControlApi": (".franka.control_reduced_skill_library", "FrankaControlApiReducedSkillLibrary", {"tcp_offset": [0.0, 0.0, -0.157], "real": True}),
    "FrankaRealControlApi": (".franka.control", "FrankaControlApi", {"tcp_offset": [0.0, 0.0, -0.157], "real": True}),
    "R1ProControlApi": (".r1pro.control", "R1ProControlApi", {"use_sam3": True}),
    "FrankaLiberoPrivilegedApi": (".franka.libero_privileged", "FrankaLiberoPrivilegedApi", {}),
    "FrankaLiberoApi": (".franka.libero", "FrankaLiberoApi", {"use_sam3": True}),
    "FrankaLiberoApiReduced": (".franka.libero_reduced", "FrankaLiberoApiReduced", {}),
    "FrankaLiberoApiReducedSkillLibrary": (".franka.libero_reduced_skill_library", "FrankaLiberoApiReducedSkillLibrary", {}),
    "RoboTwinPrivilegedApi": (".robotwin", "RoboTwinPrivilegedApi", {}),
}

for _name, (_module, _class_name, _kwargs) in _API_SPECS.items():
    register_api(_name, _lazy_api(_module, _class_name, **_kwargs))


__all__ = ["list_apis", "register_api"]

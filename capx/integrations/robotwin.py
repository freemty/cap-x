from __future__ import annotations

from typing import Any

from capx.integrations.base_api import ApiBase


class RoboTwinPrivilegedApi(ApiBase):
    """High-level, planner-backed RoboTwin primitives for generated code."""

    def functions(self) -> dict[str, Any]:
        return {
            "list_actors": self.list_actors,
            "get_actor_pose": self.get_actor_pose,
            "get_functional_point": self.get_functional_point,
            "choose_arm": self.choose_arm,
            "grasp_actor": self.grasp_actor,
            "place_actor": self.place_actor,
            "move_by_displacement": self.move_by_displacement,
            "back_to_origin": self.back_to_origin,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "get_task_status": self.get_task_status,
        }

    def list_actors(self) -> list[str]:
        """Return actor names available to the generated program."""
        return self._env.actor_names()

    def get_actor_pose(self, name: str) -> list[float]:
        """Return an actor pose as ``[x, y, z, qw, qx, qy, qz]``.

        Args:
            name: Actor attribute name returned by :func:`list_actors`.
        """
        return self._env.actor_pose(name)

    def get_functional_point(self, name: str, index: int = 1) -> list[float]:
        """Return a task-defined functional pose for an actor.

        Args:
            name: Actor attribute name.
            index: Functional point index defined by the RoboTwin asset.
        """
        return self._env.functional_point(name, index=index)

    def choose_arm(self, name: str) -> str:
        """Choose the left or right arm from the actor's current x position."""
        return self._env.choose_arm(name)

    def grasp_actor(self, name: str, arm: str = "auto", pre_grasp_dis: float = 0.09) -> bool:
        """Plan and execute a grasp.

        Args:
            name: Actor attribute name.
            arm: ``left``, ``right``, or ``auto``.
            pre_grasp_dis: Pre-grasp approach distance in metres.
        """
        return self._env.grasp_actor(name, arm=arm, pre_grasp_dis=pre_grasp_dis)

    def place_actor(
        self,
        name: str,
        target_pose: list[float],
        arm: str = "auto",
        functional_point_id: int | None = None,
        pre_dis: float = 0.05,
        dis: float = 0.0,
        pre_dis_axis: str = "fp",
    ) -> bool:
        """Plan and execute a placement.

        Args:
            name: Actor attribute name.
            target_pose: Target ``[x, y, z, qw, qx, qy, qz]`` pose.
            arm: ``left``, ``right``, or ``auto``.
            functional_point_id: Actor functional point to align at the target.
            pre_dis: Pre-place approach distance in metres.
            dis: Final offset from the target in metres.
            pre_dis_axis: RoboTwin approach-axis mode, usually ``fp``.
        """
        return self._env.place_actor(
            name,
            target_pose,
            arm=arm,
            functional_point_id=functional_point_id,
            pre_dis=pre_dis,
            dis=dis,
            pre_dis_axis=pre_dis_axis,
        )

    def move_by_displacement(
        self,
        arm: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> bool:
        """Move one end effector by a world-frame displacement in metres."""
        return self._env.move_by_displacement(arm, x=x, y=y, z=z)

    def back_to_origin(self, arm: str) -> bool:
        """Move one arm back to its configured home pose."""
        return self._env.back_to_origin(arm)

    def open_gripper(self, arm: str) -> bool:
        """Open one gripper."""
        return self._env.set_gripper(arm, open_gripper=True)

    def close_gripper(self, arm: str) -> bool:
        """Close one gripper."""
        return self._env.set_gripper(arm, open_gripper=False)

    def get_task_status(self) -> dict[str, Any]:
        """Return planner success, task success, and simulator step count."""
        return self._env.status()


__all__ = ["RoboTwinPrivilegedApi"]

"""Left arm user embodiment."""

from pathlib import Path

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot

from safe_feedback_interpretation.user_embodiments.base_embodiment import UserEmbodiment


class LeftArm(UserEmbodiment[JointPositions], SingleArmPyBulletRobot):
    """Left arm embodiment combining user embodiment interface with PyBullet
    robot."""

    def __init__(
        self,
        physics_client_id: int,
        base_pose: Pose = Pose.identity(),
        control_mode: str = "position",
        home_joint_positions: JointPositions | None = None,
        fixed_base: bool = True,
        custom_urdf_path: Path | None = None,
    ) -> None:
        super().__init__(
            physics_client_id=physics_client_id,
            base_pose=base_pose,
            control_mode=control_mode,
            home_joint_positions=home_joint_positions,
            fixed_base=fixed_base,
            custom_urdf_path=custom_urdf_path,
        )

    # UserEmbodiment abstract methods
    def reset(self, state: JointPositions) -> None:
        """Reset the left arm to the given state."""
        self.set_joints(state)

    def get_state(self) -> JointPositions:
        """Get the current state of the left arm."""
        return self.get_joint_positions()

    # SingleArmPyBulletRobot abstract properties
    @classmethod
    def get_name(cls) -> str:
        return "left_arm"

    @property
    def default_urdf_path(self) -> Path:
        return (
            Path(__file__).parent.parent.parent.parent
            / "assets"
            / "human"
            / "left_arm_6dof_continuous.urdf"
        )

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0] * 6

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"


if __name__ == "__main__":
    import time

    import pybullet as p

    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

    left_arm = LeftArm(
        physics_client_id=physics_client, base_pose=Pose((0, 0, 1), (0, 0, 0, 1))
    )

    try:
        while True:
            time.sleep(0.1)
            print(left_arm.get_state())
            new_state = left_arm.get_state().copy()
            new_state[-1] += 1
            left_arm.reset(new_state)
    except KeyboardInterrupt:
        pass

    p.disconnect(physicsClientId=physics_client)



from safe_feedback_interpretation.avatars.base_avatar import BaseAvatar
from typing import TypeVar, Optional, TypeAlias
from pybullet_helpers.joint import JointPositions

VerbalFeedback: TypeAlias = str
ArmJointState: TypeAlias = JointPositions


class PerfectCommunicator(BaseAvatar[ArmJointState, VerbalFeedback]):
    """An avatar that perfectly communicates its joint state as verbal feedback."""
    def gt_to_feedback(self) -> VerbalFeedback:
        """Convert the ground truth state to feedback."""

    def step(self) -> Optional[VerbalFeedback]:
        """Advance the avatar's state by one time step."""
        return self.gt_to_feedback()
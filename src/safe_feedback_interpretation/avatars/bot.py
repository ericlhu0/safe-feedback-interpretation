"""An avatar that perfectly communicates its joint state as verbal feedback."""

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, TypeAlias

from pybullet_helpers.joint import JointPositions

from safe_feedback_interpretation.avatars.base_avatar import BaseAvatar

VerbalFeedback: TypeAlias = str
ArmJointState: TypeAlias = JointPositions


class ComfortLevel(IntEnum):
    """Comfort levels."""

    UNCOMFORTABLE = 0
    MEDIUM = 1
    COMFORTABLE = 2


@dataclass(frozen=True)
class ComfortThreshold:
    """Comfort thresholds for a single joint."""

    comfortable: Sequence[tuple[float, float]]
    medium: Sequence[tuple[float, float]]


ComfortThresholds: TypeAlias = List[ComfortThreshold]


class PerfectCommunicator(
    BaseAvatar[ArmJointState, VerbalFeedback, ComfortLevel, ComfortThresholds]
):
    """An avatar that perfectly communicates its joint state as verbal
    feedback."""

    def __init__(self, gt_state: ArmJointState) -> None:
        super().__init__(
            gt_state,
            [ComfortThreshold(comfortable=[(30, 150)], medium=[(10, 30), (150, 170)])]
            * len(gt_state),
        )
        assert len(gt_state) == len(self._comfort_thresholds)

    def _state_to_comfort(self, state: ArmJointState) -> ComfortLevel:
        """Convert a joint state to a comfort level based on per-joint
        thresholds."""

        def _classify(angle: float, t: ComfortThreshold) -> ComfortLevel:
            if any(lo <= angle <= hi for lo, hi in t.comfortable):
                return ComfortLevel.COMFORTABLE
            if any(lo <= angle <= hi for lo, hi in t.medium):
                return ComfortLevel.MEDIUM
            return ComfortLevel.UNCOMFORTABLE

        if len(state) != len(self._comfort_thresholds):
            raise ValueError("state and comfort_thresholds must have the same length")
        return min(_classify(pos, t) for pos, t in zip(state, self._comfort_thresholds))

    def _gt_to_feedback(self) -> VerbalFeedback:
        """Convert the ground truth state to feedback."""
        comfort_state = self._state_to_comfort(self._gt_state)
        expressions = {
            ComfortLevel.COMFORTABLE: [
                "comfortable",
            ],
            ComfortLevel.MEDIUM: [
                "medium",
            ],
            ComfortLevel.UNCOMFORTABLE: [
                "uncomfortable",
            ],
        }
        return random.choice(expressions[comfort_state])

    def step(self) -> Optional[VerbalFeedback]:
        """Advance the avatar's state by one time step."""
        return self._gt_to_feedback()


if __name__ == "__main__":
    a = PerfectCommunicator(ArmJointState([0] * 6))
    print(a.step())
    a.update_state(ArmJointState([45] * 6))
    print(a.step())
    a.update_state(ArmJointState([15] * 6))
    print(a.step())

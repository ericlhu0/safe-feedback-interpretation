"""Avatar representing a care recipient."""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

StateT = TypeVar("StateT")
FeedbackT = TypeVar("FeedbackT")
ComfortRepresentationT = TypeVar("ComfortRepresentationT")
ComfortThresholdsT = TypeVar("ComfortThresholdsT")


class BaseAvatar(
    ABC, Generic[StateT, FeedbackT, ComfortRepresentationT, ComfortThresholdsT]
):
    """Base class for a care recipient avatar."""

    def __init__(
        self, gt_state: StateT, comfort_thresholds: ComfortThresholdsT
    ) -> None:
        """Initialize the avatar."""
        self._gt_state = gt_state
        self._comfort_thresholds = comfort_thresholds

    def update_state(self, new_state: StateT) -> None:
        """Update the avatar's ground truth state."""
        self._gt_state = new_state

    @abstractmethod
    def _gt_to_feedback(self) -> FeedbackT:
        """Convert the ground truth state to feedback."""

    @abstractmethod
    def step(self) -> Optional[FeedbackT]:
        """Advance the avatar's state by one time step."""

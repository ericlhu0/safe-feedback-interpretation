"""Avatar representing a care recipient."""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Generic

StateT = TypeVar("StateT")
FeedbackT = TypeVar("FeedbackT")
ComfortRepresentationT = TypeVar("ComfortRepresentationT")

class BaseAvatar(ABC, Generic[StateT, FeedbackT, ComfortRepresentationT]):
    """Base class for a care recipient avatar."""

    def __init__(self, gt_state: StateT) -> None:
        """Initialize the avatar."""
        self._gt_state = gt_state

    def update_state(self, new_state: StateT) -> None:
        """Update the avatar's ground truth state."""
        self._gt_state = new_state

    @abstractmethod
    def gt_to_feedback(self) -> FeedbackT:
        """Convert the ground truth state to feedback."""

    @abstractmethod
    def step() -> Optional[FeedbackT]:
        """Advance the avatar's state by one time step."""
    
        
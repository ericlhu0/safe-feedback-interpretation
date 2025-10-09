"""Base class for user embodiment implementations."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateT = TypeVar("StateT")


class UserEmbodiment(ABC, Generic[StateT]):
    """Abstract base class for user embodiment implementations.

    Generic over StateT, which represents the state type for this
    embodiment.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abstractmethod
    def reset(self, state: StateT) -> None:
        """Reset the embodiment to the given state."""

    @abstractmethod
    def get_state(self) -> StateT:
        """Get the current state of the embodiment."""

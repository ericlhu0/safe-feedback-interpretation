"""Abstract base classes for planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar


StateT = TypeVar("StateT")
ControlT = TypeVar("ControlT")
GoalT = TypeVar("GoalT")


class BasePlanner(ABC, Generic[StateT, ControlT, GoalT]):
    """Base interface for planners operating on state-control spaces."""

    @abstractmethod
    def plan(self, state: StateT, goal: GoalT) -> Tuple[float, Tuple[ControlT, ...]]:
        """Return optimal cost and control sequence from a given state toward a goal."""

    def solve(self, state: StateT, goal: GoalT) -> ControlT:
        """Return the first control of the optimal plan."""
        cost, sequence = self.plan(state, goal)
        if not sequence:
            raise RuntimeError(
                "plan must return at least one control action; got empty sequence"
            )
        return sequence[0]

    def rollout(
        self, state: StateT, goal: GoalT, steps: int
    ) -> Tuple[StateT, Tuple[ControlT, ...]]:
        """Apply the planner in closed loop for a number of steps."""
        if steps < 1:
            raise ValueError("steps must be at least 1")

        controls: list[ControlT] = []
        current_state = state
        for _ in range(steps):
            control = self.solve(current_state, goal)
            controls.append(control)
            current_state = self.step(current_state, control)

        return current_state, tuple(controls)

    @abstractmethod
    def step(self, state: StateT, control: ControlT) -> StateT:
        """Advance the environment dynamics with the provided control."""


__all__ = ["BasePlanner", "StateT", "ControlT", "GoalT"]

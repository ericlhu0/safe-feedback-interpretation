"""Toy model-predictive controller for a planar car model."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from .base_planner import BasePlanner


@dataclass(frozen=True)
class PlanarCarState:
    """State of a point-mass car moving on a plane."""

    x: float
    y: float
    heading: float
    speed: float

    def distance_to(self, target_x: float, target_y: float) -> float:
        """Euclidean distance to a target point."""
        return math.hypot(self.x - target_x, self.y - target_y)


@dataclass(frozen=True)
class PlanarCarControl:
    """Control inputs for the planar car."""

    acceleration: float
    yaw_rate: float


@dataclass(frozen=True)
class PlanarCarGoal:
    """Goal specification for the planar car."""

    x: float
    y: float
    target_speed: float = 0.0
    heading: float | None = None


@dataclass(frozen=True)
class PlanarCarConfig:
    """Configuration parameters for the planar car dynamics."""

    dt: float = 0.2
    max_speed: float = 5.0
    max_acceleration: float = 2.0
    max_yaw_rate: float = math.pi / 4.0
    drag_coefficient: float = 0.1
    obstacle: CircularObstacle | None = None


@dataclass(frozen=True)
class MPCWeights:
    """Cost weights used by the model-predictive controller."""

    position: float = 1.0
    heading: float = 0.2
    speed: float = 0.1
    control: float = 0.05
    terminal_position: float = 2.0
    terminal_heading: float = 0.5
    terminal_speed: float = 0.2
    obstacle: float = 100.0


@dataclass(frozen=True)
class CircularObstacle:
    """Circular obstacle used for soft collision avoidance."""

    center_x: float
    center_y: float
    radius: float
    buffer: float = 0.4


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value to a closed interval."""
    return max(lower, min(value, upper))


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    # Handle the edge case where modulo returns pi.
    if wrapped == -math.pi:
        return math.pi
    return wrapped


def _bearing_toward_goal(state: PlanarCarState, goal: PlanarCarGoal) -> float:
    """Compute the desired bearing from the current state to the goal."""
    dx = goal.x - state.x
    dy = goal.y - state.y
    if math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9):
        return state.heading
    return math.atan2(dy, dx)


class PlanarCarDynamics:
    """Simple non-holonomic planar car dynamics."""

    def __init__(self, config: PlanarCarConfig | None = None) -> None:
        self.config = config or PlanarCarConfig()

    def step(self, state: PlanarCarState, control: PlanarCarControl) -> PlanarCarState:
        """Advance the state by one time step using the provided control."""
        cfg = self.config

        acceleration = _clamp(
            control.acceleration, -cfg.max_acceleration, cfg.max_acceleration
        )
        yaw_rate = _clamp(control.yaw_rate, -cfg.max_yaw_rate, cfg.max_yaw_rate)

        speed = state.speed + cfg.dt * (
            acceleration - cfg.drag_coefficient * state.speed
        )
        speed = _clamp(speed, 0.0, cfg.max_speed)

        heading = _wrap_angle(state.heading + yaw_rate * cfg.dt)

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)

        x = state.x + vx * cfg.dt
        y = state.y + vy * cfg.dt

        return PlanarCarState(x=x, y=y, heading=heading, speed=speed)


class PlanarCarMPC(BasePlanner[PlanarCarState, PlanarCarControl, PlanarCarGoal]):
    """Sampling-based model predictive controller for the planar car."""

    def __init__(
        self,
        dynamics: PlanarCarDynamics,
        horizon: int = 3,
        acceleration_grid: Sequence[float] | None = None,
        yaw_rate_grid: Sequence[float] | None = None,
        weights: MPCWeights | None = None,
        *,
        num_samples: int = 256,
        seed: int | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        self.dynamics = dynamics
        self.horizon = horizon
        self.weights = weights or MPCWeights()
        self._num_samples = num_samples
        self._rng = random.Random(seed)
        self._obstacle = self.dynamics.config.obstacle

        cfg = self.dynamics.config
        acceleration_options = (
            acceleration_grid
            if acceleration_grid is not None
            else (-cfg.max_acceleration, 0.0, cfg.max_acceleration)
        )
        yaw_rate_options = (
            yaw_rate_grid
            if yaw_rate_grid is not None
            else (-cfg.max_yaw_rate, 0.0, cfg.max_yaw_rate)
        )

        self._control_grid = tuple(
            PlanarCarControl(acceleration=a, yaw_rate=w)
            for a in acceleration_options
            for w in yaw_rate_options
        )

        if not self._control_grid:
            raise ValueError("control grid must contain at least one control input")

        self._deterministic_sequences = self._build_seeding_sequences()

    def plan(
        self, state: PlanarCarState, goal: PlanarCarGoal
    ) -> Tuple[float, Tuple[PlanarCarControl, ...]]:
        """Return the optimal cost and control sequence for the current
        horizon."""
        best_cost = math.inf
        best_sequence: Tuple[PlanarCarControl, ...] | None = None

        for cost, sequence in self.iter_plans(state, goal):
            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence

        if best_sequence is None:
            raise RuntimeError("no control sequence evaluated")

        return best_cost, best_sequence

    def iter_plans(
        self, state: PlanarCarState, goal: PlanarCarGoal
    ) -> Iterable[Tuple[float, Tuple[PlanarCarControl, ...]]]:
        """Yield sampled plan candidates (cost, controls) in evaluation
        order."""
        for sequence in self._candidate_sequences():
            cost = self._evaluate_sequence(state, goal, sequence)
            yield cost, sequence

    def _candidate_sequences(self) -> Iterable[Tuple[PlanarCarControl, ...]]:
        """Generate candidate control sequences via deterministic seeding and
        sampling."""
        yield from self._deterministic_sequences

        seen = {self._sequence_key(seq) for seq in self._deterministic_sequences}

        for _ in range(self._num_samples):
            sequence = tuple(
                self._rng.choice(self._control_grid) for _ in range(self.horizon)
            )
            key = self._sequence_key(sequence)
            if key in seen:
                continue
            seen.add(key)
            yield sequence

    def _build_seeding_sequences(self) -> Tuple[Tuple[PlanarCarControl, ...], ...]:
        """Create a small set of deterministic seed sequences."""
        if not self._control_grid:
            return tuple()

        zero_control = min(
            self._control_grid,
            key=lambda ctrl: abs(ctrl.acceleration) + abs(ctrl.yaw_rate),
        )
        accel_control = max(
            self._control_grid,
            key=lambda ctrl: ctrl.acceleration,
        )
        brake_control = min(
            self._control_grid,
            key=lambda ctrl: ctrl.acceleration,
        )
        left_turn = max(
            self._control_grid,
            key=lambda ctrl: ctrl.yaw_rate,
        )
        right_turn = min(
            self._control_grid,
            key=lambda ctrl: ctrl.yaw_rate,
        )

        sequences = [
            tuple(zero_control for _ in range(self.horizon)),
            tuple(accel_control for _ in range(self.horizon)),
            tuple(brake_control for _ in range(self.horizon)),
            tuple(left_turn for _ in range(self.horizon)),
            tuple(right_turn for _ in range(self.horizon)),
        ]

        return tuple(sequences)

    @staticmethod
    def _sequence_key(
        sequence: Tuple[PlanarCarControl, ...],
    ) -> Tuple[Tuple[float, float], ...]:
        """Return a hashable key for a control sequence."""
        return tuple((ctrl.acceleration, ctrl.yaw_rate) for ctrl in sequence)

    def _evaluate_sequence(
        self,
        start_state: PlanarCarState,
        goal: PlanarCarGoal,
        sequence: Tuple[PlanarCarControl, ...],
    ) -> float:
        """Evaluate the cumulative cost of a candidate control sequence."""
        state = start_state
        total_cost = 0.0
        for control in sequence:
            state = self.dynamics.step(state, control)
            total_cost += self._stage_cost(state, control, goal)

        total_cost += self._terminal_cost(state, goal)
        return total_cost

    def _stage_cost(
        self, state: PlanarCarState, control: PlanarCarControl, goal: PlanarCarGoal
    ) -> float:
        """Compute the cost incurred at a single time step."""
        weights = self.weights

        position_error = state.distance_to(goal.x, goal.y)
        position_cost = weights.position * (position_error**2)

        desired_heading = (
            goal.heading
            if goal.heading is not None
            else _bearing_toward_goal(state, goal)
        )
        heading_error = _wrap_angle(state.heading - desired_heading)
        heading_cost = weights.heading * (heading_error**2)

        speed_error = state.speed - goal.target_speed
        speed_cost = weights.speed * (speed_error**2)

        control_cost = weights.control * (control.acceleration**2 + control.yaw_rate**2)
        obstacle_cost = weights.obstacle * self._obstacle_penalty(state)

        return position_cost + heading_cost + speed_cost + control_cost + obstacle_cost

    def _terminal_cost(self, state: PlanarCarState, goal: PlanarCarGoal) -> float:
        """Compute the cost applied at the terminal state."""
        weights = self.weights

        position_error = state.distance_to(goal.x, goal.y)
        desired_heading = (
            goal.heading
            if goal.heading is not None
            else _bearing_toward_goal(state, goal)
        )
        heading_error = _wrap_angle(state.heading - desired_heading)
        speed_error = state.speed - goal.target_speed

        return (
            weights.terminal_position * (position_error**2)
            + weights.terminal_heading * (heading_error**2)
            + weights.terminal_speed * (speed_error**2)
            + weights.obstacle * self._obstacle_penalty(state)
        )

    def _obstacle_penalty(self, state: PlanarCarState) -> float:
        """Soft penalty for approaching the circular obstacle."""
        obstacle = self._obstacle
        if obstacle is None:
            return 0.0

        distance = state.distance_to(obstacle.center_x, obstacle.center_y)
        margin = max(0.0, obstacle.radius + obstacle.buffer)
        violation = margin - distance
        if violation <= 0.0:
            return 0.0

        return violation**2

    def step(self, state: PlanarCarState, control: PlanarCarControl) -> PlanarCarState:
        """Advance the system dynamics with the planner's embedded model."""
        return self.dynamics.step(state, control)


__all__ = [
    "PlanarCarState",
    "PlanarCarControl",
    "PlanarCarGoal",
    "PlanarCarConfig",
    "CircularObstacle",
    "PlanarCarDynamics",
    "MPCWeights",
    "PlanarCarMPC",
]

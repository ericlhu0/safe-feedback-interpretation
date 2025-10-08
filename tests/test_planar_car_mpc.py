"""Tests for the planar car model predictive controller."""

import math

from safe_feedback_interpretation.planners import (
    MPCWeights,
    PlanarCarConfig,
    PlanarCarDynamics,
    PlanarCarGoal,
    PlanarCarMPC,
    PlanarCarState,
)


def test_planar_car_mpc_progresses_toward_goal() -> None:
    """The controller should reduce distance to a straight-ahead goal."""
    config = PlanarCarConfig(
        dt=0.2,
        max_speed=3.0,
        max_acceleration=1.0,
        max_yaw_rate=math.pi / 6.0,
    )
    dynamics = PlanarCarDynamics(config)
    mpc = PlanarCarMPC(dynamics=dynamics, horizon=3, num_samples=512, seed=7)

    initial_state = PlanarCarState(x=0.0, y=0.0, heading=0.0, speed=0.0)
    goal = PlanarCarGoal(x=3.0, y=0.0, target_speed=0.0)

    final_state, controls = mpc.rollout(initial_state, goal, steps=6)

    assert len(controls) == 6
    assert final_state.x > initial_state.x
    assert final_state.distance_to(goal.x, goal.y) < 2.5
    assert final_state.speed <= config.max_speed + 1e-6


def test_planar_car_mpc_turns_toward_off_axis_goal() -> None:
    """The controller should steer toward an off-axis target."""
    config = PlanarCarConfig(
        dt=0.2,
        max_speed=2.5,
        max_acceleration=1.0,
        max_yaw_rate=math.pi / 6.0,
    )
    dynamics = PlanarCarDynamics(config)
    weights = MPCWeights(position=1.0, heading=0.5, control=0.01)
    mpc = PlanarCarMPC(
        dynamics=dynamics,
        horizon=3,
        weights=weights,
        num_samples=512,
        seed=11,
    )

    initial_state = PlanarCarState(x=0.0, y=0.0, heading=0.0, speed=0.2)
    goal = PlanarCarGoal(x=1.0, y=1.5, target_speed=0.5)

    first_control = mpc.solve(initial_state, goal)
    assert first_control.yaw_rate > 0.0

    next_state = dynamics.step(initial_state, first_control)
    assert next_state.y > initial_state.y


def test_planar_car_mpc_respects_control_grid() -> None:
    """The control grid should be fully honored during planning."""
    config = PlanarCarConfig(dt=0.1, max_acceleration=2.0, max_yaw_rate=math.pi / 3.0)
    dynamics = PlanarCarDynamics(config)

    custom_grid = (-0.5, 0.0, 0.4)
    yaw_grid = (-0.2, 0.0, 0.3)
    mpc = PlanarCarMPC(
        dynamics=dynamics,
        horizon=2,
        acceleration_grid=custom_grid,
        yaw_rate_grid=yaw_grid,
        num_samples=256,
        seed=5,
    )

    initial_state = PlanarCarState(x=0.0, y=0.0, heading=0.0, speed=0.0)
    goal = PlanarCarGoal(x=0.5, y=0.0, target_speed=0.1)

    first_control = mpc.solve(initial_state, goal)
    # Ensure the selected control belongs to the grid exactly.
    assert first_control.acceleration in custom_grid
    assert first_control.yaw_rate in yaw_grid

"""Tests for the MuJoCo torque-based MPC."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from safe_feedback_interpretation.planners import (
    MujocoArmDynamics,
    MujocoArmState,
    MujocoGoal,
    MujocoTorqueMPC,
)

_SLIDER_XML = """
<mujoco model="slider">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 0" timestep="0.01"/>
  <default>
    <joint limited="true" range="-0.3 0.3"/>
    <geom type="box" size="0.02 0.02 0.02" density="100"/>
    <motor ctrlrange="-5 5"/>
  </default>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.03 0.03 0.03" contype="0" conaffinity="0"/>
      <body name="slider" pos="0 0 0">
        <joint name="slider_joint" type="slide" axis="1 0 0"/>
        <geom name="slider_geom"/>
        <site name="end_effector" pos="0 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider_joint" gear="1"/>
  </actuator>
</mujoco>
"""


def _site_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def _site_position(
    model: mujoco.MjModel, state: MujocoArmState, site_id: int
) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[: model.nq] = state.qpos
    data.qvel[: model.nv] = state.qvel
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()


def test_mujoco_torque_mpc_drives_slider_toward_goal() -> None:
    """Test that MPC drives slider toward goal position."""
    model = mujoco.MjModel.from_xml_string(_SLIDER_XML)
    dynamics = MujocoArmDynamics(model, substeps=5)
    mpc = MujocoTorqueMPC(
        model=model,
        dynamics=dynamics,
        end_effector_site="end_effector",
        horizon=4,
        num_samples=64,
        seed=3,
        torque_sample_std=1.5,
    )

    initial_state = MujocoArmState.from_data(model, mujoco.MjData(model))
    site_id = _site_id(model, "end_effector")
    initial_pos = _site_position(model, initial_state, site_id)

    goal = MujocoGoal(position=np.array([0.12, 0.0, 0.0]))

    final_state, controls = mpc.rollout(initial_state, goal, steps=35)

    assert len(controls) == 35

    final_pos = _site_position(model, final_state, site_id)
    assert final_pos[0] > initial_pos[0] + 0.04
    print(abs(final_pos[0] - goal.position[0]))
    assert abs(final_pos[0] - goal.position[0]) < 0.05


def test_mujoco_torque_mpc_controls_stay_within_limits() -> None:
    """Test that MPC respects control limits."""
    model = mujoco.MjModel.from_xml_string(_SLIDER_XML)
    dynamics = MujocoArmDynamics(model)
    mpc = MujocoTorqueMPC(
        model=model,
        dynamics=dynamics,
        end_effector_site="end_effector",
        horizon=3,
        num_samples=32,
        seed=1,
        torque_sample_std=2.0,
    )

    initial_state = MujocoArmState.from_data(model, mujoco.MjData(model))
    goal = MujocoGoal(position=np.array([0.1, 0.0, 0.0]))

    _, controls = mpc.rollout(initial_state, goal, steps=2)

    ctrl_range = model.actuator_ctrlrange[0]
    lower, upper = ctrl_range
    for control in controls:
        assert control.torques.shape == (model.nu,)
        assert np.all(control.torques >= lower - 1e-6)
        assert np.all(control.torques <= upper + 1e-6)

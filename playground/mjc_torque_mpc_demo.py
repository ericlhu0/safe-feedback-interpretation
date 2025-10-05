"""Interactive MuJoCo demo that drives the Panda arm toward a moving goal using MPC."""

from __future__ import annotations

import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from safe_feedback_interpretation.planners import (
    MujocoArmDynamics,
    MujocoArmState,
    MujocoGoal,
    MujocoTorqueMPC,
)


def _load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        script_dir, "mujoco_menagerie", "franka_emika_panda", "panda_nohand.xml"
    )

    with open(model_path, "r", encoding="utf-8") as handle:
        xml_content = handle.read()

    extra_bodies = """
            <body name="target" pos="0.35 0.25 0.75" mocap="true">
                <geom type="box" size="0.02 0.02 0.02" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
            </body>
    """

    xml_content = xml_content.replace("</worldbody>", extra_bodies + "  </worldbody>")

    model_dir = os.path.dirname(model_path)
    old_cwd = os.getcwd()
    os.chdir(model_dir)
    try:
        model = mujoco.MjModel.from_xml_string(xml_content)
    finally:
        os.chdir(old_cwd)

    data = mujoco.MjData(model)
    return model, data


def simulation_loop() -> None:
    model, data = _load_model()
    dynamics = MujocoArmDynamics(model, substeps=5)
    planner = MujocoTorqueMPC(
        model=model,
        dynamics=dynamics,
        end_effector_site="attachment_site",
        horizon=8,
        num_samples=96,
        seed=4,
        torque_sample_std=8.0,
    )

    target_pos = np.array([0.35, 0.25, 0.75], dtype=np.float64)
    step_size = 0.02

    lower_bounds = np.array([0.1, -0.2, 0.5])
    upper_bounds = np.array([0.6, 0.3, 0.9])

    print("\nControls:")
    print("w/s: forward/backward (x-axis)")
    print("a/d: left/right (y-axis)")
    print("q/e: up/down (z-axis)")
    print("ESC: quit")

    def key_callback(keycode: int) -> None:
        nonlocal target_pos
        if keycode in (ord("w"), ord("W")):
            target_pos[0] += step_size
        elif keycode in (ord("s"), ord("S")):
            target_pos[0] -= step_size
        elif keycode in (ord("a"), ord("A")):
            target_pos[1] += step_size
        elif keycode in (ord("d"), ord("D")):
            target_pos[1] -= step_size
        elif keycode in (ord("q"), ord("Q")):
            target_pos[2] += step_size
        elif keycode in (ord("e"), ord("E")):
            target_pos[2] -= step_size
        target_pos = np.clip(target_pos, lower_bounds, upper_bounds)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            data.mocap_pos[0] = target_pos

            state = MujocoArmState.from_data(model, data)
            goal = MujocoGoal(position=target_pos)
            control = planner.solve(state, goal)

            torques = planner.clamp_torques(control.torques)
            data.ctrl[: model.nu] = torques

            for _ in range(dynamics.substeps):
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    simulation_loop()

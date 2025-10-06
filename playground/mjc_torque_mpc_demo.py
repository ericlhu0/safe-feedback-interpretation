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


def add_obstacle(
    pos: list[float] | np.ndarray,
    size: list[float] | np.ndarray = [0.05, 0.05, 0.05],
    shape: str = "box",
    rgba: list[float] | np.ndarray = [1, 0, 0, 0.8],
    name: str | None = None,
) -> str:
    """Generate XML string for an obstacle.

    Args:
        pos: Position [x, y, z] of the obstacle
        size: Size of the obstacle (interpretation depends on shape)
        shape: Geometry type ("box", "sphere", "cylinder", "capsule")
        rgba: Color and transparency [r, g, b, a]
        name: Optional name for the obstacle body

    Returns:
        XML string for the obstacle
    """
    if name is None:
        name = f"obstacle_{np.random.randint(100000)}"

    pos_str = " ".join(map(str, pos))
    size_str = " ".join(map(str, size))
    rgba_str = " ".join(map(str, rgba))

    return f"""
            <body name="{name}" pos="{pos_str}">
                <geom type="{shape}" size="{size_str}" rgba="{rgba_str}"/>
            </body>
    """


def _load_model(obstacles: list[dict] | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        script_dir, "mujoco_menagerie", "franka_emika_panda", "panda_nohand.xml"
    )

    with open(model_path, "r", encoding="utf-8") as handle:
        xml_content = handle.read()

    # Add SMPL mesh asset
    smpl_mesh_path = os.path.join(script_dir, "smpl", "smpl.obj")
    smpl_asset = f'<mesh name="smpl_human" file="{smpl_mesh_path}"/>'
    xml_content = xml_content.replace("</asset>", smpl_asset + "\n  </asset>")

    extra_bodies = """
            <body name="target" pos="0.35 0.25 0.75" mocap="true">
                <geom type="box" size="0.02 0.02 0.02" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="human" pos="0.5 0.0 0.25">
                <geom type="mesh" mesh="smpl_human" rgba="0.8 0.6 0.4 1"/>
            </body>
            <body name="x_axis" pos="0.25 0 0">
                <geom type="cylinder" size="0.005 0.25" rgba="1 0 0 1" contype="0" conaffinity="0" euler="0 1.5708 0"/>
            </body>
            <body name="y_axis" pos="0 0.25 0">
                <geom type="cylinder" size="0.005 0.25" rgba="0 1 0 1" contype="0" conaffinity="0" euler="1.5708 0 0"/>
            </body>
            <body name="z_axis" pos="0 0 0.25">
                <geom type="cylinder" size="0.005 0.25" rgba="0 0 1 1" contype="0" conaffinity="0"/>
            </body>
    """

    # Add obstacles if provided
    if obstacles:
        for obs in obstacles:
            extra_bodies += add_obstacle(**obs)

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


def simulation_loop(obstacles: list[dict] | None = None) -> None:
    """Run the MPC simulation loop.

    Args:
        obstacles: Optional list of obstacle specifications. Each obstacle is a dict with keys:
            - pos: [x, y, z] position (required)
            - size: [sx, sy, sz] size (optional, default [0.05, 0.05, 0.05])
            - shape: "box", "sphere", "cylinder", or "capsule" (optional, default "box")
            - rgba: [r, g, b, a] color (optional, default [1, 0, 0, 0.8])
            - name: obstacle name (optional, auto-generated if not provided)

    Example:
        obstacles = [
            {"pos": [0.4, 0.0, 0.6], "size": [0.1, 0.1, 0.1]},
            {"pos": [0.3, 0.2, 0.7], "shape": "sphere", "size": [0.05]},
        ]
        simulation_loop(obstacles=obstacles)
    """
    model, data = _load_model(obstacles=obstacles)
    dynamics = MujocoArmDynamics(model, substeps=5)
    planner = MujocoTorqueMPC(
        model=model,
        dynamics=dynamics,
        end_effector_site="attachment_site",
        horizon=16,
        num_samples=96,
        seed=4,
        torque_sample_std=0.75,
    )

    target_pos = np.array([0.75, 0.00, 0.10], dtype=np.float64)
    step_size = 0.02

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
        
        print(f"New target position: {target_pos}")

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

            # Check for collisions in actual simulation
            if data.ncon > 0:
                for i in range(data.ncon):
                    contact = data.contact[i]
                    geom1 = contact.geom1
                    geom2 = contact.geom2

                    # Get body names
                    body1 = model.geom_bodyid[geom1]
                    body2 = model.geom_bodyid[geom2]
                    body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
                    body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)

                    # Get contact force from efc_force (constraint forces)
                    # data.efc_force contains forces for all constraints, contacts come first
                    if i < len(data.efc_force):
                        force_mag = abs(data.efc_force[i])
                    else:
                        force_mag = 0.0

                    print(f"Collision: {body1_name} <-> {body2_name}, force: {force_mag:.4f} N")

            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":

    obstacles = [
        # {"pos": [0.4, 0.0, 0.6], "size": [0.2, 0.2, 0.2]},
        # {"pos": [0.3, 0.2, 0.7], "shape": "sphere", "size": [0.05]},
    ]

    simulation_loop(obstacles=obstacles)

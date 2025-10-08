"""Simulation loop for contact testing with MuJoCo."""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np


def simulation_loop():
    """Run the simulation loop."""
    # Load official Panda model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Use the "no hand" variant because it keeps default collision masks enabled
    model_path = os.path.join(
        script_dir, "mujoco_menagerie/franka_emika_panda/panda_nohand.xml"
    )

    # Load base model
    with open(model_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Insert custom bodies before </worldbody>
    custom_bodies = """
            <body name="block" pos="0.3 0.0 0">
                <geom name="block" type="box" size="0.2 0.2 0.2" rgba="0.7 0.3 0.3 1" contype="1" conaffinity="1"/>
            </body>

            <body name="goal_block" pos="0.3 0.25 0.85" mocap="true">
                <geom name="goal" type="box" size="0.025 0.025 0.025" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
            </body>
"""

    xml_content = xml_content.replace("</worldbody>", custom_bodies + "  </worldbody>")

    # Change to model directory for relative paths
    model_dir = os.path.dirname(model_path)
    old_cwd = os.getcwd()
    os.chdir(model_dir)

    try:
        model = mujoco.MjModel.from_xml_string(xml_content)
    finally:
        os.chdir(old_cwd)
    data = mujoco.MjData(model)

    # Get IDs
    end_effector_id = model.site("attachment_site").id

    # Target position and orientation
    targetPos = np.array([0.3, 0.25, 0.85])
    step_size = 0.05

    print("\nControls:")
    print("w/s: forward/backward (x-axis)")
    print("a/d: left/right (y-axis)")
    print("q/e: up/down (z-axis)")
    print("ESC: quit")

    contact_wrench = np.zeros(6)

    def key_callback(keycode):
        nonlocal targetPos

        if keycode in (ord("w"), ord("W")):
            targetPos[0] += step_size
        elif keycode in (ord("s"), ord("S")):
            targetPos[0] -= step_size
        elif keycode in (ord("a"), ord("A")):
            targetPos[1] += step_size
        elif keycode in (ord("d"), ord("D")):
            targetPos[1] -= step_size
        elif keycode in (ord("q"), ord("Q")):
            targetPos[2] += step_size
        elif keycode in (ord("e"), ord("E")):
            targetPos[2] -= step_size

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            # Update goal block position (mocap body)
            data.mocap_pos[0] = targetPos

            # Simple Jacobian-based IK
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, end_effector_id)

            current_pos = data.site(end_effector_id).xpos.copy()
            error = targetPos - current_pos

            # Compute delta joint angles directly
            J = jacp[:, :7]
            dq = np.linalg.pinv(J) @ error

            # Set target positions (not velocities)
            for i in range(7):
                data.ctrl[i] = data.qpos[i] + dq[i]

            # Step simulation
            mujoco.mj_step(model, data)

            # Check for collisions
            for i in range(data.ncon):
                contact = data.contact[i]
                geom1_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                )
                geom2_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                )

                if geom1_name is None:
                    geom1_body = mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        model.geom_bodyid[contact.geom1],
                    )
                    geom1_name = f"{geom1_body or 'body'}::geom{contact.geom1}"

                if geom2_name is None:
                    geom2_body = mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        model.geom_bodyid[contact.geom2],
                    )
                    geom2_name = f"{geom2_body or 'body'}::geom{contact.geom2}"

                # Skip floor contacts with robot base
                if "floor" in [geom1_name, geom2_name]:
                    continue

                mujoco.mj_contactForce(model, data, i, contact_wrench)
                normal_force = contact_wrench[2]
                tangential_force = np.linalg.norm(contact_wrench[:2])
                total_force = np.linalg.norm(contact_wrench[:3])

                print(
                    f"Collision: {geom1_name} with {geom2_name}, "
                    f"distance: {contact.dist:.4f}, normal: {normal_force:.4f}, "
                    f"tangential: {tangential_force:.4f}, total: {total_force:.4f}"
                )

            # Update viewer
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    simulation_loop()

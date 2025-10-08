"""Simulation loop for contact testing with PyBullet."""

import time

import numpy as np
import pybullet as p  # type: ignore[import-not-found]
import pybullet_data  # type: ignore[import-untyped]


def simulation_loop() -> None:
    """Run the simulation loop."""
    # Connect to PyBullet in GUI mode
    _ = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load plane and Franka Panda robot
    _ = p.loadURDF("plane.urdf")

    # Load big collision block on the ground
    _ = p.loadURDF(
        "cube_small.urdf", [0.5, 0.0, 0.5], globalScaling=10, useFixedBase=True
    )

    pandaId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    # End effector index
    endEffectorIndex = 11

    # Collect controllable joints
    controllableJoints = []
    for i in range(p.getNumJoints(pandaId)):
        jointInfo = p.getJointInfo(pandaId, i)
        if jointInfo[2] != p.JOINT_FIXED:
            controllableJoints.append(i)

    for j in controllableJoints:
        p.enableJointForceTorqueSensor(pandaId, j, enableSensor=True)

    targetOrn = p.getQuaternionFromEuler([0, -np.pi, 0])
    targetPos = [0.3, 0.25, 0.85]

    # Load target goal block
    goalBlockId = p.loadURDF("cube_small.urdf", targetPos, globalScaling=0.5)
    p.changeVisualShape(goalBlockId, -1, rgbaColor=[0, 1, 0, 0.5])

    step_size = 0.01
    print("\nControls:")
    print("w/s: forward/backward (x-axis)")
    print("a/d: left/right (y-axis)")
    print("q/e: up/down (z-axis)")
    print("x: quit")

    running = True

    while running:
        # Handle keyboard input from PyBullet
        keys = p.getKeyboardEvents()

        if ord("w") in keys and keys[ord("w")] & p.KEY_IS_DOWN:
            targetPos[0] += step_size
        if ord("s") in keys and keys[ord("s")] & p.KEY_IS_DOWN:
            targetPos[0] -= step_size
        if ord("a") in keys and keys[ord("a")] & p.KEY_IS_DOWN:
            targetPos[1] += step_size
        if ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:
            targetPos[1] -= step_size
        if ord("q") in keys and keys[ord("q")] & p.KEY_IS_DOWN:
            targetPos[2] += step_size
        if ord("e") in keys and keys[ord("e")] & p.KEY_IS_DOWN:
            targetPos[2] -= step_size
        if ord("x") in keys and keys[ord("x")] & p.KEY_WAS_TRIGGERED:
            running = False
            break

        # Update goal block position to match target
        p.resetBasePositionAndOrientation(goalBlockId, targetPos, [0, 0, 0, 1])

        # IK
        jointPoses = p.calculateInverseKinematics(
            pandaId,
            endEffectorIndex,
            targetPos,
            targetOrn,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )

        # Apply motor control
        for i, j in enumerate(controllableJoints[:7]):
            p.setJointMotorControl2(
                pandaId, j, p.POSITION_CONTROL, targetPosition=jointPoses[i], force=100
            )

        p.stepSimulation()

        # Check for collisions between robot and environment
        contact_points = p.getContactPoints(bodyA=pandaId)
        if contact_points:
            for contact in contact_points:
                bodyA = contact[1]
                bodyB = contact[2]
                linkA = contact[3]
                linkB = contact[4]
                contact_distance = contact[8]
                contact_force = contact[9]

                # Get link name for better readability
                if linkA >= 0:
                    linkA_name = p.getJointInfo(bodyA, linkA)[12].decode("utf-8")
                else:
                    linkA_name = "base"

                print(
                    f"Collision: Robot link '{linkA_name}' with body "
                    f"{bodyB} (link {linkB}), "
                    f"distance: {contact_distance:.4f}, "
                    f"force: {contact_force:.4f}"
                )

        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    simulation_loop()

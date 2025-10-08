#!/usr/bin/env python3
"""Generate a posed SMPL mesh and export it to OBJ (manual implementation, no
smplx).

Usage:
  python smpl_to_obj.py --models_dir /path/to/SMPL_python_v.1.1.0 \
                               --gender neutral \
                               --betas 0,0,0,0,0,0,0,0,0,0 \
                               --global_orient 0,0,0 \
                               --scale 1.0 \
                               --out smpl_body.obj
"""

import argparse
import os
from typing import Any

import numpy as np


def parse_args() -> Any:
    """Parse command line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models_dir", required=True, help="Root dir containing 'smpl/SMPL_*.npz'"
    )
    p.add_argument("--gender", default="neutral", choices=["male", "female", "neutral"])
    p.add_argument(
        "--betas",
        default="0,0,0,0,0,0,0,0,0,0",
        help="Comma-separated 10D shape vector",
    )
    p.add_argument(
        "--global_orient",
        default="0,0,0",
        help="Axis-angle (radians) for root orientation, comma-separated 3D",
    )
    p.add_argument(
        "--body_pose",
        default=None,
        help="(Optional) 69D axis-angle (23 joints x 3), comma-separated",
    )
    p.add_argument("--scale", type=float, default=1.0, help="Uniform scale multiplier")
    p.add_argument("--out", default="smpl_body.obj", help="Output OBJ path")
    return p.parse_args()


def mild_apose_69d() -> np.ndarray:
    """Arms at sides pose."""
    pose = np.zeros((23, 3), dtype=np.float32)
    pose[15, 2] = -1.4  # left shoulder (? could also be "collar")
    pose[16, 2] = 1.4  # right shoulder
    return pose.reshape(-1)


def rodrigues(r: Any) -> np.ndarray:
    """Convert axis-angle to rotation matrix."""
    theta = np.linalg.norm(r)
    if theta < 1e-6:
        return np.eye(3)
    r = r / theta
    K = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def lbs(
    betas: Any,
    pose: Any,
    v_template: Any,
    shapedirs: Any,
    posedirs: Any,
    J_regressor: Any,
    weights: Any,
    kintree_table: Any,
) -> np.ndarray:
    """Linear blend skinning."""
    batch_size = betas.shape[0]

    # Pad betas if needed to match shapedirs
    num_betas = betas.shape[1]
    num_shapedirs = shapedirs.shape[2]
    if num_betas < num_shapedirs:
        betas_padded = np.zeros((batch_size, num_shapedirs))
        betas_padded[:, :num_betas] = betas
        betas = betas_padded

    # Add shape blend shapes
    v_shaped = v_template + np.tensordot(betas, shapedirs, axes=([1], [2]))

    # Get joint locations
    J = np.dot(J_regressor, v_shaped[0])

    # Pose blend shapes
    pose_cube = pose.reshape(-1, 3)
    R_cube_big = np.array([rodrigues(aa) for aa in pose_cube])

    # Subtract identity for pose blend shapes
    R_cube = R_cube_big[1:, :, :]  # Ignore global rotation
    I_cube = np.eye(3)[np.newaxis, :, :]
    pose_feature = (R_cube - I_cube).reshape(-1)

    v_posed = v_shaped + np.tensordot(pose_feature, posedirs, axes=([0], [2]))

    # Build transformation matrices
    num_joints = kintree_table.shape[1]
    A = np.zeros((num_joints, 4, 4))

    # Root
    A[0, :3, :3] = R_cube_big[0]
    A[0, :3, 3] = J[0]
    A[0, 3, 3] = 1

    # Other joints
    for i in range(1, num_joints):
        parent = kintree_table[0, i]
        A[i, :3, :3] = R_cube_big[i]
        A[i, :3, 3] = J[i] - J[parent]
        A[i, 3, 3] = 1
        A[i] = A[parent] @ A[i]

    # Apply transformations and skinning
    T = np.zeros((num_joints, 4, 4))
    for i in range(num_joints):
        T[i] = A[i].copy()
        T[i, :3, 3] -= np.dot(A[i, :3, :3], J[i])

    # Skinning
    v_posed_homo = np.concatenate([v_posed[0], np.ones((v_posed.shape[1], 1))], axis=1)
    v_homo = np.zeros_like(v_posed_homo)

    for i in range(num_joints):
        v_homo += weights[:, i : i + 1] * np.dot(T[i], v_posed_homo.T).T

    vertices = v_homo[:, :3]

    return vertices


def write_obj(path: str, verts: Any, faces: Any) -> None:
    """Write OBJ file."""
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def main() -> None:
    """Main function."""
    a = parse_args()
    gender = a.gender.lower()

    # Load SMPL model
    npz_path = os.path.join(a.models_dir, "smpl", f"SMPL_{gender.upper()}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Model file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True, encoding="latin1")

    v_template = data["v_template"]
    shapedirs = data["shapedirs"]
    J_regressor = data["J_regressor"]
    if hasattr(J_regressor, "toarray"):
        J_regressor = J_regressor.toarray()
    posedirs = data["posedirs"]
    weights = data["weights"]
    faces = data["f"]
    kintree_table = data["kintree_table"]

    # Parse parameters
    betas = np.array([float(x) for x in a.betas.split(",")], dtype=np.float32).reshape(
        1, -1
    )
    root = np.array([float(x) for x in a.global_orient.split(",")], dtype=np.float32)

    if a.body_pose is None:
        body_pose = mild_apose_69d()
    else:
        bp = [float(x) for x in a.body_pose.split(",")]
        assert len(bp) == 69, "--body_pose must have 69 values (23x3 axis-angle)."
        body_pose = np.array(bp, dtype=np.float32)

    # Combine root and body pose
    pose = np.concatenate([root, body_pose])

    # Run LBS
    verts = lbs(
        betas,
        pose,
        v_template,
        shapedirs,
        posedirs,
        J_regressor,
        weights,
        kintree_table,
    )

    # Apply scale
    if a.scale != 1.0:
        verts *= a.scale

    # Center XY
    verts[:, :2] -= verts[:, :2].mean(axis=0, keepdims=True)

    write_obj(a.out, verts, faces)
    print(f"Wrote {a.out} with {len(verts)} vertices, {len(faces)} faces.")


if __name__ == "__main__":
    main()

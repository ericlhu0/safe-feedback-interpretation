"""Model-predictive torque controller for MuJoCo manipulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from .base_planner import BasePlanner

ArrayLike = NDArray[np.float64]


@dataclass
class MujocoArmState:
    """State of a MuJoCo-controlled arm (joint positions and velocities)."""

    qpos: ArrayLike
    qvel: ArrayLike

    @classmethod
    def from_data(cls, model: mujoco.MjModel, data: mujoco.MjData) -> "MujocoArmState":
        """Create state from MuJoCo model and data."""
        return cls(qpos=data.qpos[: model.nq].copy(), qvel=data.qvel[: model.nv].copy())


@dataclass(frozen=True)
class MujocoTorqueControl:
    """Joint torque command applied to the MuJoCo model's actuators."""

    torques: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "torques", np.array(self.torques, dtype=np.float64, copy=True)
        )


@dataclass(frozen=True)
class MujocoGoal:
    """Task-space goal specification for the end-effector."""

    position: ArrayLike
    orientation: ArrayLike | None = None  # unit quaternion (w, x, y, z)
    joint_positions: ArrayLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", np.array(self.position, dtype=np.float64, copy=True)
        )
        if self.position.shape != (3,):
            raise ValueError("goal position must have shape (3,)")
        if self.orientation is not None:
            object.__setattr__(
                self,
                "orientation",
                np.array(self.orientation, dtype=np.float64, copy=True),
            )
            if self.orientation.shape != (4,):
                raise ValueError(
                    "goal orientation must be a quaternion with shape (4,)"
                )
        if self.joint_positions is not None:
            object.__setattr__(
                self,
                "joint_positions",
                np.array(self.joint_positions, dtype=np.float64, copy=True),
            )
            if self.joint_positions.ndim != 1:
                raise ValueError("joint_positions must be a 1-D array")


@dataclass(frozen=True)
class MujocoMPCWeights:
    """Cost weights used by the torque-based MPC."""

    position: float = 3.0
    orientation: float = 0.3
    joint_position: float = 0.05
    torque: float = 1e-3
    terminal_position: float = 5.0
    terminal_orientation: float = 1.0
    terminal_joint_position: float = 0.5


class MujocoArmDynamics:
    """Lightweight wrapper for stepping MuJoCo arm dynamics."""

    def __init__(self, model: mujoco.MjModel, *, substeps: int = 1) -> None:
        if substeps < 1:
            raise ValueError("substeps must be at least 1")
        self.model = model
        self.substeps = substeps
        self._nq = model.nq
        self._nv = model.nv
        self._nu = model.nu

    def step(
        self, state: MujocoArmState, control: MujocoTorqueControl
    ) -> MujocoArmState:
        """Step the dynamics forward one timestep."""
        data = mujoco.MjData(self.model)
        data.qpos[: self._nq] = state.qpos
        data.qvel[: self._nv] = state.qvel
        mujoco.mj_forward(self.model, data)
        for _ in range(self.substeps):
            data.ctrl[: self._nu] = control.torques
            mujoco.mj_step(self.model, data)
        return MujocoArmState.from_data(self.model, data)


class MujocoTorqueMPC(BasePlanner[MujocoArmState, MujocoTorqueControl, MujocoGoal]):
    """Sampling-based MPC that plans joint torques for MuJoCo manipulators."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        dynamics: MujocoArmDynamics | None = None,
        end_effector_site: str,
        horizon: int = 10,
        num_samples: int = 512,
        deterministic_sequences: Sequence[ArrayLike] | None = None,
        weights: MujocoMPCWeights | None = None,
        seed: int | None = None,
        torque_limits: Sequence[Tuple[float, float]] | None = None,
        torque_sample_std: float = 20.0,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        self.model = model
        self.dynamics = dynamics or MujocoArmDynamics(model)
        self.horizon = horizon
        self.weights = weights or MujocoMPCWeights()
        self._rng = np.random.default_rng(seed)
        self._num_samples = num_samples
        self._torque_sample_std = torque_sample_std

        try:
            self._site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site
            )
        except mujoco.Error as exc:  # pragma: no cover - MuJoCo raises custom error
            raise ValueError(f"unknown site '{end_effector_site}'") from exc

        self._nq = model.nq
        self._nv = model.nv
        self._nu = model.nu

        if torque_limits is not None:
            if len(torque_limits) != self._nu:
                raise ValueError("torque_limits must match number of actuators")
            lower, upper = zip(*torque_limits)
            self._torque_lower = np.asarray(lower, dtype=np.float64)
            self._torque_upper = np.asarray(upper, dtype=np.float64)
        else:
            self._torque_lower, self._torque_upper = self._infer_torque_limits()

        if np.any(self._torque_lower > self._torque_upper):
            raise ValueError("invalid torque limits: lower bound exceeds upper bound")

        self._deterministic_sequences = self._build_deterministic_sequences(
            deterministic_sequences
        )

    def plan(
        self, state: MujocoArmState, goal: MujocoGoal
    ) -> Tuple[float, Tuple[MujocoTorqueControl, ...]]:
        best_cost = float("inf")
        best_sequence: Tuple[MujocoTorqueControl, ...] | None = None

        for cost, sequence in self.iter_plans(state, goal):
            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence

        if best_sequence is None:
            raise RuntimeError("no control sequence evaluated")
        return best_cost, best_sequence

    def iter_plans(
        self, state: MujocoArmState, goal: MujocoGoal
    ) -> Iterable[Tuple[float, Tuple[MujocoTorqueControl, ...]]]:
        """Yield all candidate plans and their costs."""
        for sequence in self._candidate_sequences():
            cost = self._evaluate_sequence(state, goal, sequence)
            yield cost, sequence

    def step(
        self, state: MujocoArmState, control: MujocoTorqueControl
    ) -> MujocoArmState:
        clamped = self._clamp(control.torques)
        return self.dynamics.step(state, MujocoTorqueControl(clamped))

    def _candidate_sequences(self) -> Iterable[Tuple[MujocoTorqueControl, ...]]:
        yield from self._deterministic_sequences

        for _ in range(self._num_samples):
            torques = self._rng.normal(
                loc=0.0,
                scale=self._torque_sample_std,
                size=(self.horizon, self._nu),
            )
            torques = np.clip(torques, self._torque_lower, self._torque_upper)
            yield tuple(MujocoTorqueControl(t) for t in torques)

    def _build_deterministic_sequences(
        self, deterministic_sequences: Sequence[ArrayLike] | None
    ) -> Tuple[Tuple[MujocoTorqueControl, ...], ...]:
        if deterministic_sequences is not None:
            controls = []
            for seq in deterministic_sequences:
                arr = np.array(seq, dtype=np.float64, copy=True)
                if arr.ndim == 1:
                    if arr.shape != (self._nu,):
                        raise ValueError(
                            "deterministic torque vector must have shape (nu,)"
                        )
                    control_sequence = tuple(
                        MujocoTorqueControl(arr) for _ in range(self.horizon)
                    )
                elif arr.ndim == 2:
                    if arr.shape != (self.horizon, self._nu):
                        raise ValueError(
                            "deterministic sequence must have shape (horizon, nu)"
                        )
                    control_sequence = tuple(MujocoTorqueControl(row) for row in arr)
                else:
                    raise ValueError("deterministic sequence must be 1-D or 2-D array")
                controls.append(control_sequence)
            return tuple(controls)

        zero = np.zeros(self._nu, dtype=np.float64)
        max_pos = self._torque_upper.copy()
        max_neg = self._torque_lower.copy()

        return (
            tuple(MujocoTorqueControl(zero) for _ in range(self.horizon)),
            tuple(MujocoTorqueControl(max_pos) for _ in range(self.horizon)),
            tuple(MujocoTorqueControl(max_neg) for _ in range(self.horizon)),
        )

    def _evaluate_sequence(
        self,
        start_state: MujocoArmState,
        goal: MujocoGoal,
        sequence: Tuple[MujocoTorqueControl, ...],
    ) -> float:
        data = mujoco.MjData(self.model)
        data.qpos[: self._nq] = start_state.qpos
        data.qvel[: self._nv] = start_state.qvel
        mujoco.mj_forward(self.model, data)

        total_cost = 0.0
        for control in sequence:
            data.ctrl[: self._nu] = self._clamp(control.torques)
            for _ in range(self.dynamics.substeps):
                mujoco.mj_step(self.model, data)
            total_cost += self._stage_cost(data, control, goal)

        total_cost += self._terminal_cost(data, goal)
        return total_cost

    def _stage_cost(
        self, data: mujoco.MjData, control: MujocoTorqueControl, goal: MujocoGoal
    ) -> float:
        weights = self.weights

        pos_error = self._position_error(data, goal)
        pos_cost = weights.position * float(pos_error @ pos_error)

        orient_cost = 0.0
        if goal.orientation is not None:
            orient_error = self._orientation_error(data, goal.orientation)
            orient_cost = weights.orientation * (orient_error**2)

        joint_pos_cost = 0.0
        if goal.joint_positions is not None:
            joint_dim = goal.joint_positions.shape[0]
            diff = data.qpos[:joint_dim] - goal.joint_positions
            joint_pos_cost = weights.joint_position * float(diff @ diff)

        torque_cost = weights.torque * float(control.torques @ control.torques)
        torque_cost = 0

        total = pos_cost + orient_cost + joint_pos_cost + torque_cost
        return total

    def _terminal_cost(self, data: mujoco.MjData, goal: MujocoGoal) -> float:
        weights = self.weights

        pos_error = self._position_error(data, goal)
        terminal_cost = weights.terminal_position * float(pos_error @ pos_error)

        if goal.orientation is not None:
            orient_error = self._orientation_error(data, goal.orientation)
            terminal_cost += weights.terminal_orientation * (orient_error**2)

        if goal.joint_positions is not None:
            joint_dim = goal.joint_positions.shape[0]
            diff = data.qpos[:joint_dim] - goal.joint_positions
            terminal_cost += weights.terminal_joint_position * float(diff @ diff)

        return 0.0

    def _position_error(self, data: mujoco.MjData, goal: MujocoGoal) -> ArrayLike:
        current = data.site_xpos[self._site_id]
        return current - goal.position

    def _orientation_error(self, data: mujoco.MjData, target_quat: ArrayLike) -> float:
        current_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(current_quat, data.site_xmat[self._site_id])
        target_quat = self._normalize_quaternion(target_quat)
        dot = float(np.clip(np.dot(current_quat, target_quat), -1.0, 1.0))
        return 2.0 * np.arccos(abs(dot))

    @staticmethod
    def _normalize_quaternion(quat: ArrayLike) -> ArrayLike:
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("orientation quaternion must be non-zero")
        return quat / norm

    def _clamp(self, torques: ArrayLike) -> ArrayLike:
        return np.clip(torques, self._torque_lower, self._torque_upper)

    def clamp_torques(self, torques: ArrayLike) -> ArrayLike:
        """Clamp torques to the planner's internal limits."""
        return self._clamp(np.array(torques, dtype=np.float64, copy=True))

    def _infer_torque_limits(self) -> Tuple[ArrayLike, ArrayLike]:
        """Infer reasonable torque bounds from the model metadata."""
        ctrlrange = self.model.actuator_ctrlrange.copy()
        forcerange = getattr(self.model, "actuator_forcerange", None)
        if forcerange is None:
            forcerange = np.zeros_like(ctrlrange)

        lower = np.empty(self._nu, dtype=np.float64)
        upper = np.empty(self._nu, dtype=np.float64)
        default_span = max(self._torque_sample_std * 3.0, 1.0)

        for i in range(self._nu):
            lo, hi = ctrlrange[i]
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = forcerange[i]
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = (-default_span, default_span)
            lower[i] = lo
            upper[i] = hi

        return lower, upper


__all__ = [
    "MujocoArmState",
    "MujocoTorqueControl",
    "MujocoGoal",
    "MujocoMPCWeights",
    "MujocoArmDynamics",
    "MujocoTorqueMPC",
]

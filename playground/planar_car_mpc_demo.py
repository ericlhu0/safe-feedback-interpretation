"""Playground demo for the planar car model predictive controller."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Iterable, List, Tuple

from safe_feedback_interpretation.planners import (
    CircularObstacle,
    PlanarCarConfig,
    PlanarCarControl,
    PlanarCarDynamics,
    PlanarCarGoal,
    PlanarCarMPC,
    PlanarCarState,
)

DEFAULT_OBSTACLE = CircularObstacle(center_x=2.0, center_y=0.75, radius=0.6, buffer=0.3)


@dataclass(frozen=True)
class MPCPlanSnapshot:
    """Snapshot of an MPC plan from a particular state."""

    cost: float
    controls: Tuple[PlanarCarControl, ...]
    rollout_states: Tuple[PlanarCarState, ...]
    sampled_plans: Tuple[
        Tuple[float, Tuple[PlanarCarControl, ...], Tuple[PlanarCarState, ...]],
        ...,
    ] = ()


@dataclass(frozen=True)
class SimulationStepResult:
    """Result of advancing the closed-loop simulation by one step."""

    index: int
    state: PlanarCarState
    control: PlanarCarControl
    used_plan: MPCPlanSnapshot
    next_plan: MPCPlanSnapshot


class PlanarCarSimulation:
    """Closed-loop MPC simulation helper used by the playground and GUI."""

    def __init__(
        self,
        horizon: int,
        goal: PlanarCarGoal,
        config: PlanarCarConfig | None = None,
        initial_state: PlanarCarState | None = None,
        *,
        num_samples: int = 256,
        seed: int | None = None,
        obstacle: CircularObstacle | None = DEFAULT_OBSTACLE,
    ) -> None:
        if config is not None:
            if obstacle is not None and obstacle is not config.obstacle:
                config = replace(config, obstacle=obstacle)
            self._config = config
        else:
            self._config = PlanarCarConfig(
                dt=0.25,
                max_speed=3.0,
                max_acceleration=1.2,
                obstacle=obstacle,
            )
        self._dynamics = PlanarCarDynamics(self._config)
        self._mpc = PlanarCarMPC(
            dynamics=self._dynamics,
            horizon=horizon,
            num_samples=num_samples,
            seed=seed,
        )
        self._goal = goal
        self._initial_state = initial_state or PlanarCarState(
            x=0.0, y=0.0, heading=0.0, speed=0.0
        )
        self._states: List[PlanarCarState] = []
        self._controls: List[PlanarCarControl] = []
        self._last_plan: MPCPlanSnapshot | None = None
        self._pending_plan: MPCPlanSnapshot | None = None
        self.reset()

    def reset(self) -> None:
        """Reset the simulation to the initial state and recompute the plan."""
        self._states = [self._initial_state]
        self._controls = []
        self._last_plan = None
        self._pending_plan = self._compute_plan(self._initial_state)

    @property
    def current_state(self) -> PlanarCarState:
        """Return the latest state in the closed-loop trajectory."""
        return self._states[-1]

    @property
    def state_history(self) -> Tuple[PlanarCarState, ...]:
        """Return the sequence of closed-loop states."""
        return tuple(self._states)

    @property
    def control_history(self) -> Tuple[PlanarCarControl, ...]:
        """Return the applied control history."""
        return tuple(self._controls)

    @property
    def last_plan(self) -> MPCPlanSnapshot | None:
        """Return the plan that produced the latest applied control."""
        return self._last_plan

    @property
    def current_plan(self) -> MPCPlanSnapshot:
        """Return the plan that will be used for the next control action."""
        if self._pending_plan is None:
            self._pending_plan = self._compute_plan(self.current_state)
        return self._pending_plan

    @property
    def config(self) -> PlanarCarConfig:
        """Return the simulation dynamics configuration."""
        return self._config

    def step(self) -> SimulationStepResult:
        """Advance the simulation by one MPC step."""
        plan = self.current_plan
        if not plan.controls or not plan.rollout_states:
            raise RuntimeError("MPC plan returned no controls for the current state.")

        control = plan.controls[0]
        new_state = plan.rollout_states[0]

        self._states.append(new_state)
        self._controls.append(control)
        self._last_plan = plan
        self._pending_plan = self._compute_plan(new_state)

        return SimulationStepResult(
            index=len(self._states) - 1,
            state=new_state,
            control=control,
            used_plan=plan,
            next_plan=self.current_plan,
        )

    def _compute_plan(self, state: PlanarCarState) -> MPCPlanSnapshot:
        sampled: List[
            Tuple[float, Tuple[PlanarCarControl, ...], Tuple[PlanarCarState, ...]]
        ] = []
        for cost, controls in self._mpc.iter_plans(state, self._goal):
            rollout_state = state
            predicted: List[PlanarCarState] = []
            for control in controls:
                rollout_state = self._mpc.step(rollout_state, control)
                predicted.append(rollout_state)
            sampled.append((cost, controls, tuple(predicted)))

        if not sampled:
            raise RuntimeError("no plans evaluated by MPC")

        sampled.sort(key=lambda item: item[0])
        best_cost, best_controls, best_rollout = sampled[0]
        return MPCPlanSnapshot(
            cost=best_cost,
            controls=best_controls,
            rollout_states=best_rollout,
            sampled_plans=tuple(sampled),
        )


def run_demo(
    horizon: int = 4,
    steps: int = 12,
    goal: PlanarCarGoal | None = None,
    *,
    num_samples: int = 256,
    seed: int | None = None,
) -> Tuple[PlanarCarState, Tuple[PlanarCarState, ...]]:
    """Run a small MPC rollout and return the final and intermediate states."""
    default_goal = goal or PlanarCarGoal(
        x=4.0,
        y=1.5,
        target_speed=0.5,
    )
    simulation = PlanarCarSimulation(
        horizon=horizon,
        goal=default_goal,
        num_samples=num_samples,
        seed=seed,
    )

    for _ in range(max(0, steps)):
        simulation.step()

    return simulation.current_state, simulation.state_history


def print_rollout(states: Iterable[PlanarCarState]) -> None:
    """Print the rollout states in a human readable form."""
    for index, state in enumerate(states):
        print(
            f"step={index:02d} x={state.x:5.2f} y={state.y:5.2f} "
            f"heading={state.heading:5.2f} speed={state.speed:4.2f}"
        )


def launch_gui(
    horizon: int,
    goal: PlanarCarGoal,
    steps: int,
    interval_ms: int,
    auto_start: bool = False,
    *,
    num_samples: int = 256,
    seed: int | None = None,
) -> None:
    """Launch a Tkinter GUI that visualizes the MPC rollout interactively."""

    try:
        import tkinter as tk
    except ImportError as exc:  # pragma: no cover - Tkinter may be missing.
        raise RuntimeError("Tkinter is required to launch the GUI demo.") from exc

    class MPCDemoGUI:
        """Tk GUI for stepping the planar car MPC and visualizing rollouts."""

        def __init__(self) -> None:
            self._simulation = PlanarCarSimulation(
                horizon=horizon,
                goal=goal,
                num_samples=num_samples,
                seed=seed,
            )
            self._interval_ms = interval_ms
            self._max_steps = steps
            self._steps_taken = 0
            self._playing = False
            self._after_id: str | None = None
            self._upcoming_plan = self._simulation.current_plan
            self._last_plan: MPCPlanSnapshot | None = None

            self._root = tk.Tk()
            self._root.title("Planar Car MPC Demo")

            self._width = 720
            self._height = 480
            self._margin = 40

            self._canvas = tk.Canvas(
                self._root, width=self._width, height=self._height, bg="white"
            )
            self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            sidebar = tk.Frame(self._root, width=300)
            sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
            sidebar.pack_propagate(False)

            tk.Label(sidebar, text="MPC Console", font=("Helvetica", 13, "bold")).pack(
                anchor=tk.W
            )

            self._info_text = tk.Text(
                sidebar,
                width=36,
                height=24,
                state=tk.DISABLED,
                wrap=tk.WORD,
            )
            self._info_text.configure(font=("Courier", 11))
            self._info_text.pack(fill=tk.BOTH, expand=True, pady=(6, 10))

            buttons = tk.Frame(sidebar)
            buttons.pack(fill=tk.X)

            self._step_button = tk.Button(
                buttons, text="Step", width=8, command=self._on_step
            )
            self._step_button.pack(side=tk.LEFT, padx=2)

            self._play_button = tk.Button(
                buttons, text="Play", width=10, command=self._toggle_play
            )
            self._play_button.pack(side=tk.LEFT, padx=2)

            self._reset_button = tk.Button(
                buttons, text="Reset", width=8, command=self._reset
            )
            self._reset_button.pack(side=tk.LEFT, padx=2)

            self._scale = 1.0
            self._min_x = 0.0
            self._min_y = 0.0
            self._max_display_samples = 60

            self._render_scene()
            self._update_rollout_panel()
            self._update_button_states()

            if auto_start and self._can_step():
                self._start_playback()

        def run(self) -> None:
            """Enter the Tk main loop."""
            self._root.mainloop()

        def _can_step(self) -> bool:
            return self._max_steps <= 0 or self._steps_taken < self._max_steps

        def _on_step(self) -> None:
            self._do_step()

        def _reset(self) -> None:
            self._stop_playback()
            self._simulation.reset()
            self._steps_taken = 0
            self._last_plan = None
            self._upcoming_plan = self._simulation.current_plan
            self._render_scene()
            self._update_rollout_panel()
            self._update_button_states()

        def _toggle_play(self) -> None:
            if self._playing:
                self._stop_playback()
            else:
                self._start_playback()

        def _start_playback(self) -> None:
            if not self._can_step():
                return
            self._playing = True
            self._play_button.config(text="Pause")
            self._schedule_next_step()

        def _stop_playback(self) -> None:
            if self._after_id is not None:
                self._root.after_cancel(self._after_id)
                self._after_id = None
            if self._playing:
                self._playing = False
                self._play_button.config(text="Play")

        def _schedule_next_step(self) -> None:
            self._after_id = self._root.after(self._interval_ms, self._auto_step)

        def _auto_step(self) -> None:
            self._after_id = None
            if not self._playing:
                return
            if not self._do_step():
                self._stop_playback()
                return
            if self._playing:
                self._schedule_next_step()

        def _do_step(self) -> bool:
            if not self._can_step():
                self._update_button_states()
                return False

            result = self._simulation.step()
            self._steps_taken += 1
            self._last_plan = result.used_plan
            self._upcoming_plan = result.next_plan

            self._render_scene()
            self._update_rollout_panel()
            self._update_button_states()
            return True

        def _compute_scale(self) -> Tuple[float, float, float]:
            states: List[PlanarCarState] = list(self._simulation.state_history)
            if self._upcoming_plan.rollout_states:
                states.extend(self._upcoming_plan.rollout_states)
            if self._upcoming_plan.sampled_plans:
                for _, _, sample_states in self._upcoming_plan.sampled_plans[
                    : self._max_display_samples
                ]:
                    states.extend(sample_states)

            if not states:
                states = [PlanarCarState(0.0, 0.0, 0.0, 0.0)]

            xs = [state.x for state in states]
            ys = [state.y for state in states]
            padding = 0.5
            min_x = min(xs) - padding
            max_x = max(xs) + padding
            min_y = min(ys) - padding
            max_y = max(ys) + padding

            span_x = max(max_x - min_x, 1e-6)
            span_y = max(max_y - min_y, 1e-6)
            scale_x = (self._width - 2 * self._margin) / span_x
            scale_y = (self._height - 2 * self._margin) / span_y
            scale = max(min(scale_x, scale_y), 1e-6)
            return scale, min_x, min_y

        def _state_to_canvas(
            self, x_value: float, y_value: float
        ) -> Tuple[float, float]:
            cx = self._margin + (x_value - self._min_x) * self._scale
            cy = self._height - self._margin - (y_value - self._min_y) * self._scale
            return cx, cy

        def _render_scene(self) -> None:
            self._canvas.delete("all")
            self._scale, self._min_x, self._min_y = self._compute_scale()
            self._draw_axes()
            self._draw_obstacle()
            self._draw_history()
            self._draw_prediction()

        def _draw_axes(self) -> None:
            x0 = self._margin
            y0 = self._height - self._margin
            x1 = self._width - self._margin
            y1 = self._margin
            self._canvas.create_rectangle(x0, y1, x1, y0, outline="#cccccc")
            self._canvas.create_text(
                self._width - self._margin,
                y0 + 18,
                text="x-axis",
                anchor=tk.E,
                fill="#555555",
            )
            self._canvas.create_text(
                x0 - 18,
                self._margin,
                text="y-axis",
                anchor=tk.N,
                fill="#555555",
            )

        def _draw_obstacle(self) -> None:
            obstacle = self._simulation.config.obstacle
            if obstacle is None:
                return

            cx, cy = self._state_to_canvas(obstacle.center_x, obstacle.center_y)
            radius = obstacle.radius * self._scale
            buffer_radius = (obstacle.radius + max(obstacle.buffer, 0.0)) * self._scale

            if buffer_radius > 0.0:
                self._canvas.create_oval(
                    cx - buffer_radius,
                    cy - buffer_radius,
                    cx + buffer_radius,
                    cy + buffer_radius,
                    outline="#ffe066",
                    dash=(4, 4),
                )

            self._canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                outline="#ff8c42",
                width=3,
                fill="",
            )

        def _draw_history(self) -> None:
            history = self._simulation.state_history
            if not history:
                return

            coords: List[float] = []
            for state in history:
                cx, cy = self._state_to_canvas(state.x, state.y)
                coords.extend([cx, cy])

            if len(coords) >= 4:
                self._canvas.create_line(*coords, fill="#1f77b4", width=2)

            current_state = history[-1]
            cx, cy = coords[-2], coords[-1]
            radius = 8
            self._canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill="#d62728",
                outline="#8c1c13",
            )

            heading_length = 28
            end_x = cx + heading_length * math.cos(current_state.heading)
            end_y = cy - heading_length * math.sin(current_state.heading)
            self._canvas.create_line(
                cx,
                cy,
                end_x,
                end_y,
                fill="#2ca02c",
                width=3,
                arrow=tk.LAST,
            )

        def _draw_prediction(self) -> None:
            history = self._simulation.state_history
            if not history or not self._upcoming_plan.sampled_plans:
                return

            start_state = history[-1]
            start_x, start_y = self._state_to_canvas(start_state.x, start_state.y)

            for idx, (_, _, sample_states) in enumerate(
                self._upcoming_plan.sampled_plans[: self._max_display_samples]
            ):
                coords: List[float] = [start_x, start_y]
                for state in sample_states:
                    cx, cy = self._state_to_canvas(state.x, state.y)
                    coords.extend([cx, cy])

                if len(coords) < 4:
                    continue

                is_best = idx == 0
                color = "#6c5ce7" if is_best else "#c5b4e3"
                width = 3 if is_best else 1
                dash = () if is_best else (3, 5)

                self._canvas.create_line(*coords, fill=color, width=width, dash=dash)

                if not sample_states:
                    continue

                if is_best:
                    marker_radius = 5
                    for state in sample_states:
                        cx, cy = self._state_to_canvas(state.x, state.y)
                        self._canvas.create_oval(
                            cx - marker_radius,
                            cy - marker_radius,
                            cx + marker_radius,
                            cy + marker_radius,
                            outline=color,
                        )
                else:
                    final_state = sample_states[-1]
                    cx, cy = self._state_to_canvas(final_state.x, final_state.y)
                    marker_radius = 2
                    self._canvas.create_oval(
                        cx - marker_radius,
                        cy - marker_radius,
                        cx + marker_radius,
                        cy + marker_radius,
                        outline="#a99bd4",
                    )

        def _update_rollout_panel(self) -> None:
            lines: List[str] = []
            lines.append(f"Steps taken: {self._steps_taken}")
            lines.append("")
            lines.append("Actual trajectory:")
            for idx, state in enumerate(self._simulation.state_history):
                lines.append(
                    f"  {idx:02d}: x={state.x:5.2f}, y={state.y:5.2f}, "
                    f"h={state.heading:5.2f}, v={state.speed:4.2f}"
                )

            if self._last_plan is not None:
                control = self._last_plan.controls[0]
                lines.append("")
                lines.append(
                    f"Last applied control: a={control.acceleration:+.2f}, "
                    f"w={control.yaw_rate:+.2f}"
                )
                lines.append(
                    f"Plan cost used: {self._last_plan.cost:.3f} "
                    f"(horizon {len(self._last_plan.controls)})"
                )

            if self._upcoming_plan.controls:
                lines.append("")
                lines.append(
                    f"Upcoming MPC rollout (cost {self._upcoming_plan.cost:.3f}):"
                )
                for idx, (control, state) in enumerate(
                    zip(
                        self._upcoming_plan.controls,
                        self._upcoming_plan.rollout_states,
                    ),
                    start=1,
                ):
                    lines.append(
                        f"  +{idx:02d}: a={control.acceleration:+.2f}, "
                        f"w={control.yaw_rate:+.2f} -> "
                        f"x={state.x:5.2f}, y={state.y:5.2f}, "
                        f"h={state.heading:5.2f}, v={state.speed:4.2f}"
                    )

            if self._upcoming_plan.sampled_plans:
                lines.append("")
                lines.append(
                    f"Sampled plans: {len(self._upcoming_plan.sampled_plans)} evaluated"
                )
                history = self._simulation.state_history
                last_state = (
                    history[-1] if history else PlanarCarState(0.0, 0.0, 0.0, 0.0)
                )
                for idx, (cost, controls, states) in enumerate(
                    self._upcoming_plan.sampled_plans[:5]
                ):
                    marker = "*" if idx == 0 else "-"
                    first_control = controls[0] if controls else None
                    accel = first_control.acceleration if first_control else 0.0
                    yaw_rate = first_control.yaw_rate if first_control else 0.0
                    final_state = states[-1] if states else last_state
                    lines.append(
                        f"  {marker}{idx+1:02d}: cost={cost:.3f} "
                        f"a0={accel:+.2f}, w0={yaw_rate:+.2f}, "
                        f"xf={final_state.x:5.2f}, yf={final_state.y:5.2f}"
                    )
                if len(self._upcoming_plan.sampled_plans) > 5:
                    lines.append("  ...")

            obstacle = self._simulation.config.obstacle
            if obstacle is not None:
                lines.append("")
                lines.append(
                    "Obstacle: center=("
                    f"{obstacle.center_x:.2f}, {obstacle.center_y:.2f}), "
                    f"radius={obstacle.radius:.2f}"
                )

            text = "\n".join(lines)
            self._info_text.configure(state=tk.NORMAL)
            self._info_text.delete("1.0", tk.END)
            self._info_text.insert(tk.END, text)
            self._info_text.configure(state=tk.DISABLED)

        def _update_button_states(self) -> None:
            if self._can_step():
                self._step_button.config(state=tk.NORMAL)
                self._play_button.config(state=tk.NORMAL)
            else:
                self._step_button.config(state=tk.DISABLED)
                self._stop_playback()
                self._play_button.config(state=tk.DISABLED)

    gui = MPCDemoGUI()
    gui.run()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="MPC planning horizon.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Rollout steps to simulate (<=0 for unlimited in GUI).",
    )
    parser.add_argument(
        "--goal-x", type=float, default=4.0, help="Target x-position for the goal."
    )
    parser.add_argument(
        "--goal-y", type=float, default=1.5, help="Target y-position for the goal."
    )
    parser.add_argument(
        "--goal-speed",
        type=float,
        default=0.5,
        help="Desired terminal speed at the goal.",
    )
    parser.add_argument(
        "--goal-heading",
        type=float,
        default=None,
        help="Optional desired terminal heading (radians).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch an animated Tkinter GUI instead of printing to stdout.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=250,
        help="Animation frame interval for the GUI in milliseconds.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically play the GUI animation on launch.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of sampled control sequences per MPC solve.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the MPC sampler (default: random).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for CLI usage."""
    args = parse_args()

    goal = PlanarCarGoal(
        x=args.goal_x,
        y=args.goal_y,
        target_speed=args.goal_speed,
        heading=args.goal_heading,
    )

    if args.gui:
        launch_gui(
            horizon=args.horizon,
            goal=goal,
            steps=args.steps,
            interval_ms=args.interval_ms,
            auto_start=args.auto,
            num_samples=args.samples,
            seed=args.seed,
        )
        return

    final_state, states = run_demo(
        horizon=args.horizon,
        steps=args.steps,
        goal=goal,
        num_samples=args.samples,
        seed=args.seed,
    )

    print("Planar car MPC demo rollout:\n")
    print_rollout(states)
    print(
        "\nFinal state:\n"
        f"  x={final_state.x:.2f}, y={final_state.y:.2f}, "
        f"heading={final_state.heading:.2f}, speed={final_state.speed:.2f}"
    )


if __name__ == "__main__":
    main()

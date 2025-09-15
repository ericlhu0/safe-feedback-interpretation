#!/usr/bin/env python3
"""Parametric visualization system for experiments results.

This script creates histograms, heatmaps, and individual trial plots
with configurable X and Y axes based on experimental variables.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from .data_utils import (
    calculate_average_distributions,
    calculate_metrics,
    group_results_by_axes_with_filter,
    load_results_data,
    print_data_summary,
)


def create_histogram_grid(
    stats: Dict[tuple, Dict],
    x_axis: str,
    y_axis: str,
    output_dir: Path,
    *,  # Force remaining args to be keyword-only
    body_part: str = "wrist",
    active_filters: Dict[str, str] | None = None,
    filter_values_order: list | None = None,
    no_ground_truth: bool = False,
):
    """Create histogram grid with configurable axes and optional multi-value
    filters.

    Args:
        stats: Statistics grouped by (x_value, y_value) or
            (x_value, y_value, filter_value)
        x_axis: Variable name for X axis
        y_axis: Variable name for Y axis
        output_dir: Directory to save plots
        body_part: Body part being analyzed
        active_filters: Dictionary of active filter settings
    """
    if not stats:
        print("No data available for histogram grid")
        return

    # Determine if we have multi-value filter data (3-tuple keys)
    has_filter_dimension = any(len(key) == 3 for key in stats.keys())

    if has_filter_dimension:
        # Get unique values for each dimension
        x_values = sorted(set(key[0] for key in stats.keys()))
        y_values = sorted(set(key[1] for key in stats.keys()), reverse=True)
        # Use provided order if available, otherwise extract from keys
        if filter_values_order:
            filter_values = filter_values_order
        else:
            filter_values = list(set(key[2] for key in stats.keys()))
        print(
            f"Multi-value filter detected: {len(filter_values)} filter values: "
            f"{filter_values}"
        )
    else:
        # Get unique values for each axis (2-tuple keys)
        x_values = sorted(set(key[0] for key in stats.keys()))
        y_values = sorted(set(key[1] for key in stats.keys()), reverse=True)
        filter_values = None

    class_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    # Handle flat layout (y_axis = "none")
    if y_axis == "none":
        # Create single row layout
        fig, axes = plt.subplots(1, len(x_values), figsize=(5 * len(x_values), 6))
        if len(x_values) == 1:
            axes = [axes]
        # Convert to 2D array format for consistent indexing
        axes = [axes]
        y_values = ["all"]  # Single row
    else:
        # Create subplots
        fig, axes = plt.subplots(
            len(y_values), len(x_values), figsize=(5 * len(x_values), 4 * len(y_values))
        )

        # Handle single subplot case
        if len(y_values) == 1 and len(x_values) == 1:
            axes = [[axes]]
        elif len(y_values) == 1:
            axes = [axes]
        elif len(x_values) == 1:
            axes = [[ax] for ax in axes]

    # Create title
    if y_axis == "none":
        title = (
            f"Model Predictions vs Ground Truth\n"
            f"{x_axis.replace('_', ' ').title()}\n"
            f"Body Part: {body_part}"
        )
    else:
        title = (
            f"Model Predictions vs Ground Truth\n"
            f"{y_axis.replace('_', ' ').title()} (rows) × "
            f"{x_axis.replace('_', ' ').title()} (columns)\n"
            f"Body Part: {body_part}"
        )

    # Add filter information to title
    if active_filters:
        filter_str = ", ".join(f"{k}={v}" for k, v in active_filters.items())
        title += f"\nFilters: {filter_str}"

    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Set up colors for multiple filter values
    filter_colors = ["skyblue", "lightcoral", "lightgreen", "gold", "plum", "lightgray"]

    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            ax = axes[i][j]

            if has_filter_dimension and filter_values is not None:
                # Multi-filter plotting: create grouped bars for each filter value
                n_filters = len(filter_values)
                bar_width = 0.8 / n_filters  # Divide bar space among filter values
                bar_positions = np.arange(
                    len(class_labels)
                )  # Base positions for comfort levels

                # Collect data for all filter values at this (x,y) position
                filter_data = {}
                total_trials = 0
                for filter_val in filter_values:
                    key = (x_val, y_val, filter_val)
                    if key in stats:
                        filter_data[filter_val] = stats[key]
                        total_trials += stats[key]["n_trials"]

                if filter_data:
                    # Plot bars for each filter value
                    for filter_idx, filter_val in enumerate(filter_values):
                        if filter_val in filter_data:
                            data = filter_data[filter_val]
                            model_probs = data["model_mean"]
                            model_std = data["model_std"]

                            # Calculate offset for this filter's bars
                            offset = (filter_idx - (n_filters - 1) / 2) * bar_width
                            positions = bar_positions + offset

                            # Create bars for this filter value
                            pred_bars = ax.bar(
                                positions,
                                model_probs,
                                width=bar_width,
                                color=filter_colors[filter_idx % len(filter_colors)],
                                alpha=0.7,
                                edgecolor="black",
                                linewidth=0.5,
                                label=f"{filter_val} (n={data['n_trials']})",
                                yerr=model_std,
                                capsize=2,
                                error_kw={"linewidth": 1, "capthick": 1},
                            )

                            # Add value labels on bars
                            for bar_rect, prob in zip(pred_bars, model_probs):
                                if prob > 0.01:
                                    ax.text(
                                        bar_rect.get_x() + bar_rect.get_width() / 2,
                                        bar_rect.get_height() + 0.01,
                                        f"{prob:.2f}",
                                        ha="center",
                                        va="bottom",
                                        fontsize=6,
                                        rotation=(
                                            90 if n_filters > 2 else 0
                                        ),  # Rotate text if many filters
                                    )

                    # Draw ground truth lines for each filter value (body part),
                    # only if labels exist and not hidden
                    if not no_ground_truth:
                        any_labels_drawn = False
                        for filter_idx, filter_val in enumerate(filter_values):
                            if filter_val in filter_data:
                                data = filter_data[filter_val]
                                label_probs = data.get("label_mean")
                                n_label_trials = data.get("n_label_trials", 0)
                                if label_probs is not None and n_label_trials > 0:
                                    # Calculate offset for this filter's
                                    # ground truth lines
                                    offset = (
                                        filter_idx - (n_filters - 1) / 2
                                    ) * bar_width
                                    for k, prob in enumerate(label_probs):
                                        if prob > 0.001:
                                            # Position the ground truth line to align
                                            # with the corresponding bar
                                            bar_center = k + offset
                                            ax.hlines(
                                                y=prob,
                                                xmin=bar_center - bar_width / 2,
                                                xmax=bar_center + bar_width / 2,
                                                colors="red",
                                                linewidth=3,
                                                alpha=0.8,
                                            )
                                            any_labels_drawn = True

                        # Add invisible line for ground truth legend only if any
                        # labels were drawn
                        if any_labels_drawn:
                            ax.hlines(
                                [],
                                [],
                                [],
                                colors="red",
                                linewidth=3,
                                alpha=0.8,
                                label="Ground Truth",
                            )

                    ax.set_title(
                        f"{y_axis}: {y_val}, {x_axis}: {x_val}\n"
                        f"(Total n={total_trials})",
                        fontsize=10,
                    )

                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No Data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{y_axis}: {y_val}, {x_axis}: {x_val}", fontsize=10)

            else:
                # Single-filter plotting: original logic
                single_key: tuple = (x_val, y_val)

                if single_key in stats:
                    data = stats[single_key]
                    model_probs = data["model_mean"]
                    model_std = data["model_std"]
                    label_probs = data.get("label_mean")
                    n_trials = data["n_trials"]

                    # Create prediction bars with error bars
                    pred_bars = ax.bar(
                        class_labels,
                        model_probs,
                        color="skyblue",
                        alpha=0.7,
                        edgecolor="navy",
                        label="Model Predictions",
                        yerr=model_std,
                        capsize=3,
                        error_kw={"linewidth": 1, "capthick": 1},
                    )

                    # Draw red horizontal lines for ground truth if labels exist
                    # and not disabled
                    if (
                        not no_ground_truth
                        and label_probs is not None
                        and data.get("n_label_trials", 0) > 0
                    ):
                        for k, prob in enumerate(label_probs):
                            if prob > 0.001:
                                ax.hlines(
                                    y=prob,
                                    xmin=k - 0.4,
                                    xmax=k + 0.4,
                                    colors="red",
                                    linewidth=4,
                                    alpha=0.8,
                                )
                        # Add invisible line for legend only when labels exist
                        ax.hlines(
                            [],
                            [],
                            [],
                            colors="red",
                            linewidth=4,
                            alpha=0.8,
                            label="Ground Truth",
                        )

                    # Add value labels on bars
                    for bar_rect, prob in zip(pred_bars, model_probs):
                        if prob > 0.01:
                            ax.text(
                                bar_rect.get_x() + bar_rect.get_width() / 2,
                                bar_rect.get_height() + 0.01,
                                f"{prob:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                            )

                    ax.set_title(
                        f"{y_axis}: {y_val}, {x_axis}: {x_val}\n(n={n_trials})",
                        fontsize=10,
                    )

                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No Data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{y_axis}: {y_val}, {x_axis}: {x_val}", fontsize=10)

            # Common formatting
            ax.set_ylim(0, 1.0)
            ax.legend(loc="upper right", fontsize=7)
            ax.set_ylabel("Probability")
            ax.set_xlabel("Comfort Threshold")
            if has_filter_dimension:
                ax.set_xticks(range(len(class_labels)))
                ax.set_xticklabels(class_labels, fontsize=8)
            else:
                ax.set_xticks(range(len(class_labels)))
                ax.set_xticklabels(class_labels, fontsize=8)

    # Add axis labels
    if y_axis != "none":
        fig.text(
            0.02,
            0.5,
            y_axis.replace("_", " ").title(),
            va="center",
            rotation="vertical",
            fontsize=14,
            fontweight="bold",
        )
    fig.text(
        0.5,
        -0.05,
        x_axis.replace("_", " ").title(),
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.12, hspace=0.4, wspace=0.3)

    # Save plot
    filename = f"histogram_grid_{y_axis}_x_{x_axis}_{body_part}.png"
    output_path = output_dir / "histograms" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved histogram grid: {output_path}")


def extract_filter_value_data(
    multi_value_data: Dict[Tuple[str, str, str], Dict], filter_value: str
) -> Dict[Tuple[str, str], Dict]:
    """Extract data for a specific filter value from 3-tuple keys.

    Args:
        multi_value_data: Data with 3-tuple keys (x_axis, y_axis, filter_value)
        filter_value: The specific filter value to extract

    Returns:
        Data with 2-tuple keys (x_axis, y_axis) for the specified filter value
    """
    extracted_data = {}
    for (x_val, y_val, f_val), data in multi_value_data.items():
        if f_val == filter_value:
            extracted_data[(x_val, y_val)] = data
    return extracted_data


def calculate_global_metric_ranges(
    metrics: Dict[Tuple, Dict],
) -> Dict[str, Tuple[float, float]]:
    """Calculate global min/max values for each metric across all data.

    Args:
        metrics: Metrics data (can have 2-tuple or 3-tuple keys)

    Returns:
        Dictionary mapping metric names to (min_value, max_value) tuples
    """
    metric_values: dict[str, list[float]] = {}

    # Collect all values for each metric
    for data in metrics.values():
        for metric_name, value in data.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)

    # Calculate global ranges
    global_ranges = {}
    for metric_name, values in metric_values.items():
        if values:
            global_ranges[metric_name] = (min(values), max(values))

    return global_ranges


def create_heatmap(
    metrics: Dict[Tuple[str, str], Dict],
    x_axis: str,
    y_axis: str,
    metric: str,
    output_dir: Path,
    *,  # Force remaining args to be keyword-only
    body_part: str = "wrist",
    active_filters: Dict[str, str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Create heatmap with configurable axes.

    Args:
        metrics: Metrics grouped by (x_value, y_value)
        x_axis: Variable name for X axis
        y_axis: Variable name for Y axis
        metric: Metric to display
        output_dir: Directory to save plots
        body_part: Body part being analyzed
        active_filters: Additional filters to include in title
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
    """
    if not metrics:
        print("No data available for heatmap")
        return

    # Get unique values for each axis
    x_values = sorted(set(x for x, y in metrics.keys()))
    y_values = sorted(set(y for x, y in metrics.keys()), reverse=True)

    # Create matrix and annotations
    matrix = np.full((len(y_values), len(x_values)), np.nan)
    annotations = np.full((len(y_values), len(x_values)), "", dtype=object)

    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            key = (x_val, y_val)
            if key in metrics:
                value = metrics[key].get(metric, np.nan)
                matrix[i, j] = value

                if not np.isnan(value):
                    # Check if there's a corresponding std metric
                    std_metric = metric.replace("_mean", "_std")
                    std_value = metrics[key].get(std_metric, np.nan)

                    if not np.isnan(std_value):
                        annotations[i, j] = f"{value:.3f}\n±{std_value:.3f}"
                    else:
                        annotations[i, j] = f"{value:.3f}"
                else:
                    annotations[i, j] = "N/A"

    # Create DataFrame for better labeling
    df_matrix = pd.DataFrame(
        matrix,
        index=[f"{y_val}" for y_val in y_values],
        columns=[f"{x_val}" for x_val in x_values],
    )

    # Create heatmap
    plt.figure(figsize=(2 + len(x_values) * 2, 2 + len(y_values) * 2))

    # Choose colormap based on metric
    if metric in ["brier_score_mean", "mae_mean"]:
        cmap = "Blues_r"  # Lower is better
    else:
        cmap = "Blues"  # Higher is better

    sns.heatmap(
        df_matrix,
        annot=annotations,
        fmt="",
        cmap=cmap,
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
    )

    title = (
        f"{metric.replace('_', ' ').title()}\n"
        f"{y_axis.replace('_', ' ').title()} (rows) × "
        f"{x_axis.replace('_', ' ').title()} (columns)\n"
        f"Body Part: {body_part}"
    )
    if active_filters:
        filter_str = ", ".join(f"{k}={v}" for k, v in active_filters.items())
        title += f"\nFilters: {filter_str}"

    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    plt.ylabel(y_axis.replace("_", " ").title(), fontsize=12)
    plt.xlabel(x_axis.replace("_", " ").title(), fontsize=12)

    plt.tight_layout()

    # Save plot
    filename = f"heatmap_{metric}_{y_axis}_x_{x_axis}_{body_part}.png"
    output_path = output_dir / "heatmaps" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved heatmap: {output_path}")


def create_individual_trials_plot(
    stats: Dict[Tuple[str, str], Dict],
    x_axis: str,
    y_axis: str,
    output_dir: Path,
    *,  # Force remaining args to be keyword-only
    body_part: str = "wrist",
    active_filters: Dict[str, str] | None = None,
):
    """Create individual trials plot with configurable axes.

    Args:
        stats: Statistics grouped by (x_value, y_value)
        x_axis: Variable name for X axis
        y_axis: Variable name for Y axis
        output_dir: Directory to save plots
        body_part: Body part being analyzed
    """
    if not stats:
        print("No data available for individual trials plot")
        return

    # Get unique values for each axis
    x_values = sorted(set(x for x, y in stats.keys()))
    y_values = sorted(set(y for x, y in stats.keys()), reverse=True)

    class_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    colors = plt.cm.get_cmap("Set3")(
        np.linspace(0, 1, 12)
    )  # Distinct colors for trials

    # Create subplots
    fig, axes = plt.subplots(
        len(y_values), len(x_values), figsize=(6 * len(x_values), 5 * len(y_values))
    )

    # Handle single subplot case
    if len(y_values) == 1 and len(x_values) == 1:
        axes = [[axes]]
    elif len(y_values) == 1:
        axes = [axes]
    elif len(x_values) == 1:
        axes = [[ax] for ax in axes]

    title = (
        f"Individual Trial Distributions\n"
        f"{y_axis.replace('_', ' ').title()} (rows) × "
        f"{x_axis.replace('_', ' ').title()} (columns)\n"
        f"Body Part: {body_part}"
    )
    if active_filters:
        filter_str = ", ".join(f"{k}={v}" for k, v in active_filters.items())
        title += f"\nFilters: {filter_str}"

    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            ax = axes[i][j]
            key = (x_val, y_val)

            if key in stats:
                data = stats[key]
                trials = data["individual_trials"]["model_distributions"]
                labels = data["label_mean"]
                n_trials = len(trials)

                if trials:
                    # Calculate bar positions
                    total_width = 0.8
                    bar_width = total_width / max(n_trials, 1)

                    # Plot each trial
                    for trial_idx, trial_probs in enumerate(trials):
                        x_positions = np.arange(len(class_labels))
                        x_offset = (trial_idx - (n_trials - 1) / 2) * bar_width
                        x_trial_positions = x_positions + x_offset

                        color = colors[trial_idx % len(colors)]

                        ax.bar(
                            x_trial_positions,
                            trial_probs,
                            width=bar_width * 0.9,
                            color=color,
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=0.5,
                        )

                    # Draw red horizontal lines for ground truth
                    for k, prob in enumerate(labels):
                        if prob > 0.001:
                            ax.hlines(
                                y=prob,
                                xmin=k - 0.4,
                                xmax=k + 0.4,
                                colors="red",
                                linewidth=4,
                                alpha=0.9,
                            )

                    # Add legend

                    legend_elements = [
                        Line2D(
                            [0], [0], color="red", linewidth=4, label="Ground Truth"
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="gray",
                            linewidth=2,
                            label=f"Individual Trials (n={n_trials})",
                        ),
                    ]
                    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

                ax.set_title(
                    f"{y_axis}: {y_val}, {x_axis}: {x_val}\n(n={n_trials} trials)",
                    fontsize=10,
                )

            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{y_axis}: {y_val}, {x_axis}: {x_val}", fontsize=10)

            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Probability")
            ax.set_xlabel("Comfort Threshold")
            ax.set_xticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, fontsize=8)

    # Add axis labels
    fig.text(
        0.04,
        0.5,
        y_axis.replace("_", " ").title(),
        va="center",
        rotation="vertical",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        x_axis.replace("_", " ").title(),
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.1, hspace=0.5, wspace=0.3)

    # Save plot
    filename = f"individual_trials_{y_axis}_x_{x_axis}_{body_part}.png"
    output_path = output_dir / "individual_trials" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved individual trials plot: {output_path}")


def main() -> None:
    """Main function for creating parametric visualizations."""
    parser = argparse.ArgumentParser(
        description="Create parametric visualizations for experiments results"
    )
    parser.add_argument(
        "--type",
        choices=["histogram", "heatmap", "individual"],
        required=True,
        help="Type of visualization to create",
    )
    parser.add_argument(
        "--x-axis",
        choices=[
            "verbal_intensity",
            "facial_intensity",
            "source_specificity",
            "face_modality",
        ],
        default="facial_intensity",
        help="Variable for X axis (default: facial_intensity)",
    )
    parser.add_argument(
        "--y-axis",
        choices=[
            "verbal_intensity",
            "facial_intensity",
            "source_specificity",
            "face_modality",
            "none",
        ],
        default="verbal_intensity",
        help="Variable for Y axis (default: verbal_intensity, use 'none' "
        "for flat layout)",
    )
    parser.add_argument(
        "--metric",
        choices=[
            "brier_score_mean",
            "entropy_model_mean",
            "mae_mean",
            "cosine_similarity_mean",
            "prob_overlap_mean",
            "accuracy",
            "f1_score",
            "all",
        ],
        default="all",
        help="Metric for heatmap (default: all - creates heatmaps for all metrics)",
    )
    parser.add_argument(
        "--body-part",
        default="wrist",
        help="Body part(s) to analyze. Use comma-separated values for multiple "
        "(e.g., 'wrist,forearm') (default: wrist)",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Path to results file (default: ../results/results.jsonl "
        "relative to script)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for plots (default: outputs)",
    )

    # Filter arguments for non-plotted variables
    parser.add_argument(
        "--verbal-filter",
        help="Filter to specific verbal_intensity value(s). Use comma-separated "
        "values for multiple (e.g., 'high,mid')",
    )
    parser.add_argument(
        "--facial-filter",
        help="Filter to specific facial_intensity value(s). Use comma-separated "
        "values for multiple (e.g., 'high,mid')",
    )
    parser.add_argument(
        "--source-filter",
        help="Filter to specific source_specificity value(s). Use comma-separated "
        "values for multiple (e.g., 'very,region')",
    )
    parser.add_argument(
        "--modality-filter",
        help="Filter to specific face_modality value(s). Use comma-separated "
        "values for multiple (e.g., 'img,text')",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Hide ground truth red lines in histograms",
    )

    args = parser.parse_args()

    # Parse comma-separated filter values into lists
    def parse_filter_values(filter_arg: str | None) -> list[str] | None:
        """Parse comma-separated filter values into a list."""
        if filter_arg is None:
            return None
        return [val.strip() for val in filter_arg.split(",") if val.strip()]

    args.verbal_filter = parse_filter_values(args.verbal_filter)
    args.facial_filter = parse_filter_values(args.facial_filter)
    args.source_filter = parse_filter_values(args.source_filter)
    args.modality_filter = parse_filter_values(args.modality_filter)
    args.body_part = parse_filter_values(args.body_part)

    # Set up paths relative to this script file
    script_dir = Path(__file__).parent

    # Set default paths if not provided
    if args.results_file is None:
        args.results_file = script_dir.parent / "results" / "results.jsonl"
    else:
        args.results_file = Path(args.results_file)

    # Make output directory relative to script if not absolute
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    # Validate axis combination
    if args.x_axis == args.y_axis and args.y_axis != "none":
        print(f"❌ Error: X and Y axis cannot be the same variable ({args.x_axis})")
        sys.exit(1)

    # Validate filter arguments - cannot filter variables that are on axes
    filters = {
        "verbal_intensity": args.verbal_filter,
        "facial_intensity": args.facial_filter,
        "source_specificity": args.source_filter,
        "face_modality": args.modality_filter,
        "body_part": args.body_part,
    }

    for var, filter_values in filters.items():
        if filter_values and var in (args.x_axis, args.y_axis):
            print(
                f"❌ Error: Cannot filter {var} because it's on the "
                f"{var == args.x_axis and 'X' or 'Y'} axis"
            )
            sys.exit(1)

    # Handle multiple body parts
    body_parts_to_process = args.body_part if args.body_part else ["wrist"]

    print(f"🚀 Creating {args.type} visualization...")
    print(f"📊 Axes: {args.y_axis} (Y) × {args.x_axis} (X)")
    if len(body_parts_to_process) == 1:
        print(f"🎯 Body part: {body_parts_to_process[0]}")
    else:
        print(
            f"🎯 Body parts: {', '.join(body_parts_to_process)} "
            f"({len(body_parts_to_process)} total)"
        )

    # Identify which filter (if any) has multiple values
    multi_value_filter = None
    multi_value_filter_name = None
    for var, filter_values in filters.items():
        if filter_values and len(filter_values) > 1:
            if multi_value_filter is not None:
                print("❌ Error: Only one filter can have multiple values at a time")
                sys.exit(1)
            multi_value_filter = var
            multi_value_filter_name = var.replace("_", " ")

    # Show active filters
    active_filters = {}
    for k, v in filters.items():
        if v:
            if len(v) > 1:
                active_filters[k.replace("_", " ")] = ",".join(v)
            else:
                active_filters[k.replace("_", " ")] = v[0]

    if active_filters:
        filter_str = ", ".join(f"{k}={v}" for k, v in active_filters.items())
        print(f"🔍 Filters: {filter_str}")

    if multi_value_filter:
        print(
            f"📊 Multi-value filter: {multi_value_filter_name} will create "
            f"separate bars for each value"
        )

    print(f"📁 Results file: {args.results_file}")
    print(f"📁 Output directory: {output_dir}")

    # Load and process data
    try:
        results = load_results_data(str(args.results_file))

        # Apply single-value filters to the data
        filtered_results = results
        if any(filters.values()):
            original_count = len(filtered_results)
            for var, filter_values in filters.items():
                if filter_values:
                    config_key = var
                    if len(filter_values) == 1:
                        # Single value filter - filter out non-matching results
                        # Exception: don't filter by single body_part when it's
                        # the multi-value filter
                        if not (
                            var == "body_part" and multi_value_filter == "body_part"
                        ):
                            filtered_results = [
                                r
                                for r in filtered_results
                                if r.get("config_used", {}).get(config_key)
                                == filter_values[0]
                            ]
                    # Multi-value filters are handled later in grouping
            print(
                f"🔍 Filtered from {original_count} to {len(filtered_results)} results"
            )

        # Determine body_part for grouping (use first one if multi-value,
        # or the single value)
        body_part_for_grouping = (
            body_parts_to_process[0]
            if not (multi_value_filter == "body_part")
            else body_parts_to_process[0]
        )

        grouped_results = group_results_by_axes_with_filter(
            filtered_results,
            args.x_axis,
            args.y_axis,
            body_part=body_part_for_grouping,
            multi_value_filter=multi_value_filter,
            multi_filter_values=(
                filters.get(multi_value_filter) if multi_value_filter else None
            ),
        )
        print_data_summary(grouped_results, args.x_axis, args.y_axis)

        if not grouped_results:
            print("❌ No data found for the specified axis combination")
            sys.exit(1)

        if args.type == "histogram":
            stats = calculate_average_distributions(grouped_results)
            # Use appropriate body part for filename
            body_part_for_filename = (
                "multi_" + "_".join(body_parts_to_process)
                if multi_value_filter == "body_part"
                else body_parts_to_process[0]
            )
            # Pass the original filter order if it's a multi-value filter
            filter_order = (
                filters.get(multi_value_filter) if multi_value_filter else None
            )
            create_histogram_grid(
                stats,
                args.x_axis,
                args.y_axis,
                output_dir,
                body_part=body_part_for_filename,
                active_filters=active_filters,
                filter_values_order=filter_order,
                no_ground_truth=args.no_ground_truth,
            )

        elif args.type == "heatmap":
            metrics = calculate_metrics(grouped_results)

            # For standardized color ranges, use the same scope as what's being plotted
            global_ranges = calculate_global_metric_ranges(metrics)

            if multi_value_filter:
                # Handle multi-value filters by creating separate heatmaps
                # for each filter value
                filter_values_order = (
                    filters.get(multi_value_filter)
                    if filters.get(multi_value_filter)
                    else list(set(key[2] for key in metrics))
                )

                if args.metric == "all":
                    all_metrics = [
                        "brier_score_mean",
                        "entropy_model_mean",
                        "mae_mean",
                        "cosine_similarity_mean",
                        "prob_overlap_mean",
                        "accuracy",
                        "f1_score",
                    ]
                    if filter_values_order is not None:
                        print(
                            f"🔥 Creating heatmaps for all {len(all_metrics)} metrics "
                            f"across {len(filter_values_order)} "
                            f"{multi_value_filter} values..."
                        )

                        for filter_value in filter_values_order:
                            print(
                                f"  📊 Processing {multi_value_filter}: {filter_value}"
                            )
                            filter_metrics = extract_filter_value_data(
                                metrics, filter_value
                            )

                            for metric in all_metrics:
                                vmin, vmax = global_ranges.get(metric, (None, None))
                                create_heatmap(
                                    filter_metrics,
                                    args.x_axis,
                                    args.y_axis,
                                    metric,
                                    output_dir,
                                    body_part=filter_value,
                                    active_filters=active_filters,
                                    vmin=vmin,
                                    vmax=vmax,
                                )
                else:
                    if filter_values_order is not None:
                        print(
                            f"🔥 Creating {args.metric} heatmaps for "
                            f"{len(filter_values_order)} {multi_value_filter} values..."
                        )

                        for filter_value in filter_values_order:
                            print(
                                f"  📊 Processing {multi_value_filter}: {filter_value}"
                            )
                            filter_metrics = extract_filter_value_data(
                                metrics, filter_value
                            )
                            vmin, vmax = global_ranges.get(args.metric, (None, None))
                            create_heatmap(
                                filter_metrics,
                                args.x_axis,
                                args.y_axis,
                                args.metric,
                                output_dir,
                                body_part=filter_value,
                                active_filters=active_filters,
                                vmin=vmin,
                                vmax=vmax,
                            )
            else:
                # Handle single-value filters (original behavior)
                body_part_for_filename = body_parts_to_process[0]

                if args.metric == "all":
                    all_metrics = [
                        "brier_score_mean",
                        "entropy_model_mean",
                        "mae_mean",
                        "cosine_similarity_mean",
                        "prob_overlap_mean",
                        "accuracy",
                        "f1_score",
                    ]
                    print(f"🔥 Creating heatmaps for all {len(all_metrics)} metrics...")
                    for metric in all_metrics:
                        vmin, vmax = global_ranges.get(metric, (None, None))
                        create_heatmap(
                            metrics,
                            args.x_axis,
                            args.y_axis,
                            metric,
                            output_dir,
                            body_part=body_part_for_filename,
                            active_filters=active_filters,
                            vmin=vmin,
                            vmax=vmax,
                        )
                else:
                    vmin, vmax = global_ranges.get(args.metric, (None, None))
                    create_heatmap(
                        metrics,
                        args.x_axis,
                        args.y_axis,
                        args.metric,
                        output_dir,
                        body_part=body_part_for_filename,
                        active_filters=active_filters,
                        vmin=vmin,
                        vmax=vmax,
                    )

        elif args.type == "individual":
            stats = calculate_average_distributions(grouped_results)
            body_part_for_filename = (
                "multi_" + "_".join(body_parts_to_process)
                if multi_value_filter == "body_part"
                else body_parts_to_process[0]
            )
            create_individual_trials_plot(
                stats,
                args.x_axis,
                args.y_axis,
                output_dir,
                body_part=body_part_for_filename,
                active_filters=active_filters,
            )

        # Success message
        if args.type == "heatmap":
            if multi_value_filter:
                filter_values_order = (
                    filters.get(multi_value_filter)
                    if filters.get(multi_value_filter)
                    else list(set(key[2] for key in grouped_results.keys()))
                )
                if args.metric == "all":
                    all_metrics = [
                        "brier_score_mean",
                        "entropy_model_mean",
                        "mae_mean",
                        "cosine_similarity_mean",
                        "prob_overlap_mean",
                        "accuracy",
                        "f1_score",
                    ]
                    if filter_values_order is not None:
                        total_heatmaps = len(filter_values_order) * len(all_metrics)
                        print(
                            f"✅ {total_heatmaps} heatmap visualizations "
                            f"completed for all "
                            f"metrics across {len(filter_values_order)} "
                            f"{multi_value_filter} values!"
                        )
                else:
                    if filter_values_order is not None:
                        print(
                            f"✅ {len(filter_values_order)} heatmap visualizations "
                            f"completed for "
                            f"{args.metric} across {len(filter_values_order)} "
                            f"{multi_value_filter} values!"
                        )
            else:
                if args.metric == "all":
                    print("✅ Heatmap visualizations completed for all metrics!")
                else:
                    print("✅ Heatmap visualization completed!")
        else:
            print(f"✅ {args.type.title()} visualization completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

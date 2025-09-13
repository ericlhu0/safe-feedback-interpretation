#!/usr/bin/env python3
"""Create distribution histograms for smile experiments showing average
prediction distributions across scenarios."""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def load_smile_results(file_path: str) -> Dict:
    """Load smile experiment results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_average_distributions(results: Dict) -> Dict[str, Dict]:
    """Calculate average prediction distributions across all smile
    scenarios."""

    # Collect all distributions
    full_distributions = []
    single_distributions = []
    label_distributions = []

    for scenario_name, data in results.items():
        # Get wrist predictions (focus on wrist like the original analysis)
        if "wrist" in data.get("predictions_full", {}):
            wrist_full = data["predictions_full"]["wrist"]
            # Convert to array format (class 1-5 with probabilities)
            prob_array = [0.0] * 5
            for class_str, prob in wrist_full.items():
                class_idx = int(class_str) - 1
                if 0 <= class_idx < 5:
                    prob_array[class_idx] = float(prob)
            full_distributions.append(prob_array)

        if "wrist" in data.get("predictions_single", {}):
            wrist_single = data["predictions_single"]["wrist"]
            # Convert to array format (class 1-5 with probabilities)
            prob_array = [0.0] * 5
            for class_str, prob in wrist_single.items():
                class_idx = int(class_str) - 1
                if 0 <= class_idx < 5:
                    prob_array[class_idx] = float(prob)
            single_distributions.append(prob_array)

        # Get expected labels (same for all scenarios)
        if "wrist" in data.get("labels", {}):
            wrist_labels = data["labels"]["wrist"]
            prob_array = [0.0] * 5
            for class_str, prob in wrist_labels.items():
                class_idx = int(class_str) - 1
                if 0 <= class_idx < 5:
                    prob_array[class_idx] = float(prob)
            label_distributions.append(prob_array)

    # Calculate averages and standard deviations
    avg_distributions = {
        "smile_contradiction": {
            "verbal_level": "high",  # High discomfort verbal
            "facial_level": "none",  # Smiling (no discomfort facial)
        }
    }

    # Labels
    if label_distributions:
        label_arrays = np.array(label_distributions)
        avg_distributions["smile_contradiction"]["label_avg"] = np.mean(
            label_arrays, axis=0
        )
        avg_distributions["smile_contradiction"]["label_std"] = (
            np.std(label_arrays, axis=0, ddof=1)
            if len(label_arrays) > 1
            else np.zeros(5)
        )
        avg_distributions["smile_contradiction"]["label_count"] = len(
            label_distributions
        )
    else:
        avg_distributions["smile_contradiction"]["label_avg"] = np.zeros(5)
        avg_distributions["smile_contradiction"]["label_std"] = np.zeros(5)
        avg_distributions["smile_contradiction"]["label_count"] = 0

    # Full output predictions
    if full_distributions:
        full_arrays = np.array(full_distributions)
        avg_distributions["smile_contradiction"]["full_avg"] = np.mean(
            full_arrays, axis=0
        )
        avg_distributions["smile_contradiction"]["full_std"] = (
            np.std(full_arrays, axis=0, ddof=1) if len(full_arrays) > 1 else np.zeros(5)
        )
        avg_distributions["smile_contradiction"]["full_count"] = len(full_distributions)
    else:
        avg_distributions["smile_contradiction"]["full_avg"] = np.zeros(5)
        avg_distributions["smile_contradiction"]["full_std"] = np.zeros(5)
        avg_distributions["smile_contradiction"]["full_count"] = 0

    # Single token predictions
    if single_distributions:
        single_arrays = np.array(single_distributions)
        avg_distributions["smile_contradiction"]["single_avg"] = np.mean(
            single_arrays, axis=0
        )
        avg_distributions["smile_contradiction"]["single_std"] = (
            np.std(single_arrays, axis=0, ddof=1)
            if len(single_arrays) > 1
            else np.zeros(5)
        )
        avg_distributions["smile_contradiction"]["single_count"] = len(
            single_distributions
        )
    else:
        avg_distributions["smile_contradiction"]["single_avg"] = np.zeros(5)
        avg_distributions["smile_contradiction"]["single_std"] = np.zeros(5)
        avg_distributions["smile_contradiction"]["single_count"] = 0

    return avg_distributions


def create_smile_plot(avg_distributions: Dict, output_type: str):
    """Create a single histogram plot for smile contradiction experiment."""

    # Create output directory
    output_dir = Path("playground/smile_histograms")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if output_type == "labels":
        fig.suptitle(
            "Smile Contradiction Experiment\nAverage True Label Distributions\n",
            fontsize=16,
            fontweight="bold",
        )

        exp_key = "smile_contradiction"
        if exp_key in avg_distributions:
            probabilities = avg_distributions[exp_key]["label_avg"]
            std_values = avg_distributions[exp_key]["label_std"]
            count = avg_distributions[exp_key]["label_count"]

            # Create bar plot for labels only with error bars
            bars = ax.bar(
                class_labels,
                probabilities,
                width=0.8,
                color="orange",
                alpha=0.7,
                edgecolor="navy",
                yerr=std_values,
                capsize=5,
                error_kw={"linewidth": 2, "capthick": 2},
            )

            # Add value labels on bars
            for chart_bar, prob in zip(bars, probabilities):
                if prob > 0.01:
                    ax.text(
                        chart_bar.get_x() + chart_bar.get_width() / 2,
                        chart_bar.get_height() + 0.02,
                        f"{prob:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                    )

    else:
        fig.suptitle(
            f"Smile Contradiction Experiment\n{output_type.title()} Output Predictions vs True Labels\n",
            fontsize=16,
            fontweight="bold",
        )

        exp_key = "smile_contradiction"
        if exp_key in avg_distributions:
            # Get prediction probabilities and standard deviations
            if output_type == "full":
                pred_probabilities = avg_distributions[exp_key]["full_avg"]
                pred_std_values = avg_distributions[exp_key]["full_std"]
                count = avg_distributions[exp_key]["full_count"]
                color = "skyblue"
            else:
                pred_probabilities = avg_distributions[exp_key]["single_avg"]
                pred_std_values = avg_distributions[exp_key]["single_std"]
                count = avg_distributions[exp_key]["single_count"]
                color = "lightgreen"

            # Get label probabilities
            label_probabilities = avg_distributions[exp_key]["label_avg"]

            # Create bar plot for predictions with error bars
            pred_bars = ax.bar(
                class_labels,
                pred_probabilities,
                color=color,
                alpha=0.7,
                edgecolor="navy",
                label="Predictions",
                yerr=pred_std_values,
                capsize=5,
                error_kw={"linewidth": 2, "capthick": 2},
            )

            # Draw red horizontal lines (tick marks) for true labels at each class position
            for k, prob in enumerate(label_probabilities):
                if prob > 0.001:  # Only draw lines for non-zero probabilities
                    # Draw horizontal line spanning the width of the bar
                    ax.hlines(
                        y=prob,
                        xmin=k - 0.4,
                        xmax=k + 0.4,
                        colors="red",
                        linewidth=6,
                        alpha=0.8,
                    )

            # Add a single invisible line for legend
            ax.hlines(
                y=[],
                xmin=[],
                xmax=[],
                colors="red",
                linewidth=6,
                alpha=0.8,
                label="True Labels",
            )

            # Add value labels on prediction bars
            for chart_bar, prob in zip(pred_bars, pred_probabilities):
                if prob > 0.01:
                    ax.text(
                        chart_bar.get_x() + chart_bar.get_width() / 2,
                        chart_bar.get_height() + 0.02,
                        f"{prob:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                    )

            # Add legend
            ax.legend(loc="upper right", fontsize=12)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Average Probability", fontsize=14)
    ax.set_xlabel("Comfort Threshold", fontsize=14)
    ax.set_title(
        f"High Verbal Discomfort + Smiling Face\n(n={count} scenarios)",
        fontsize=14,
        pad=20,
    )

    # Style improvements
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f"smile_distribution_histogram_{output_type}.png"
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved smile histogram plot: {full_path}")


def create_smile_comparison_plot(avg_distributions: Dict):
    """Create side-by-side comparison between single token and full output for
    smile experiment."""

    # Create output directory
    output_dir = Path("playground/smile_histograms")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(
        "Smile Contradiction Experiment\nSingle Token vs Full Output Comparison\n",
        fontsize=16,
        fontweight="bold",
    )

    exp_key = "smile_contradiction"
    if exp_key in avg_distributions:
        # Get prediction probabilities and standard deviations for both output types
        single_pred = avg_distributions[exp_key]["single_avg"]
        single_std = avg_distributions[exp_key]["single_std"]
        full_pred = avg_distributions[exp_key]["full_avg"]
        full_std = avg_distributions[exp_key]["full_std"]
        single_count = avg_distributions[exp_key]["single_count"]
        full_count = avg_distributions[exp_key]["full_count"]

        # Get label probabilities
        label_probabilities = avg_distributions[exp_key]["label_avg"]

        # Create side-by-side bar plot
        x = np.arange(len(class_labels))
        width = 0.35  # Width of the bars

        # Create bars with error bars
        single_bars = ax.bar(
            x - width / 2,
            single_pred,
            width,
            color="lightgreen",
            alpha=0.7,
            edgecolor="darkgreen",
            label="Single Token",
            yerr=single_std,
            capsize=4,
            error_kw={"linewidth": 2, "capthick": 2},
        )
        full_bars = ax.bar(
            x + width / 2,
            full_pred,
            width,
            color="skyblue",
            alpha=0.7,
            edgecolor="navy",
            label="Full Output",
            yerr=full_std,
            capsize=4,
            error_kw={"linewidth": 2, "capthick": 2},
        )

        # Draw red horizontal lines (tick marks) for true labels at each class position
        for k, prob in enumerate(label_probabilities):
            if prob > 0.001:  # Only draw lines for non-zero probabilities
                # Draw horizontal line spanning both bars
                ax.hlines(
                    y=prob,
                    xmin=k - width * 0.6,
                    xmax=k + width * 0.6,
                    colors="red",
                    linewidth=6,
                    alpha=0.8,
                )

        # Add a single invisible line for legend
        ax.hlines(
            y=[],
            xmin=[],
            xmax=[],
            colors="red",
            linewidth=6,
            alpha=0.8,
            label="True Labels",
        )

        # Add value labels on bars
        for chart_bar, prob in zip(single_bars, single_pred):
            if prob > 0.01:
                ax.text(
                    chart_bar.get_x() + chart_bar.get_width() / 2,
                    chart_bar.get_height() + 0.02,
                    f"{prob:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        for chart_bar, prob in zip(full_bars, full_pred):
            if prob > 0.01:
                ax.text(
                    chart_bar.get_x() + chart_bar.get_width() / 2,
                    chart_bar.get_height() + 0.02,
                    f"{prob:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Average Probability", fontsize=14)
        ax.set_xlabel("Comfort Threshold", fontsize=14)
        ax.set_title(
            f"High Verbal Discomfort + Smiling Face\n"
            f"Single(n={single_count}), Full(n={full_count})",
            fontsize=14,
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels)
        ax.legend(loc="upper right", fontsize=12)

        # Style improvements
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

    plt.tight_layout()
    filename = "smile_distribution_histogram_comparison.png"
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved smile comparison histogram plot: {full_path}")


def main():
    """Main function to create smile experiment distribution histograms."""

    results_file = "playground/smile_experiment_results.json"

    # Check if results file exists
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("Please run the use_llm.py script first to generate results.")
        return

    print("🚀 Starting smile distribution histogram analysis...")

    # Load results
    print(f"📊 Loading results from {results_file}...")
    results = load_smile_results(results_file)
    print(f"Loaded {len(results)} scenarios")

    # Calculate average distributions
    print("🔄 Calculating average distributions...")
    avg_distributions = calculate_average_distributions(results)

    # Create histogram plots
    print("\n🔍 Creating distribution histograms...")

    # Create individual plots for labels, full output, and single token
    create_smile_plot(avg_distributions, "labels")
    create_smile_plot(avg_distributions, "full")
    create_smile_plot(avg_distributions, "single")

    # Create comparison plot
    create_smile_comparison_plot(avg_distributions)

    print("\n✅ Smile distribution histogram analysis completed!")
    print("   Generated 4 histogram plots:")
    print("   - True labels distribution")
    print("   - Full output predictions vs labels")
    print("   - Single token predictions vs labels")
    print("   - Single token vs full output comparison")
    print(f"   All plots saved to playground/smile_histograms/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create histograms showing individual trial distributions instead of averaged
values.

This script generates 3x3 grid plots where each subplot shows individual
trials as thin bars rather than averaged values with error bars,
providing better insight into trial-to-trial variability.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(".")

from analyze_q1_disagreement import (  # pylint: disable=wrong-import-position
    is_q1_scenario,
    organize_predictions_by_scenario,
    parse_scenario_classification,
)


def load_q1_data_from_results_dir(results_dir: str) -> tuple:
    """Load Q1 data from a specific results directory."""
    curr_dir = Path(__file__).resolve().parent

    # Load expert labels from experiment config
    config_path = curr_dir / "configs/experiment_1_disagreement.json"
    if not config_path.exists():
        raise FileNotFoundError(f"ERROR: Q1 config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        expert_data = json.load(f)

    # Load model predictions from specified results directory
    final_results_path = curr_dir / "results" / results_dir / "final_results.json"
    if not final_results_path.exists():
        raise FileNotFoundError(
            f"ERROR: Final results file not found at {final_results_path}"
        )

    with open(final_results_path, "r", encoding="utf-8") as f:
        final_results = json.load(f)

    # Extract Q1 experiment results
    q1_results = (
        final_results.get("raw_results", {}).get("experiment_1", {}).get("results", {})
    )
    if not q1_results:
        raise ValueError("ERROR: No experiment_1 results found in final_results.json")

    # Convert to format expected by rest of the code
    final_results_list = []
    for scenario_name, scenario_data in q1_results.items():
        if is_q1_scenario(scenario_name):
            # Extract predictions for each body part and query type
            for body_part in ["entire_arm", "upper_arm", "forearm", "wrist"]:
                if body_part in scenario_data.get("predictions_full", {}):
                    final_results_list.append(
                        {
                            "input_metadata": {
                                "scenario_name": scenario_name,
                                "query_type": "body_part_full",
                            },
                            "query_type": "body_part_full",
                            "body_part": body_part,
                            "cleaned_probabilities": scenario_data["predictions_full"][
                                body_part
                            ],
                        }
                    )

                if body_part in scenario_data.get("predictions_single", {}):
                    final_results_list.append(
                        {
                            "input_metadata": {
                                "scenario_name": scenario_name,
                                "query_type": "body_part_single",
                            },
                            "query_type": "body_part_single",
                            "body_part": body_part,
                            "cleaned_probabilities": scenario_data[
                                "predictions_single"
                            ][body_part],
                        }
                    )

    # Parse scenario classifications
    scenarios_by_type = {}
    for scenario in expert_data["scenarios"]:
        scenario_name = scenario["name"]
        scenario_type = parse_scenario_classification(scenario_name)
        scenarios_by_type[scenario_name] = {
            "expert_labels": scenario["labels"],
            "classification": scenario_type,
            "received_feedback": scenario["received_feedback"],
        }

    return expert_data, final_results_list, scenarios_by_type


def organize_individual_distributions(
    final_results_list: List,
    scenarios_by_type: Dict,
    body_part: str = "wrist",
    query_type: str = "body_part_full",
) -> Dict:
    """Organize individual trial distributions by verbal/facial combination.

    Args:
        final_results_list: List of prediction results
        scenarios_by_type: Dictionary mapping scenario names to metadata
        body_part: Body part to analyze (default: "wrist")
        query_type: Query type to analyze (default: "body_part_full")

    Returns:
        Dictionary with structure:
        {
            "verbal_high_facial_low": {
                "individual_trials": [[trial1_probs], [trial2_probs], ...],
                "label_distribution": [label_probs],
                "verbal_level": "high",
                "facial_level": "low",
                "scenario_names": ["scenario1", "scenario2", ...]
            }
        }
    """
    distributions_by_type = {}

    for result in final_results_list:
        if result["body_part"] == body_part and result["query_type"] == query_type:
            scenario_name = result["input_metadata"]["scenario_name"]

            if scenario_name in scenarios_by_type:
                classification = scenarios_by_type[scenario_name]["classification"]
                verbal_level = classification["verbal_level"]
                facial_level = classification["facial_level"]

                exp_key = f"verbal_{verbal_level}_facial_{facial_level}"

                # Initialize if not exists
                if exp_key not in distributions_by_type:
                    distributions_by_type[exp_key] = {
                        "individual_trials": [],
                        "label_distribution": [],
                        "verbal_level": verbal_level,
                        "facial_level": facial_level,
                        "scenario_names": [],
                    }

                # Convert prediction probabilities to array format (comfort levels 1-5)
                prob_array = [0.0] * 5
                for class_str, prob in result["cleaned_probabilities"].items():
                    class_idx = int(class_str) - 1  # Convert 1-5 to 0-4
                    if 0 <= class_idx < 5:
                        prob_array[class_idx] = float(prob)

                distributions_by_type[exp_key]["individual_trials"].append(prob_array)
                distributions_by_type[exp_key]["scenario_names"].append(scenario_name)

                # Add label distribution (same for all trials in this category)
                if not distributions_by_type[exp_key]["label_distribution"]:
                    expert_labels = scenarios_by_type[scenario_name]["expert_labels"]
                    if body_part in expert_labels:
                        label_array = [0.0] * 5
                        for class_str, prob in expert_labels[body_part].items():
                            class_idx = int(class_str) - 1
                            if 0 <= class_idx < 5:
                                label_array[class_idx] = float(prob)
                        distributions_by_type[exp_key][
                            "label_distribution"
                        ] = label_array

    return distributions_by_type


def create_individual_trial_plot(
    individual_distributions: Dict, output_type: str, modality: str = "image"
):
    """Create individual trial histogram plot with thin bars for each trial."""

    # Create output directory
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "results" / "histograms" / "individual_trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = ["none", "mid", "high"]
    class_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    if output_type == "labels":
        fig.suptitle(
            "Individual Trial True Label Distributions\n",
            fontsize=16,
            fontweight="bold",
        )
    else:
        modality_title = "Images" if modality == "image" else "Text Descriptions"
        fig.suptitle(
            f"Facial Expressions as {modality_title}\n"
            f"Individual {output_type.title()} Output Trials\n",
            fontsize=16,
            fontweight="bold",
        )

    # Color map for individual trials
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Generate distinct colors

    for i, verbal_level in enumerate(levels):
        for j, facial_level in enumerate(levels):
            ax = axes[i, j]
            exp_key = f"verbal_{verbal_level}_facial_{facial_level}"

            if exp_key in individual_distributions:
                data = individual_distributions[exp_key]
                trials = data["individual_trials"]
                labels = data["label_distribution"]
                num_trials = len(trials)

                if output_type == "labels":
                    # For labels, just show the single label distribution
                    if labels:
                        bars = ax.bar(
                            class_labels,
                            labels,
                            width=0.8,
                            color="orange",
                            alpha=0.7,
                            edgecolor="navy",
                        )

                        # Add value labels
                        for bar, prob in zip(bars, labels):
                            if prob > 0.01:
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + 0.02,
                                    f"{prob:.2f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=8,
                                )
                else:
                    # Show individual trials as thin bars
                    if trials and num_trials > 0:
                        # Calculate bar width and positions
                        total_width = 0.8  # Total width for all bars in each category
                        bar_width = total_width / max(num_trials, 1)

                        # Plot each trial
                        for trial_idx, trial_probs in enumerate(trials):
                            # Calculate x positions for this trial
                            x_positions = np.arange(len(class_labels))
                            x_offset = (trial_idx - (num_trials - 1) / 2) * bar_width
                            x_trial_positions = x_positions + x_offset

                            # Choose color for this trial
                            color = colors[trial_idx % len(colors)]

                            # Plot thin bars for this trial
                            ax.bar(
                                x_trial_positions,
                                trial_probs,
                                width=bar_width * 0.9,  # Slightly smaller for spacing
                                color=color,
                                alpha=0.7,
                                edgecolor="black",
                                linewidth=0.5,
                            )

                        # Draw red horizontal lines for true labels
                        if labels:
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
                        from matplotlib.lines import Line2D

                        legend_elements = [
                            Line2D(
                                [0], [0], color="red", linewidth=4, label="True Labels"
                            ),
                            Line2D(
                                [0],
                                [0],
                                color="gray",
                                linewidth=2,
                                label=f"Individual Trials (n={num_trials})",
                            ),
                        ]
                        ax.legend(
                            handles=legend_elements, loc="upper right", fontsize=7
                        )

                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Probability", fontsize=10)
                ax.set_xlabel("Comfort Threshold", fontsize=10)
                ax.set_title(
                    f"Verbal: {verbal_level.title()}, "
                    f"Facial: {facial_level.title()}\n(n={num_trials} trials)",
                    fontsize=10,
                )

                # Set x-axis labels to use the class_labels (words, not numbers)
                ax.set_xticks(range(len(class_labels)))
                ax.set_xticklabels(class_labels, fontsize=8)

            else:
                # No data available
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Probability")
                ax.set_xlabel("Comfort Threshold")
                ax.set_title(
                    f"Verbal: {verbal_level.title()}, "
                    f"Facial: {facial_level.title()}",
                    fontsize=10,
                )
                # Set x-axis labels to use words even for no data case
                ax.set_xticks(range(len(class_labels)))
                ax.set_xticklabels(class_labels, fontsize=8)

    # Add axis labels for the grid structure
    fig.text(
        0.02,
        0.5,
        "Verbal Expressed Discomfort",
        va="center",
        rotation="vertical",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Facial Expressed Discomfort",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08, hspace=0.5, wspace=0.3)

    filename = f"q1_individual_trials_{output_type}_{modality}.png"
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved individual trial histogram: {full_path}")


def main():
    """Main function to generate individual trial histograms."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate individual trial histograms for Q1 disagreement experiment"
    )
    parser.add_argument(
        "--results-dir",
        default="image_facial_expression_results",
        help="Results directory name (default: image_facial_expression_results)",
    )
    parser.add_argument(
        "--body-part",
        default="wrist",
        choices=["entire_arm", "upper_arm", "forearm", "wrist"],
        help="Body part to analyze (default: wrist)",
    )
    parser.add_argument(
        "--modality",
        default="image",
        choices=["image", "text"],
        help="Modality type (default: image)",
    )

    args = parser.parse_args()

    print("🚀 Starting individual trial histogram generation...")
    print(f"📊 Results directory: {args.results_dir}")
    print(f"🎯 Body part: {args.body_part}")
    print(f"🖼️  Modality: {args.modality}")

    try:
        # Load data
        print("📊 Loading Q1 data...")
        expert_data, final_results_list, scenarios_by_type = (
            load_q1_data_from_results_dir(args.results_dir)
        )

        # Organize individual distributions for full output
        print("🔍 Organizing individual trial distributions (full output)...")
        individual_distributions_full = organize_individual_distributions(
            final_results_list, scenarios_by_type, args.body_part, "body_part_full"
        )

        # Organize individual distributions for single token
        print("🔍 Organizing individual trial distributions (single token)...")
        individual_distributions_single = organize_individual_distributions(
            final_results_list, scenarios_by_type, args.body_part, "body_part_single"
        )

        # Generate plots
        print("📈 Creating individual trial histogram plots...")

        # Labels plot
        create_individual_trial_plot(
            individual_distributions_full, "labels", args.modality
        )

        # Full output plot
        create_individual_trial_plot(
            individual_distributions_full, "full", args.modality
        )

        # Single token plot
        create_individual_trial_plot(
            individual_distributions_single, "single", args.modality
        )

        print("✅ Individual trial histogram generation completed!")
        print(f"   Generated 3 histogram plots for {args.body_part}")
        print(f"   Saved to experiments/results/histograms/individual_trials/")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

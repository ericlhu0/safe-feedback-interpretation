"""Plot histograms for smile experiment results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_smile_results(file_path: str):
    """Load smile experiment results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def plot_uncertainty_histograms(results):
    """Plot uncertainty histograms for single token vs full output."""
    single_token_uncertainties = []
    full_output_uncertainties = []

    for scenario_name, data in results.items():
        single_token_uncertainties.append(data["single_token"]["uncertainty"])
        full_output_uncertainties.append(data["full_output"]["uncertainty"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Single token uncertainty histogram
    ax1.hist(
        single_token_uncertainties, bins=10, alpha=0.7, color="blue", edgecolor="black"
    )
    ax1.set_title("Single Token Uncertainty Distribution")
    ax1.set_xlabel("Uncertainty (1 - max_prob)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    # Full output uncertainty histogram
    ax2.hist(
        full_output_uncertainties, bins=10, alpha=0.7, color="red", edgecolor="black"
    )
    ax2.set_title("Full Output Uncertainty Distribution")
    ax2.set_xlabel("Uncertainty (1 - max_prob)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "playground/smile_uncertainty_histograms.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_entropy_histograms(results):
    """Plot entropy histograms for single token vs full output."""
    single_token_entropies = []
    full_output_entropies = []

    for scenario_name, data in results.items():
        single_token_entropies.append(data["single_token"]["entropy"])
        full_output_entropies.append(data["full_output"]["entropy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Single token entropy histogram
    ax1.hist(
        single_token_entropies, bins=10, alpha=0.7, color="green", edgecolor="black"
    )
    ax1.set_title("Single Token Entropy Distribution")
    ax1.set_xlabel("Entropy (bits)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    # Full output entropy histogram
    ax2.hist(
        full_output_entropies, bins=10, alpha=0.7, color="orange", edgecolor="black"
    )
    ax2.set_title("Full Output Entropy Distribution")
    ax2.set_xlabel("Entropy (bits)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("playground/smile_entropy_histograms.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_brier_score_histograms(results):
    """Plot Brier score histograms comparing single token vs full output vs
    labels."""
    single_token_brier = []
    full_output_brier = []
    comparison_brier = []

    for scenario_name, data in results.items():
        if "single_token_vs_labels" in data["brier_scores"]:
            single_token_brier.append(data["brier_scores"]["single_token_vs_labels"])
        if "full_output_vs_labels" in data["brier_scores"]:
            full_output_brier.append(data["brier_scores"]["full_output_vs_labels"])
        comparison_brier.append(data["comparisons"]["brier_single_vs_full"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Single token vs labels Brier scores
    if single_token_brier:
        ax1.hist(
            single_token_brier, bins=10, alpha=0.7, color="blue", edgecolor="black"
        )
        ax1.set_title("Single Token vs Labels\nBrier Score Distribution")
        ax1.set_xlabel("Brier Score")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

    # Full output vs labels Brier scores
    if full_output_brier:
        ax2.hist(full_output_brier, bins=10, alpha=0.7, color="red", edgecolor="black")
        ax2.set_title("Full Output vs Labels\nBrier Score Distribution")
        ax2.set_xlabel("Brier Score")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

    # Single vs full comparison Brier scores
    ax3.hist(comparison_brier, bins=10, alpha=0.7, color="purple", edgecolor="black")
    ax3.set_title("Single Token vs Full Output\nBrier Score Distribution")
    ax3.set_xlabel("Brier Score")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "playground/smile_brier_score_histograms.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_body_part_brier_scores(results):
    """Plot Brier scores by body part for both single token and full output."""
    body_parts = ["entire_arm", "upper_arm", "forearm", "wrist"]

    # Collect Brier scores by body part
    single_brier_by_part = {part: [] for part in body_parts}
    full_brier_by_part = {part: [] for part in body_parts}

    for scenario_name, data in results.items():
        for part in body_parts:
            if part in data["brier_scores_single"]:
                single_brier_by_part[part].append(data["brier_scores_single"][part])
            if part in data["brier_scores_full"]:
                full_brier_by_part[part].append(data["brier_scores_full"][part])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, part in enumerate(body_parts):
        ax = axes[i]

        # Plot both single token and full output Brier scores
        single_scores = single_brier_by_part[part]
        full_scores = full_brier_by_part[part]

        if single_scores:
            ax.hist(
                single_scores,
                bins=8,
                alpha=0.6,
                color="blue",
                label="Single Token",
                edgecolor="black",
            )
        if full_scores:
            ax.hist(
                full_scores,
                bins=8,
                alpha=0.6,
                color="red",
                label="Full Output",
                edgecolor="black",
            )

        ax.set_title(f'{part.replace("_", " ").title()} Brier Scores')
        ax.set_xlabel("Brier Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "playground/smile_body_part_brier_histograms.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_probability_distributions(results):
    """Plot example probability distributions for wrist predictions."""
    # Take first few scenarios as examples
    scenarios_to_plot = list(results.keys())[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, scenario_name in enumerate(scenarios_to_plot):
        ax = axes[i]
        data = results[scenario_name]

        # Get wrist probabilities
        single_probs = data["single_token"]["probs"]
        full_probs = data["full_output"]["probs"]

        # Convert to arrays for plotting
        levels = [1, 2, 3, 4, 5]
        single_values = [float(single_probs.get(str(level), 0)) for level in levels]
        full_values = [float(full_probs.get(str(level), 0)) for level in levels]

        # Plot bars
        x = np.arange(len(levels))
        width = 0.35

        ax.bar(
            x - width / 2,
            single_values,
            width,
            label="Single Token",
            alpha=0.7,
            color="blue",
        )
        ax.bar(
            x + width / 2,
            full_values,
            width,
            label="Full Output",
            alpha=0.7,
            color="red",
        )

        ax.set_title(f"{scenario_name} - Wrist Comfort Levels")
        ax.set_xlabel("Comfort Level")
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "playground/smile_probability_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    """Main function to generate all smile experiment plots."""
    results_file = "playground/smile_experiment_results.json"

    # Check if results file exists
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        print("Please run the use_llm.py script first to generate results.")
        return

    # Load results
    print(f"Loading results from {results_file}...")
    results = load_smile_results(results_file)
    print(f"Loaded {len(results)} scenarios")

    # Create output directory if it doesn't exist
    Path("playground").mkdir(exist_ok=True)

    # Generate plots
    print("Generating uncertainty histograms...")
    plot_uncertainty_histograms(results)

    print("Generating entropy histograms...")
    plot_entropy_histograms(results)

    print("Generating Brier score histograms...")
    plot_brier_score_histograms(results)

    print("Generating body part Brier score histograms...")
    plot_body_part_brier_scores(results)

    print("Generating probability distribution examples...")
    plot_probability_distributions(results)

    print("All plots generated and saved to playground/ directory!")


if __name__ == "__main__":
    main()

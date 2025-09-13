#!/usr/bin/env python3
"""Calculate all heatmap metrics for smile experiment results."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


def load_smile_results(file_path: str) -> Dict:
    """Load smile experiment results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_entropy(probs: Dict) -> float:
    """Calculate entropy from probability distribution."""
    entropy = 0.0
    for prob in probs.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy


def calculate_label_probability_mass(predictions: Dict, labels: Dict) -> float:
    """Calculate sum of predicted probabilities for labels with non-zero true
    probability."""
    mass = 0.0
    for label, true_prob in labels.items():
        if true_prob > 0:
            predicted_prob = predictions.get(label, 0.0)
            mass += predicted_prob
    return mass


def calculate_mae(predictions: Dict, labels: Dict) -> float:
    """Calculate Mean Absolute Error between predicted and true
    distributions."""
    # Convert to arrays aligned by class
    classes = set(predictions.keys()) | set(labels.keys())
    pred_array = np.array([float(predictions.get(str(c), 0.0)) for c in range(1, 6)])
    true_array = np.array([float(labels.get(str(c), 0.0)) for c in range(1, 6)])

    return np.mean(np.abs(pred_array - true_array))


def calculate_cosine_sim(predictions: Dict, labels: Dict) -> float:
    """Calculate cosine similarity between predicted and true distributions."""
    # Convert to arrays
    pred_array = np.array([float(predictions.get(str(c), 0.0)) for c in range(1, 6)])
    true_array = np.array([float(labels.get(str(c), 0.0)) for c in range(1, 6)])

    # Reshape for sklearn
    pred_array = pred_array.reshape(1, -1)
    true_array = true_array.reshape(1, -1)

    return cosine_similarity(pred_array, true_array)[0, 0]


def get_argmax_predictions(predictions: Dict) -> int:
    """Get the class with highest probability."""
    max_prob = 0
    max_class = 1
    for class_str, prob in predictions.items():
        if float(prob) > max_prob:
            max_prob = float(prob)
            max_class = int(class_str)
    return max_class


def get_true_labels(labels: Dict) -> int:
    """Get the true class (class with highest probability in labels)."""
    max_prob = 0
    max_class = 1
    for class_str, prob in labels.items():
        if float(prob) > max_prob:
            max_prob = float(prob)
            max_class = int(class_str)
    return max_class


def calculate_classification_metrics(
    all_predictions: list,
    all_labels: list,
    all_true_classes: list,
    all_pred_classes: list,
) -> Dict:
    """Calculate F1, precision, recall, accuracy for multi-class
    classification."""

    # Convert to numpy arrays
    true_classes = np.array(all_true_classes)
    pred_classes = np.array(all_pred_classes)

    # Calculate metrics using macro averaging (treats all classes equally)
    f1 = f1_score(true_classes, pred_classes, average="macro", zero_division=0)
    precision = precision_score(
        true_classes, pred_classes, average="macro", zero_division=0
    )
    recall = recall_score(true_classes, pred_classes, average="macro", zero_division=0)
    accuracy = accuracy_score(true_classes, pred_classes)

    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}


def calculate_smile_metrics(results: Dict) -> Dict[str, Any]:
    """Calculate comprehensive metrics for smile experiment results."""

    print("🧮 Calculating comprehensive metrics for smile experiments...")

    # Collect all data for wrist predictions
    full_predictions = []
    single_predictions = []
    label_distributions = []

    # Individual scenario metrics
    scenario_metrics = {}

    print(f"Processing {len(results)} scenarios...")

    for scenario_name, data in results.items():
        scenario_metrics[scenario_name] = {}

        # Get wrist data (focus on wrist like heatmap analysis)
        if "wrist" in data.get("predictions_full", {}):
            full_pred = data["predictions_full"]["wrist"]
            single_pred = data["predictions_single"]["wrist"]
            labels = data["labels"]["wrist"]

            # Convert string keys to consistent format
            full_pred_clean = {str(k): float(v) for k, v in full_pred.items()}
            single_pred_clean = {str(k): float(v) for k, v in single_pred.items()}
            labels_clean = {str(k): float(v) for k, v in labels.items()}

            full_predictions.append(full_pred_clean)
            single_predictions.append(single_pred_clean)
            label_distributions.append(labels_clean)

            # Calculate per-scenario metrics
            scenario_metrics[scenario_name] = {
                # Entropy metrics
                "entropy_full": calculate_entropy(full_pred_clean),
                "entropy_single": calculate_entropy(single_pred_clean),
                "entropy_labels": calculate_entropy(labels_clean),
                # Label probability mass
                "label_prob_mass_full": calculate_label_probability_mass(
                    full_pred_clean, labels_clean
                ),
                "label_prob_mass_single": calculate_label_probability_mass(
                    single_pred_clean, labels_clean
                ),
                # MAE
                "mae_full": calculate_mae(full_pred_clean, labels_clean),
                "mae_single": calculate_mae(single_pred_clean, labels_clean),
                # Cosine similarity
                "cosine_similarity_full": calculate_cosine_sim(
                    full_pred_clean, labels_clean
                ),
                "cosine_similarity_single": calculate_cosine_sim(
                    single_pred_clean, labels_clean
                ),
                # Brier scores (already calculated in original data)
                "brier_full": data.get("brier_scores", {}).get(
                    "full_output_vs_labels", 0.0
                ),
                "brier_single": data.get("brier_scores", {}).get(
                    "single_token_vs_labels", 0.0
                ),
            }

    print(f"Collected data from {len(full_predictions)} scenarios")

    # Get classification predictions for F1/precision/recall/accuracy
    all_true_classes = []
    all_pred_classes_full = []
    all_pred_classes_single = []

    for i in range(len(full_predictions)):
        true_class = get_true_labels(label_distributions[i])
        pred_class_full = get_argmax_predictions(full_predictions[i])
        pred_class_single = get_argmax_predictions(single_predictions[i])

        all_true_classes.append(true_class)
        all_pred_classes_full.append(pred_class_full)
        all_pred_classes_single.append(pred_class_single)

    # Calculate classification metrics
    full_classification = calculate_classification_metrics(
        full_predictions, label_distributions, all_true_classes, all_pred_classes_full
    )
    single_classification = calculate_classification_metrics(
        single_predictions,
        label_distributions,
        all_true_classes,
        all_pred_classes_single,
    )

    # Collect all individual metrics for statistics
    metrics_data = {
        "entropy_full": [m["entropy_full"] for m in scenario_metrics.values()],
        "entropy_single": [m["entropy_single"] for m in scenario_metrics.values()],
        "entropy_labels": [m["entropy_labels"] for m in scenario_metrics.values()],
        "label_prob_mass_full": [
            m["label_prob_mass_full"] for m in scenario_metrics.values()
        ],
        "label_prob_mass_single": [
            m["label_prob_mass_single"] for m in scenario_metrics.values()
        ],
        "mae_full": [m["mae_full"] for m in scenario_metrics.values()],
        "mae_single": [m["mae_single"] for m in scenario_metrics.values()],
        "cosine_similarity_full": [
            m["cosine_similarity_full"] for m in scenario_metrics.values()
        ],
        "cosine_similarity_single": [
            m["cosine_similarity_single"] for m in scenario_metrics.values()
        ],
        "brier_full": [m["brier_full"] for m in scenario_metrics.values()],
        "brier_single": [m["brier_single"] for m in scenario_metrics.values()],
    }

    # Calculate summary statistics for each metric
    summary_stats = {}

    for metric_name, values in metrics_data.items():
        values_array = np.array(values)
        summary_stats[f"{metric_name}_mean"] = np.mean(values_array)
        summary_stats[f"{metric_name}_std"] = (
            np.std(values_array, ddof=1) if len(values_array) > 1 else 0.0
        )
        summary_stats[f"{metric_name}_min"] = np.min(values_array)
        summary_stats[f"{metric_name}_max"] = np.max(values_array)

    # Add classification metrics
    summary_stats.update(
        {
            "f1_full": full_classification["f1"],
            "f1_single": single_classification["f1"],
            "precision_full": full_classification["precision"],
            "precision_single": single_classification["precision"],
            "recall_full": full_classification["recall"],
            "recall_single": single_classification["recall"],
            "accuracy_full": full_classification["accuracy"],
            "accuracy_single": single_classification["accuracy"],
        }
    )

    # Calculate entropy differences (predicted - labels)
    entropy_diffs_full = []
    entropy_diffs_single = []

    for i in range(len(scenario_metrics)):
        scenario_name = list(scenario_metrics.keys())[i]
        ef = scenario_metrics[scenario_name]["entropy_full"]
        es = scenario_metrics[scenario_name]["entropy_single"]
        el = scenario_metrics[scenario_name]["entropy_labels"]

        entropy_diffs_full.append(ef - el)
        entropy_diffs_single.append(es - el)

    summary_stats.update(
        {
            "entropy_diff_full_mean": np.mean(entropy_diffs_full),
            "entropy_diff_full_std": (
                np.std(entropy_diffs_full, ddof=1)
                if len(entropy_diffs_full) > 1
                else 0.0
            ),
            "entropy_diff_single_mean": np.mean(entropy_diffs_single),
            "entropy_diff_single_std": (
                np.std(entropy_diffs_single, ddof=1)
                if len(entropy_diffs_single) > 1
                else 0.0
            ),
        }
    )

    # Add experiment metadata
    summary_stats.update(
        {
            "n_scenarios": len(results),
            "verbal_level": "high",  # High discomfort verbal feedback
            "facial_level": "none",  # Smiling (no discomfort) facial expression
            "experiment_type": "smile_contradiction",
        }
    )

    # Print summary
    print(f"\n📊 SMILE EXPERIMENT METRICS SUMMARY:")
    print(f"=" * 60)
    print(f"Number of scenarios: {summary_stats['n_scenarios']}")
    print(f"Experiment type: High verbal discomfort + Smiling face")
    print()

    # Key metrics comparison
    metrics_to_show = [
        ("brier_full_mean", "brier_single_mean", "Brier Score", "lower is better"),
        ("entropy_full_mean", "entropy_single_mean", "Entropy", "lower is better"),
        ("f1_full", "f1_single", "F1 Score", "higher is better"),
        ("accuracy_full", "accuracy_single", "Accuracy", "higher is better"),
        ("mae_full_mean", "mae_single_mean", "Mean Absolute Error", "lower is better"),
        (
            "label_prob_mass_full_mean",
            "label_prob_mass_single_mean",
            "Label Probability Mass",
            "higher is better",
        ),
        (
            "cosine_similarity_full_mean",
            "cosine_similarity_single_mean",
            "Cosine Similarity",
            "higher is better",
        ),
    ]

    for full_metric, single_metric, name, direction in metrics_to_show:
        full_val = summary_stats.get(full_metric, 0)
        single_val = summary_stats.get(single_metric, 0)

        if direction == "lower is better":
            better = "Full" if full_val < single_val else "Single"
        else:
            better = "Full" if full_val > single_val else "Single"

        print(
            f"{name:25} | Full: {full_val:.4f} | Single: {single_val:.4f} | Better: {better}"
        )

    return {
        "summary_stats": summary_stats,
        "scenario_metrics": scenario_metrics,
        "individual_values": metrics_data,
    }


def main():
    """Main function to calculate all heatmap metrics for smile experiments."""

    results_file = "playground/smile_experiment_results.json"

    # Check if results file exists
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("Please run the use_llm.py script first to generate results.")
        return

    print("🚀 Starting smile experiment heatmap metrics calculation...")

    # Load results
    print(f"📊 Loading results from {results_file}...")
    results = load_smile_results(results_file)
    print(f"Loaded {len(results)} scenarios")

    # Calculate all metrics
    all_metrics = calculate_smile_metrics(results)

    # Save metrics to file
    output_file = "playground/smile_heatmap_metrics.json"
    with open(output_file, "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(v) for v in d]
            else:
                return convert_numpy(d)

        clean_metrics = clean_dict(all_metrics)
        json.dump(clean_metrics, f, indent=2)

    print(f"\n✅ All metrics calculated and saved to {output_file}")
    print(f"Generated {len(all_metrics['summary_stats'])} summary statistics")
    print(
        f"Calculated per-scenario metrics for {len(all_metrics['scenario_metrics'])} scenarios"
    )

    print(f"\nKey findings:")
    stats = all_metrics["summary_stats"]
    print(
        f"- Average Brier Score: Full={stats['brier_full_mean']:.4f}, Single={stats['brier_single_mean']:.4f}"
    )
    print(
        f"- Average Entropy: Full={stats['entropy_full_mean']:.4f}, Single={stats['entropy_single_mean']:.4f}"
    )
    print(
        f"- Average F1 Score: Full={stats['f1_full']:.4f}, Single={stats['f1_single']:.4f}"
    )
    print(
        f"- Average Accuracy: Full={stats['accuracy_full']:.4f}, Single={stats['accuracy_single']:.4f}"
    )


if __name__ == "__main__":
    main()

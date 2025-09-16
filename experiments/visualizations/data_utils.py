"""Data loading and processing utilities for visualization."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def load_results_data(results_file: str) -> List[Dict]:
    """Load results from JSON Lines file.

    Args:
        results_file: Path to results.json file

    Returns:
        List of result dictionaries
    """
    results = []
    results_path = Path(results_file)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"Loaded {len(results)} result entries from {results_file}")
    return results


def get_available_variables(results: List[Dict]) -> Dict[str, List[str]]:
    """Get all available experimental variables and their values.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping variable names to list of unique values
    """
    variables = defaultdict(set)

    for result in results:
        config = result["config_used"]

        # Add experimental variables
        if "verbal_intensity" in config:
            variables["verbal_intensity"].add(config["verbal_intensity"])
        if "facial_intensity" in config:
            variables["facial_intensity"].add(config["facial_intensity"])
        if "source_specificity" in config:
            variables["source_specificity"].add(config["source_specificity"])
        if "face_modality" in config:
            variables["face_modality"].add(config["face_modality"])

    # Convert sets to sorted lists
    return {var: sorted(list(values)) for var, values in variables.items()}


def parse_model_response(response_str: str) -> Dict[str, float]:
    """Parse model response JSON string to probability distribution.

    Args:
        response_str: JSON string from model response

    Returns:
        Dictionary mapping comfort levels (as strings) to probabilities
    """
    if not response_str:
        return {}

    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse model response: {response_str}")
        return {}


def group_results_by_axes(
    results: List[Dict], x_axis: str, y_axis: str, body_part: str = "wrist"
) -> Dict[Tuple[str, str], List[Dict]]:
    """Group results by specified X and Y axis variables.

    Args:
        results: List of result dictionaries
        x_axis: Variable name for X axis (e.g., "facial_intensity")
        y_axis: Variable name for Y axis (e.g., "verbal_intensity")
        body_part: Body part to analyze (default: "wrist")

    Returns:
        Dictionary mapping (x_value, y_value) tuples to lists of matching results
    """
    grouped = defaultdict(list)

    for result in results:
        config = result["config_used"]

        # Extract axis values
        x_val = config[x_axis]
        y_val = config[y_axis] if y_axis != "none" else "all"

        if x_val is not None and y_val is not None:
            # Parse model response
            model_response = parse_model_response(result["model_response"])

            # Get labels for the specified body part
            labels = result["labels"][body_part]

            # Include entries with model response even if labels are missing;
            # attach labels if present (may be empty dict)
            if model_response:
                enhanced_result = result.copy()
                enhanced_result["parsed_response"] = model_response
                enhanced_result["body_part_labels"] = labels or {}
                grouped[(x_val, y_val)].append(enhanced_result)

    print(f"Grouped results by {y_axis} (Y) × {x_axis} (X)")
    print(f"Found combinations: {list(grouped.keys())}")
    print(f"Total entries per combination: {[(k, len(v)) for k, v in grouped.items()]}")

    return dict(grouped)


def group_results_by_axes_with_filter(
    results: List[Dict],
    x_axis: str,
    y_axis: str,
    *,  # Force remaining args to be keyword-only
    body_part: str = "wrist",
    multi_value_filter: str | None = None,
    multi_filter_values: List[str] | None = None,
) -> Dict[Tuple, List[Dict]]:
    """Group results by specified X and Y axis variables, with optional multi-
    value filter dimension.

    Args:
        results: List of result dictionaries
        x_axis: Variable name for X axis (e.g., "facial_intensity")
        y_axis: Variable name for Y axis (e.g., "verbal_intensity")
        body_part: Body part to analyze (default: "wrist")
        multi_value_filter: Variable name for multi-value filter
            (e.g., "verbal_intensity")
        multi_filter_values: List of values for the multi-value filter

    Returns:
        Dictionary mapping tuples to lists of matching results.
        - If no multi-value filter: (x_value, y_value) -> results
        - If multi-value filter: (x_value, y_value, filter_value) -> results
    """
    if multi_value_filter is None:
        # Use the original grouping function
        return group_results_by_axes(results, x_axis, y_axis, body_part)

    grouped = defaultdict(list)

    if multi_value_filter == "body_part":
        # Special handling for body_part multi-value filter
        # Use the actual body_part from config_used, not replicate the same response
        for result in results:
            config = result["config_used"]

            # Extract axis values
            x_val = config.get(x_axis)
            y_val = config.get(y_axis) if y_axis != "none" else "all"

            # Get the actual body part this experiment was run for
            actual_body_part = config["body_part"]

            # Only include this result if it matches one of our requested body parts
            if (
                x_val is not None
                and y_val is not None
                and multi_filter_values is not None
                and actual_body_part in multi_filter_values
            ):

                # Parse model response (this is specific to this body part)
                model_response = parse_model_response(result["model_response"])

                # Get labels for this specific body part
                all_labels = result["labels"]
                labels = all_labels[actual_body_part]

                # Include entries with model response even if labels are missing
                if model_response:
                    enhanced_result = result.copy()
                    enhanced_result["parsed_response"] = model_response
                    enhanced_result["body_part_labels"] = labels or {}
                    grouped[(x_val, y_val, actual_body_part)].append(enhanced_result)
    else:
        # Regular multi-value filter handling
        for result in results:
            config = result["config_used"]

            # Extract axis values
            x_val = config.get(x_axis)
            y_val = config.get(y_axis) if y_axis != "none" else "all"

            # Extract filter value
            filter_val = config[multi_value_filter]

            # Only include results that match one of the specified filter values
            if (
                x_val is not None
                and y_val is not None
                and filter_val is not None
                and multi_filter_values is not None
                and filter_val in multi_filter_values
            ):

                # Parse model response
                model_response = parse_model_response(result["model_response"])

                # Get labels for the specified body part (single body part in this case)
                labels = result["labels"][body_part]

                # Include entries with model response even if labels are missing
                if model_response:
                    enhanced_result = result.copy()
                    enhanced_result["parsed_response"] = model_response
                    enhanced_result["body_part_labels"] = labels or {}
                    grouped[(x_val, y_val, filter_val)].append(enhanced_result)

    print(
        f"Grouped results by {y_axis} (Y) × {x_axis} (X) × "
        f"{multi_value_filter} (Filter)"
    )
    print(f"Found combinations: {list(grouped.keys())}")
    print(f"Total entries per combination: {[(k, len(v)) for k, v in grouped.items()]}")

    return dict(grouped)


def calculate_average_distributions(
    grouped_results: Dict[Tuple, List[Dict]],
) -> Dict[Tuple, Dict]:
    """Calculate average probability distributions for each group.

    Args:
        grouped_results: Results grouped by (x_value, y_value) or
                        (x_value, y_value, filter_value)

    Returns:
        Dictionary with statistics for each group
    """
    stats = {}

    for key_tuple, results in grouped_results.items():
        if len(key_tuple) == 2:
            x_val, y_val = key_tuple
            filter_val = None
        elif len(key_tuple) == 3:
            x_val, y_val, filter_val = key_tuple
        else:
            continue  # Skip invalid key formats
        if not results:
            continue

        # Collect all probability distributions
        model_distributions = []
        label_distributions = []

        for result in results:
            # Model predictions (convert to array format for comfort levels 1-5)
            model_probs = result["parsed_response"]
            model_array = [0.0] * 5
            for level_str, prob in model_probs.items():
                level_idx = int(level_str) - 1  # Convert 1-5 to 0-4
                if 0 <= level_idx < 5:
                    model_array[level_idx] = float(prob)
            model_distributions.append(model_array)

            # Ground truth labels
            labels = result["body_part_labels"]
            if labels:
                label_array = [0.0] * 5
                for level_str, prob in labels.items():
                    level_idx = int(level_str) - 1  # Convert 1-5 to 0-4
                    if 0 <= level_idx < 5:
                        label_array[level_idx] = float(prob)
                # Only count as a label distribution if any value > 0
                if any(val > 0 for val in label_array):
                    label_distributions.append(label_array)

        if model_distributions:
            model_arrays = np.array(model_distributions)
            stats_entry = {
                "x_value": x_val,
                "y_value": y_val,
                "n_trials": len(model_distributions),
                "model_mean": np.mean(model_arrays, axis=0),
                "model_std": (
                    np.std(model_arrays, axis=0, ddof=1)
                    if len(model_arrays) > 1
                    else np.zeros(5)
                ),
                "n_label_trials": len(label_distributions),
                "individual_trials": {
                    "model_distributions": model_distributions,
                    "label_distributions": label_distributions,
                },
            }
            if label_distributions:
                label_arrays = np.array(label_distributions)
                stats_entry["label_mean"] = np.mean(label_arrays, axis=0)
                stats_entry["label_std"] = (
                    np.std(label_arrays, axis=0, ddof=1)
                    if len(label_arrays) > 1
                    else np.zeros(5)
                )
            else:
                # Mark label fields as None-equivalents to signal plotting
                # to skip markers
                stats_entry["label_mean"] = None
                stats_entry["label_std"] = None

            # Add filter value if present
            if filter_val is not None:
                stats_entry["filter_value"] = filter_val

            stats[key_tuple] = stats_entry

    return stats


def calculate_metrics(grouped_results: Dict[Tuple, List[Dict]]) -> Dict[Tuple, Dict]:
    """Calculate performance metrics for each group.

    Args:
        grouped_results: Results grouped by (x_value, y_value) or
                        (x_value, y_value, filter_value)

    Returns:
        Dictionary with metrics for each group
    """

    metrics = {}

    for key_tuple, results in grouped_results.items():
        if len(key_tuple) == 2:
            x_val, y_val = key_tuple
            filter_val = None
        elif len(key_tuple) == 3:
            x_val, y_val, filter_val = key_tuple
        else:
            continue  # Skip invalid key formats
        if not results:
            continue

        brier_scores = []
        entropies_model = []
        entropies_labels = []
        mae_scores = []
        cosine_similarities = []
        prob_overlaps = []

        # For classification metrics, we need to convert to discrete predictions
        y_true_discrete = []
        y_pred_discrete = []

        for result in results:
            model_probs = result["parsed_response"]
            labels = result["body_part_labels"]

            # Convert to arrays
            model_array = np.array([model_probs[str(i)] for i in range(1, 6)])
            label_array = np.array([labels[str(i)] for i in range(1, 6)])

            if np.sum(model_array) > 0 and np.sum(label_array) > 0:
                # Normalize to ensure they sum to 1
                model_array = model_array / np.sum(model_array)
                label_array = label_array / np.sum(label_array)

                # Brier score
                brier_score = np.sum((model_array - label_array) ** 2)
                brier_scores.append(brier_score)

                # Entropy
                model_entropy = -np.sum(model_array * np.log(model_array + 1e-10))
                label_entropy = -np.sum(label_array * np.log(label_array + 1e-10))
                entropies_model.append(model_entropy)
                entropies_labels.append(label_entropy)

                # MAE (using expected values)
                levels = np.arange(1, 6)
                model_expected = np.sum(levels * model_array)
                label_expected = np.sum(levels * label_array)
                mae_scores.append(abs(model_expected - label_expected))

                # Cosine similarity
                cosine_sim = 1 - cosine(model_array, label_array)
                cosine_similarities.append(cosine_sim)

                # Probability overlap (intersection)
                prob_overlap = np.sum(np.minimum(model_array, label_array))
                prob_overlaps.append(prob_overlap)

                # For discrete metrics, use argmax
                y_pred_discrete.append(np.argmax(model_array))
                y_true_discrete.append(np.argmax(label_array))

        if brier_scores:
            # Classification metrics
            accuracy = accuracy_score(y_true_discrete, y_pred_discrete)
            f1 = f1_score(
                y_true_discrete, y_pred_discrete, average="weighted", zero_division=0
            )
            precision = precision_score(
                y_true_discrete, y_pred_discrete, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_true_discrete, y_pred_discrete, average="weighted", zero_division=0
            )

            metrics_entry = {
                "x_value": x_val,
                "y_value": y_val,
                "n_trials": len(results),
                "brier_score_mean": np.mean(brier_scores),
                "brier_score_std": (
                    np.std(brier_scores, ddof=1) if len(brier_scores) > 1 else 0
                ),
                "entropy_model_mean": np.mean(entropies_model),
                "entropy_model_std": (
                    np.std(entropies_model, ddof=1) if len(entropies_model) > 1 else 0
                ),
                "entropy_labels_mean": np.mean(entropies_labels),
                "entropy_labels_std": (
                    np.std(entropies_labels, ddof=1) if len(entropies_labels) > 1 else 0
                ),
                "mae_mean": np.mean(mae_scores),
                "mae_std": np.std(mae_scores, ddof=1) if len(mae_scores) > 1 else 0,
                "cosine_similarity_mean": np.mean(cosine_similarities),
                "cosine_similarity_std": (
                    np.std(cosine_similarities, ddof=1)
                    if len(cosine_similarities) > 1
                    else 0
                ),
                "prob_overlap_mean": (
                    np.mean(prob_overlaps) if len(prob_overlaps) > 0 else 0.0
                ),
                "prob_overlap_std": (
                    np.std(prob_overlaps, ddof=1) if len(prob_overlaps) > 1 else 0.0
                ),
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }

            # Add filter value if present
            if filter_val is not None:
                metrics_entry["filter_value"] = filter_val

            metrics[key_tuple] = metrics_entry

    return metrics


def print_data_summary(
    grouped_results: Dict[Tuple, List[Dict]], x_axis: str, y_axis: str
):
    """Print summary of loaded data.

    Args:
        grouped_results: Results grouped by axes (2-tuple or 3-tuple keys)
        x_axis: X axis variable name
        y_axis: Y axis variable name
    """
    print(f"\n📊 Data Summary for {y_axis} × {x_axis} Grid:")
    print("=" * 50)

    total_entries = sum(len(results) for results in grouped_results.values())
    print(f"Total entries: {total_entries}")
    print(f"Grid combinations: {len(grouped_results)}")

    # Print grid layout
    if grouped_results:
        # Handle both 2-tuple and 3-tuple keys
        x_values = sorted(set(key[0] for key in grouped_results.keys()))
        y_values = sorted(set(key[1] for key in grouped_results.keys()), reverse=True)

        # Check if we have multi-value filter (3-tuple keys)
        has_filter_dimension = any(len(key) == 3 for key in grouped_results.keys())
        if has_filter_dimension:
            filter_values = list(set(key[2] for key in grouped_results.keys()))
            print(f"Filter values: {filter_values}")
            print(
                f"\nGrid layout with multi-value filter "
                f"({len(y_values)}×{len(x_values)}×{len(filter_values)}):"
            )

            # For multi-value filters, show total count per (x, y) combination
            print(f"{'':>12}", end="")
            for x_val in x_values:
                print(f"{x_val:>12}", end="")
            print()

            for y_val in y_values:
                print(f"{y_val:>12}", end="")
                for x_val in x_values:
                    # Sum counts across all filter values for this (x, y) combination
                    total_count = sum(
                        len(grouped_results[(x_val, y_val, filt_val)])
                        for filt_val in filter_values
                    )
                    print(f"{total_count:>12}", end="")
                print()
        else:
            print(f"\nGrid layout ({len(y_values)}×{len(x_values)}):")
            print(f"{'':>12}", end="")
            for x_val in x_values:
                print(f"{x_val:>8}", end="")
            print()

            for y_val in y_values:
                print(f"{y_val:>12}", end="")
                for x_val in x_values:
                    count = len(grouped_results[(x_val, y_val)])
                    print(f"{count:>8}", end="")
                print()

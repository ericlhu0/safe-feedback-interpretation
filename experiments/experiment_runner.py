"""Main experiment runner for all research questions."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from experiment_utils import (
    get_system_prompt,
    load_experiment_configs,
    run_uncertainty_analysis,
)

from safe_feedback_interpretation.models.openai_model import OpenAIModel


def save_incremental_result(result: Dict[str, Any], output_file: str) -> None:
    """Save a single result to JSONL file for incremental tracking.

    Args:
        result: Result dictionary to save
        output_file: Path to JSONL file
    """
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )

    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(result, f)
        f.write("\n")


def classify_scenario(
    scenario_name: str, experiment_name: str
) -> Dict[str, str | None]:
    """Classify scenario by type based on naming patterns.

    Args:
        scenario_name: Name of the scenario (e.g., 'vhfh_1', 'specific_location')
        experiment_name: Name of the experiment

    Returns:
        Dictionary with scenario classification details
    """
    # Remove trailing numbers and underscores
    clean_name = scenario_name.rstrip("_0123456789")

    classification = {
        "experiment_type": "unknown",
        "scenario_category": clean_name,
        "scenario_description": clean_name,
        "verbal_level": None,
        "facial_level": None,
        "specificity_level": None,
        "intensity_level": None,
    }

    # Parse disagreement pattern: v[hml]f[hml]
    if "disagreement" in experiment_name.lower():
        classification["experiment_type"] = "disagreement"
        if len(clean_name) == 4 and clean_name.startswith("v") and "f" in clean_name:
            verbal_code = clean_name[1]  # h, m, or n
            facial_code = clean_name[3]  # h, m, or n

            level_map = {"h": "high", "m": "mid", "n": "none"}
            classification["verbal_level"] = level_map.get(verbal_code, verbal_code)
            classification["facial_level"] = level_map.get(facial_code, facial_code)
            classification["scenario_description"] = (
                f"verbal_{classification['verbal_level']}_"
                f"facial_{classification['facial_level']}"
            )

    # Parse ambiguity patterns
    elif "ambiguity" in experiment_name.lower():
        classification["experiment_type"] = "ambiguity"
        if "specific" in clean_name:
            classification["specificity_level"] = "low"
        elif "moderate" in clean_name or "ambiguous" in clean_name:
            classification["specificity_level"] = "mid"
        elif "very" in clean_name or "vague" in clean_name:
            classification["specificity_level"] = "high"

    # Parse intensity patterns
    elif "intensity" in experiment_name.lower():
        classification["experiment_type"] = "intensity"
        if "mild" in clean_name:
            classification["intensity_level"] = "low"
        elif "moderate" in clean_name:
            classification["intensity_level"] = "mid"
        elif "severe" in clean_name or "extreme" in clean_name:
            classification["intensity_level"] = "high"

    # Parse modality patterns
    elif "modality" in experiment_name.lower():
        classification["experiment_type"] = "modality"

    # Parse uncertainty patterns
    elif "uncertainty" in experiment_name.lower():
        classification["experiment_type"] = "uncertainty"

    return classification


def merge_base_with_scenario(base_config: Dict, scenario: Dict) -> Dict:
    """Merge base config with scenario-specific config."""
    merged = base_config.copy()

    # Add received_feedback and labels from scenario
    merged["received_feedback"] = scenario["received_feedback"]
    if "labels" in scenario:
        merged["labels"] = scenario["labels"]

    return merged


def run_experiment(
    config_file: str,
    model: OpenAIModel,
    incremental_file: str = "incremental_results.jsonl",
    use_text_descriptions: bool = False,
    run_single_token: bool = True,
) -> Dict[str, Any]:
    """Run a single experiment from config file.

    Args:
        config_file: Path to experiment configuration file
        model: OpenAI model instance
        incremental_file: Path to save incremental results
        use_text_descriptions: If True, use text descriptions instead of images
        run_single_token: If True, run single token predictions; if False, only run full output
    """

    # Load experiment config
    experiment_config = load_experiment_configs(config_file)
    experiment_name = experiment_config["experiment_name"]

    print(f"\n=== {experiment_name} ===")
    print(f"Description: {experiment_config['description']}")

    # Load image-to-text mapping if using text descriptions
    img_to_text_map = None
    if use_text_descriptions:
        img_to_text_path = (
            Path(__file__).resolve().parent / "assets" / "img_to_text_map.json"
        )
        try:
            with open(img_to_text_path, "r", encoding="utf-8") as f:
                img_to_text_map = json.load(f)
            print(
                f"✅ Loaded image-to-text mapping with {len(img_to_text_map)} entries"
            )
        except Exception as e:
            print(f"🚨 ERROR: Failed to load image-to-text mapping: {e}")
            print("🚨 Falling back to regular image mode")
            use_text_descriptions = False

    results = {}
    base_config = experiment_config["base_config"]
    scenarios = experiment_config["scenarios"]

    # Collections for scenario classification and disaggregated metrics
    scenario_classifications = {}
    scenario_type_stats = {}
    scores_by_scenario_type: Dict[str, Dict[str, Any]] = {}

    for scenario_data in scenarios:
        scenario_name = scenario_data["name"]
        print(f"\nRunning scenario: {scenario_name}")

        # Classify scenario
        classification = classify_scenario(scenario_name, experiment_name)
        scenario_classifications[scenario_name] = classification

        # Track by scenario type
        category = classification["scenario_category"] or "unknown"
        if category not in scores_by_scenario_type:
            scores_by_scenario_type[category] = {
                "scenarios": [],
                "brier_scores_single": [],
                "brier_scores_full": [],
                "count": 0,
                "classification": classification,
            }

        # Merge base config with scenario
        full_config = merge_base_with_scenario(base_config, scenario_data)

        # Run analysis with incremental saving
        result = run_uncertainty_analysis(
            model,
            full_config,
            scenario_name,
            incremental_file,
            experiment_name,
            use_text_descriptions,
            img_to_text_map,
            run_single_token,
        )

        # Add expected values from config for comparison
        result["expected"] = {
            k: v
            for k, v in scenario_data.items()
            if k not in ["received_feedback", "name"]
        }

        results[scenario_name] = result

        # Aggregate by scenario type
        single_brier = result["brier_scores"].get("single_token_vs_labels", 0.0)
        full_brier = result["brier_scores"]["full_output_vs_labels"]

        scores_by_scenario_type[category]["scenarios"].append(scenario_name)
        scores_by_scenario_type[category]["brier_scores_single"].append(single_brier)
        scores_by_scenario_type[category]["brier_scores_full"].append(full_brier)
        scores_by_scenario_type[category]["count"] += 1

        # Print key metrics - focus on prediction accuracy first
        if "single_token_vs_labels" in result["brier_scores"]:
            print(f"  Brier score (single-token vs labels): {single_brier:.3f}")
        print(f"  Brier score (full output vs labels): {full_brier:.3f}")
        if result["single_token"]["probs"]:  # Only show if single token was run
            print(
                f"  Single-token uncertainty: {result['single_token']['uncertainty']:.3f}"
            )
            print(f"  Entropy: {result['single_token']['entropy']:.3f}")
        print(f"  Scenario type: {category}")

    # Calculate scenario type statistics
    for category, data in scores_by_scenario_type.items():
        if data["count"] > 0:
            scenario_type_stats[category] = {
                "avg_brier_single": float(np.mean(data["brier_scores_single"])),
                "avg_brier_full": float(np.mean(data["brier_scores_full"])),
                "count": data["count"],
                "scenarios": data["scenarios"],
                "classification": data["classification"],
            }

    return {
        "experiment_name": experiment_config["experiment_name"],
        "description": experiment_config["description"],
        "results": results,
        "scenario_classifications": scenario_classifications,
        "scenario_type_stats": scenario_type_stats,
        "num_scenarios": len(scenarios),
    }


def analyze_experiment_1_disagreement(results: Dict) -> Dict:
    """Analyze Q1: Speech vs facial expression disagreement using Brier scores."""
    analysis = {}

    # Collect Brier scores for all scenarios
    single_token_brier_scores = []
    full_output_brier_scores = []

    agreement_single_token = []
    disagreement_single_token = []
    agreement_full_output = []
    disagreement_full_output = []

    agreement_scenarios = ["agreement_comfortable", "agreement_uncomfortable"]
    disagreement_scenarios = [
        "disagreement_verbal_bad_face_good",
        "disagreement_verbal_good_face_bad",
    ]

    for scenario_name, result in results["results"].items():
        single_brier = result["brier_scores"].get("single_token_vs_labels", 0.0)
        full_brier = result["brier_scores"]["full_output_vs_labels"]

        single_token_brier_scores.append(single_brier)
        full_output_brier_scores.append(full_brier)

        if scenario_name in agreement_scenarios:
            agreement_single_token.append(single_brier)
            agreement_full_output.append(full_brier)
        elif scenario_name in disagreement_scenarios:
            disagreement_single_token.append(single_brier)
            disagreement_full_output.append(full_brier)

    analysis["avg_single_token_brier"] = np.mean(single_token_brier_scores)
    analysis["avg_full_output_brier"] = np.mean(full_output_brier_scores)
    analysis["agreement_single_token_brier"] = (
        np.mean(agreement_single_token) if agreement_single_token else 0
    )
    analysis["disagreement_single_token_brier"] = (
        np.mean(disagreement_single_token) if disagreement_single_token else 0
    )
    analysis["agreement_full_output_brier"] = (
        np.mean(agreement_full_output) if agreement_full_output else 0
    )
    analysis["disagreement_full_output_brier"] = (
        np.mean(disagreement_full_output) if disagreement_full_output else 0
    )

    # Higher Brier score means worse prediction accuracy
    analysis["disagreement_hurts_single_token"] = (
        analysis["disagreement_single_token_brier"]
        > analysis["agreement_single_token_brier"]
    )
    analysis["disagreement_hurts_full_output"] = (
        analysis["disagreement_full_output_brier"]
        > analysis["agreement_full_output_brier"]
    )

    return analysis


def analyze_experiment_2_ambiguity(results: Dict) -> Dict:
    """Analyze Q2: Location specificity ambiguity using Brier scores."""
    analysis = {}

    # Collect Brier scores for all scenarios
    single_token_brier_scores = []
    full_output_brier_scores = []

    for scenario_name, result in results["results"].items():
        single_brier = result["brier_scores"].get("single_token_vs_labels", 0.0)
        full_brier = result["brier_scores"]["full_output_vs_labels"]

        single_token_brier_scores.append(single_brier)
        full_output_brier_scores.append(full_brier)

        analysis[f"{scenario_name}_single_token_brier"] = single_brier
        analysis[f"{scenario_name}_full_output_brier"] = full_brier

    analysis["avg_single_token_brier"] = np.mean(single_token_brier_scores)
    analysis["avg_full_output_brier"] = np.mean(full_output_brier_scores)
    analysis["single_token_better_than_full"] = (
        analysis["avg_single_token_brier"] < analysis["avg_full_output_brier"]
    )

    return analysis


def analyze_experiment_3_intensity(results: Dict) -> Dict:
    """Analyze Q3: Adaptation to discomfort intensities using Brier scores."""
    analysis = {}

    # Collect Brier scores for all scenarios
    single_token_brier_scores = []
    full_output_brier_scores = []

    for scenario_name, result in results["results"].items():
        single_brier = result["brier_scores"].get("single_token_vs_labels", 0.0)
        full_brier = result["brier_scores"]["full_output_vs_labels"]

        single_token_brier_scores.append(single_brier)
        full_output_brier_scores.append(full_brier)

        analysis[f"{scenario_name}_single_token_brier"] = single_brier
        analysis[f"{scenario_name}_full_output_brier"] = full_brier

    analysis["avg_single_token_brier"] = np.mean(single_token_brier_scores)
    analysis["avg_full_output_brier"] = np.mean(full_output_brier_scores)
    analysis["single_token_better_than_full"] = (
        analysis["avg_single_token_brier"] < analysis["avg_full_output_brier"]
    )

    return analysis


def analyze_experiment_4_modality(results: Dict) -> Dict:
    """Analyze Q4: Image vs text modality comparison using Brier scores."""
    analysis = {}

    # Collect Brier scores for all scenarios
    single_token_brier_scores = []
    full_output_brier_scores = []

    # Group by modality type
    text_single_token = []
    text_full_output = []
    image_single_token = []
    image_full_output = []

    for scenario_name, result in results["results"].items():
        single_brier = result["brier_scores"].get("single_token_vs_labels", 0.0)
        full_brier = result["brier_scores"]["full_output_vs_labels"]

        single_token_brier_scores.append(single_brier)
        full_output_brier_scores.append(full_brier)

        analysis[f"{scenario_name}_single_token_brier"] = single_brier
        analysis[f"{scenario_name}_full_output_brier"] = full_brier

        # Group by modality if available
        if "modality_type" in result.get("expected", {}):
            if result["expected"]["modality_type"] == "text":
                text_single_token.append(single_brier)
                text_full_output.append(full_brier)
            elif result["expected"]["modality_type"] == "image":
                image_single_token.append(single_brier)
                image_full_output.append(full_brier)

    analysis["avg_single_token_brier"] = np.mean(single_token_brier_scores)
    analysis["avg_full_output_brier"] = np.mean(full_output_brier_scores)

    if text_single_token and image_single_token:
        analysis["text_single_token_brier"] = np.mean(text_single_token)
        analysis["image_single_token_brier"] = np.mean(image_single_token)
        analysis["text_better_than_image_single"] = (
            analysis["text_single_token_brier"] < analysis["image_single_token_brier"]
        )

    if text_full_output and image_full_output:
        analysis["text_full_output_brier"] = np.mean(text_full_output)
        analysis["image_full_output_brier"] = np.mean(image_full_output)
        analysis["text_better_than_image_full"] = (
            analysis["text_full_output_brier"] < analysis["image_full_output_brier"]
        )

    return analysis


def analyze_experiment_5_uncertainty(results: Dict) -> Dict:
    """Analyze Q5: Uncertainty vs accuracy (token-based only)."""
    analysis = {}

    # Collect Brier scores and uncertainty metrics for comparison
    single_token_brier_scores = []
    full_output_brier_scores = []
    single_token_uncertainties = []

    for scenario_name, result in results["results"].items():
        single_brier = result["brier_scores"].get("single_token_vs_labels", 0.0)
        full_brier = result["brier_scores"]["full_output_vs_labels"]
        single_unc = result["single_token"]["uncertainty"]

        single_token_brier_scores.append(single_brier)
        full_output_brier_scores.append(full_brier)
        single_token_uncertainties.append(single_unc)

        analysis[f"{scenario_name}_single_token_brier"] = single_brier
        analysis[f"{scenario_name}_full_output_brier"] = full_brier

    # Primary analysis: prediction accuracy
    analysis["avg_single_token_brier"] = np.mean(single_token_brier_scores)
    analysis["avg_full_output_brier"] = np.mean(full_output_brier_scores)
    analysis["single_token_better_than_full"] = (
        analysis["avg_single_token_brier"] < analysis["avg_full_output_brier"]
    )

    # Secondary analysis: uncertainty correlation with prediction accuracy
    # Higher uncertainty should correlate with higher Brier scores (worse predictions)
    uncertainty_brier_correlation = (
        np.corrcoef(single_token_uncertainties, single_token_brier_scores)[0, 1]
        if len(single_token_uncertainties) > 1
        else 0
    )
    analysis["uncertainty_predicts_accuracy"] = uncertainty_brier_correlation

    return analysis


def main(use_text_descriptions: bool = False, run_single_token: bool = True):
    """Run all experiments and generate comprehensive analysis.

    Args:
        use_text_descriptions: If True, use text descriptions instead of images
        run_single_token: If True, run single token predictions; if False, only run full output
    """

    # Initialize model
    model = OpenAIModel(model="gpt-5", system_prompt=get_system_prompt())

    print(
        f"🔬 Running experiments with "
        f"{'text descriptions' if use_text_descriptions else 'images'} "
        f"for facial expressions"
    )
    if not run_single_token:
        print("⚡ Running in full-output-only mode (skipping single token predictions)")
    else:
        print("🎯 Running both single token and full output predictions")

    # Experiment configurations
    experiments = [
        "experiments/configs/experiment_1_disagreement.json",
        "experiments/configs/experiment_2_ambiguity.json",
        "experiments/configs/experiment_3_intensity.json",
        "experiments/configs/experiment_4_modality.json",
        "experiments/configs/experiment_5_uncertainty.json",
    ]

    curr_time = datetime.now().strftime("%m-%d-%H:%M")
    all_results = {}
    analyses = {}
    incremental_file = f"{curr_time}/incremental_results.jsonl"

    # Run each experiment
    for i, config_file in enumerate(experiments, 1):
        try:
            experiment_results = run_experiment(
                config_file,
                model,
                incremental_file,
                use_text_descriptions,
                run_single_token,
            )
            experiment_name = f"experiment_{i}"
            all_results[experiment_name] = experiment_results

            # Run specific analysis for each experiment
            if i == 1:
                analyses[experiment_name] = analyze_experiment_1_disagreement(
                    experiment_results
                )
            elif i == 2:
                analyses[experiment_name] = analyze_experiment_2_ambiguity(
                    experiment_results
                )
            elif i == 3:
                analyses[experiment_name] = analyze_experiment_3_intensity(
                    experiment_results
                )
            elif i == 4:
                analyses[experiment_name] = analyze_experiment_4_modality(
                    experiment_results
                )
            elif i == 5:
                analyses[experiment_name] = analyze_experiment_5_uncertainty(
                    experiment_results
                )

        except Exception as e:
            print(f"Error running {config_file}: {e}")
            continue

    # Print summary analyses
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY ANALYSES")
    print("=" * 60)

    for exp_name, analysis in analyses.items():
        print(f"\n{exp_name.upper()}:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

    # Save detailed results automatically
    output_data = {
        "raw_results": all_results,
        "analyses": analyses,
        "experiment_config": {
            "use_text_descriptions": use_text_descriptions,
            "model_name": model.model,
            "timestamp": datetime.now().isoformat(),
        },
    }
    suffix = "_text_desc" if use_text_descriptions else ""
    final_results_file = f"{curr_time}/final_results{suffix}.json"

    with open(final_results_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {final_results_file}")
    print(f"Incremental results saved to: {incremental_file}")

    # Also save legacy comprehensive_results.json for backward compatibility
    with open(
        "/Users/eric/conformal/conformal_setup/experiments/comprehensive_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output_data, f, indent=2, default=str)

    return output_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run feedback interpretation experiments"
    )
    parser.add_argument(
        "--use-text-descriptions",
        action="store_true",
        help="Use text descriptions instead of images for facial expressions",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Run only full output predictions, skip single token predictions (faster)",
    )

    args = parser.parse_args()
    main(
        use_text_descriptions=args.use_text_descriptions,
        run_single_token=not args.full_only,
    )
